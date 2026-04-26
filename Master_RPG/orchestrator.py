from langgraph.graph import StateGraph, END
from state import GameState
from llm import get_llm
import json
import re

# Import agents
from agents.npc_creator import npc_creator_node, populador_cena_node
from agents.npc_actor import npc_actor_node
from agents.evaluators import avaliador_acoes_node, avaliador_testes_node, mestre_geral_node
from agents.roteirista import roteirista_node, world_updater_node
from agents.cenografo import cenografo_node
from agents.guardrails import guardrail_autoplay_node

def orquestrador_router(state: GameState) -> GameState:
    """
    Agente 8: Orquestrador. Analisa o input do usuário e decide o fluxo.
    Verifica se precisa de rolagem de dados (Agente 6 vs 7).
    """
    if not state.get("current_input"):
        return state
        
    if state.get("current_input") == "INTRO_START":
        print("--> [Orquestrador] Iniciando aventura via Cenógrafo...")
        state["requires_roll"] = False
        state["next_node"] = "cenografo"
        return state
        
    print("--> [Orquestrador] Analisando intenção do jogador...")
    llm = get_llm()
    
    prompt = f"""
    Você é o Orquestrador do RPG.
    Ação do jogador: "{state.get('current_input')}"
    
    REGRAS DO SISTEMA (Tormenta Adaptado - D10):
    1. Determinar Nível da Tarefa:
       - Tarefa Fácil: CD 5
       - Tarefa Média: CD 7 ou 8
       - Tarefa Difícil: CD 9 ou 10
    2. Determinar a Perícia Relevante:
       - Lábia / Persuasão: Para tirar informações, mentir, convencer.
       - Adivinhação: Para ler mentes e detectar enganos.
       - Armas Brancas: Para lutar com espadas, machados, etc.
       - Conhecimento de X: Para responder perguntas.
       - Outras aplicáveis conforme o bom senso.
    
    PERGUNTE A SI MESMO:
    1. O jogador está falando EM VOZ ALTA com um NPC (ex: "Olá taverneiro" ou usando aspas "")? Se sim, is_dialogue = true.
    2. O jogador está declarando uma INTENÇÃO MECÂNICA ao mestre (ex: "Quero usar persuasão", "Faço um teste de força")? Se sim, ISSO NÃO É DIÁLOGO. is_dialogue = false, requires_roll = true.
    3. É apenas uma ação de exploração simples sem dados?
    
    Retorne APENAS um JSON:
    {{
        "is_dialogue": true/false,
        "npc_name": "Nome do NPC se houver, ou vazio",
        "requires_roll": true/false,
        "skill": "Perícia se precisar de dado",
        "dc": 10
    }}
    """
    
    response = llm.invoke(prompt)
    
    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if data.get("is_dialogue") and data.get("npc_name"):
                state["interacting_npc"] = data.get("npc_name")
                state["requires_roll"] = False
                
                # Check if NPC exists, if not route to creator first
                npc_exists = False
                if "active_npcs" in state:
                    for npc_id, npc_data in state["active_npcs"].items():
                        if npc_data.get("nome", "").lower() == state["interacting_npc"].lower():
                            npc_exists = True
                            break
                            
                if not npc_exists:
                    state["next_node"] = "npc_creator" # Creator will create and route to actor
                else:
                    state["next_node"] = "npc_actor"
                    
            elif data.get("requires_roll"):
                state["requires_roll"] = True
                state["roll_details"] = {"skill": data.get("skill", "Geral"), "dc": data.get("dc", 10)}
                state["next_node"] = "avaliador_testes"
            else:
                state["requires_roll"] = False
                state["next_node"] = "avaliador_acoes"
    except Exception:
        # Fallback
        state["requires_roll"] = False
        state["next_node"] = "avaliador_acoes"
        
    return state

def router_condition(state: GameState) -> str:
    # Retorna o nome do próximo nó com base no estado do orquestrador
    node = state.get("next_node", "mestre_geral")
    if node == "END":
        return END
    return node

def finalize_turn(state: GameState) -> str:
    # Agente do Mundo: Atualiza o mundo a cada 5 turnos
    if state.get("turn_count", 0) > 0 and state.get("turn_count", 0) % 5 == 0:
        return "world_updater"
    return "guardrail_autoplay"

def build_graph():
    builder = StateGraph(GameState)
    
    # Adicionando os nós
    builder.add_node("roteirista", roteirista_node)
    builder.add_node("populador_cena", populador_cena_node)
    builder.add_node("orquestrador", orquestrador_router)
    builder.add_node("avaliador_acoes", avaliador_acoes_node)
    builder.add_node("avaliador_testes", avaliador_testes_node)
    builder.add_node("cenografo", cenografo_node)
    builder.add_node("mestre_geral", mestre_geral_node)
    builder.add_node("npc_creator", npc_creator_node)
    builder.add_node("npc_actor", npc_actor_node)
    builder.add_node("guardrail_autoplay", guardrail_autoplay_node)
    builder.add_node("world_updater", world_updater_node)
    
    # Fluxo Base
    builder.set_entry_point("roteirista")
    
    builder.add_edge("roteirista", "populador_cena")
    builder.add_edge("populador_cena", "orquestrador")
    builder.add_conditional_edges("orquestrador", router_condition)
    
    # Routing out of creation nodes
    builder.add_edge("npc_creator", "npc_actor") # NPC creator must go to Actor
    
    # As ações terminam no conditional edge para atualizar o mundo
    builder.add_conditional_edges("avaliador_acoes", finalize_turn)
    builder.add_conditional_edges("avaliador_testes", finalize_turn)
    builder.add_conditional_edges("cenografo", finalize_turn)
    builder.add_conditional_edges("mestre_geral", finalize_turn)
    builder.add_conditional_edges("npc_actor", finalize_turn)
    
    builder.add_edge("world_updater", END)
    
    return builder.compile()
