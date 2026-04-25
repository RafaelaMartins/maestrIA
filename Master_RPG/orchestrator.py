from langgraph.graph import StateGraph, END
from state import GameState
from llm import get_llm
import json
import re

# Import agents
from agents.roteirista import roteirista_node
from agents.cenografo import cenografo_node
from agents.npc_creator import npc_creator_node
from agents.evaluators import avaliador_acoes_node, avaliador_testes_node, mestre_geral_node

def orquestrador_router(state: GameState) -> GameState:
    """
    Agente 8: Orquestrador. Analisa o input do usuário e decide o fluxo.
    Verifica se precisa de rolagem de dados (Agente 6 vs 7).
    """
    if not state.get("current_input"):
        return state
        
    print("--> [Orquestrador] Analisando intenção do jogador...")
    llm = get_llm()
    
    prompt = f"""
    Você é o Orquestrador do RPG.
    Ação do jogador: "{state.get('current_input')}"
    
    A ação requer um teste de dados (ex: atacar, pular, mentir) ou é uma ação narrativa simples (ex: olhar, andar, falar com alguém)?
    Retorne APENAS um JSON:
    {{
        "requires_roll": true/false,
        "skill": "Perícia se precisar",
        "dc": 10
    }}
    """
    
    response = llm.invoke(prompt)
    
    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if data.get("requires_roll"):
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

def build_graph():
    builder = StateGraph(GameState)
    
    # Adicionando os nós
    builder.add_node("roteirista", roteirista_node)
    builder.add_node("orquestrador", orquestrador_router)
    builder.add_node("avaliador_acoes", avaliador_acoes_node)
    builder.add_node("avaliador_testes", avaliador_testes_node)
    builder.add_node("cenografo", cenografo_node)
    builder.add_node("mestre_geral", mestre_geral_node)
    builder.add_node("npc_creator", npc_creator_node)
    
    # Fluxo Base: Todo turno começa no orquestrador (se houver input)
    builder.set_entry_point("roteirista") # O roteirista roda primeiro para setup inicial
    
    # Do roteirista, vai para o orquestrador para processar a ação
    builder.add_edge("roteirista", "orquestrador")
    
    # Do orquestrador, usa roteamento condicional
    builder.add_conditional_edges("orquestrador", router_condition)
    
    # As ações terminam no END para devolver resposta ao usuário no Streamlit
    builder.add_edge("avaliador_acoes", END)
    builder.add_edge("avaliador_testes", END)
    builder.add_edge("cenografo", END)
    builder.add_edge("mestre_geral", END)
    builder.add_edge("npc_creator", END)
    
    return builder.compile()
