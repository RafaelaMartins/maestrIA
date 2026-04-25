from state import GameState
from llm import get_llm
import json

def npc_actor_node(state: GameState) -> GameState:
    """
    Agente 3.5: Character Agent (Ator de NPC).
    Interpreta um NPC específico durante um diálogo com o jogador.
    """
    print(f"--> [Ator de NPC] Interpretando {state.get('interacting_npc')}...")
    llm = get_llm()
    
    npc_name = state.get("interacting_npc")
    npc_data = None
    
    # Busca o NPC pelo nome nos npcs ativos
    if npc_name and "active_npcs" in state:
        for npc_id, data in state["active_npcs"].items():
            if data.get("nome", "").lower() == npc_name.lower():
                npc_data = data
                break
                
    if not npc_data:
        # Se não encontrou dados estruturados, cria um improvisado com base no nome
        npc_data = {
            "nome": npc_name,
            "tipo": "Desconhecido",
            "dificuldade": "Médio",
            "objetivo": "Desconhecido",
            "pontos_fortes": "Misterioso",
            "pontos_fracos": "Desconfiado",
            "historia": "Alguém que o jogador acabou de encontrar."
        }
        
    recent_msgs = state.get("messages", [])[-3:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Você é o NPC: {npc_data.get('nome')}.
    
    Sua ficha:
    - Tipo: {npc_data.get('tipo')}
    - Objetivo: {npc_data.get('objetivo')}
    - Pontos Fortes: {npc_data.get('pontos_fortes')}
    - Pontos Fracos: {npc_data.get('pontos_fracos')}
    - História: {npc_data.get('historia')}
    
    Histórico recente da cena (leia com atenção para saber quem falou o quê):
    {context}
    
    O Herói ({state.get('char_name')}) diz/faz agora: "{state.get('current_input')}"
    
    Instruções Rigorosas:
    1. Responda em primeira pessoa COMO O {npc_data.get('nome')}. 
    2. MANTENHA A LÓGICA DO DIÁLOGO. Se no histórico recente VOCÊ ofereceu uma informação (ex: "Eu sei onde fica o castelo"), não pergunte ao jogador onde fica o castelo na sua próxima fala. Lembre-se do que você já disse!
    3. NÃO descreva seus pensamentos internos. Descreva APENAS a sua fala e a sua ação física imediata (EXEMPLO DE FORMATO: "Bato o copo na mesa e encaro você. - O que quer aqui?").
    4. IDIOMA E GRAMÁTICA: Responda 100% em Português do Brasil. Conjugue os verbos na primeira pessoa corretamente (ex: "Eu movo", não "Eu move").
    5. REGRA DE OURO ABSOLUTA: VOCÊ É APENAS O NPC. VOCÊ ESTÁ ESTRITAMENTE PROIBIDO de narrar os sentimentos, as ações, as falas ou a presença de {state.get('char_name')}. Pare a sua resposta IMEDIATAMENTE após a sua própria fala. NUNCA ESCREVA O QUE O HERÓI FAZ!
    """
    
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    
    # Formata para ficar claro que é o NPC falando
    formatted_response = f"**{npc_data.get('nome')}**: {response.content}"
    state["messages"].append({"role": "assistant", "content": formatted_response})
    
    state["interacting_npc"] = None # Reset interaction
    state["next_node"] = "END"
    return state
