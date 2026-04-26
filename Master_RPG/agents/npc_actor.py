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
            "historia": "Alguém que o jogador acabou de encontrar.",
            "voz_escolhida": "pt-BR-ThalitaMultilingualNeural"
        }
        
    recent_msgs = state.get("messages", [])[-3:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Você é o NPC: {npc_data.get('nome')}.
    
    Sua ficha:
    - Objetivo: {npc_data.get('objetivo')}
    - História: {npc_data.get('historia')}
    
    Histórico recente: {context}
    O Herói ({state.get('char_name')}) diz agora: "{state.get('current_input')}"
    
    REGRA MÁXIMA ABSOLUTA (NUNCA VIOLAR):
    - NÃO narre nada que o herói {state.get('char_name')} faça ou diga.
    - É EXPRESSAMENTE PROIBIDO inventar falas, movimentos ou sentimentos para o jogador.
    - EXEMPLO RUIM: "[VOICE:...] "Olá!" - diz o NPC enquanto Findariel sorri." (PROIBIDO)
    - EXEMPLO BOM: "[VOICE:...] "Olá!" - diz o NPC, mantendo o olhar fixo em {state.get('char_name')}." (CORRETO)
    
    TAREFA:
    1. FALA DO NPC: Use aspas ("...") e coloque a tag [VOICE:{npc_data.get('voz_escolhida')}] imediatamente ANTES da abertura das aspas. Exemplo: [VOICE:{npc_data.get('voz_escolhida')}] "Fala..."
    2. REAÇÃO FÍSICA DO NPC: Descreva o que o NPC faz em terceira pessoa.
    3. GANCHO: Termine com "O que você faz?" ou "O que você responde?"
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
