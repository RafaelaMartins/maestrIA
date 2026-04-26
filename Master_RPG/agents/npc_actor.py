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
    
    Instruções Rigorosas (Interpretação + Narração):
    1. LÓGICA DO DIÁLOGO: Se no histórico você já ofereceu uma informação, não se repita. Reaja à última fala do jogador.
    2. A FALA DO NPC (PRIMEIRA PESSOA): A fala do NPC deve estar ESTRITAMENTE entre aspas duplas ("...") e PRECEDIDA pela tag de voz. Use a tag: [VOICE:{npc_data.get('voz_escolhida', 'Thalita')}] antes da abertura das aspas. Nunca use travessão (-).
       - Exemplo: [VOICE:{npc_data.get('voz_escolhida', 'Thalita')}]"Chamem-me de Gilded, jovem."
    3. A NARRAÇÃO DO MESTRE (TERCEIRA PESSOA): FORA DAS ASPAS (e sem tags), você atua como o Mestre Narrador. Descreva o que o NPC faz fisicamente (em terceira pessoa) e como o ambiente reage.
       - Exemplo: O homem dourado sorri e sai pela porta da taverna.
    4. O GANCHO (NARRADOR): Como Narrador, conclua a cena com um objetivo claro. Termine SEMPRE com a pergunta explícita: "O que você faz?" ou "O que você responde?"
    5. REGRA DE OURO: VOCÊ ESTÁ PROIBIDO de escrever os sentimentos, falas ou ações do herói {state.get('char_name')}. NUNCA jogue por ele!
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
