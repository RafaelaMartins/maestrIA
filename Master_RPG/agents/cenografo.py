from state import GameState
from llm import get_llm

def cenografo_node(state: GameState) -> GameState:
    """
    Agente 2: Cria ambiente cinematográfico. Descreve o ambiente, o que tem à disposição e quem está perto.
    """
    print("--> [Agente Cenógrafo] Descrevendo o ambiente...")
    llm = get_llm()
    
    char_name = state.get("char_name", "Aventureiro")
    lore = state.get("world_lore", "")
    current_loc = state.get("current_location", "um local desconhecido")
    
    # Busca os detalhes dos NPCs presentes na cena
    scene_npcs_data = []
    if state.get("current_scene_npcs"):
        for npc_id in state["current_scene_npcs"]:
            if npc_id in state.get("active_npcs", {}):
                scene_npcs_data.append(state["active_npcs"][npc_id])
    
    npcs_str = "\n".join([f"- {n['nome']} ({n['tipo']}): {n['historia']}" for n in scene_npcs_data])
    
    prompt = f"""
    Você é o Cenógrafo e Narrador do RPG.
    Local: {current_loc}
    Lore do Mundo: {lore}
    NPCs PRESENTES: {npcs_str}
    
    DIRETRIZ DE NARRATIVA: Sua missão é conduzir o herói {char_name} para a QUEST PRINCIPAL.
    
    REGRA MÁXIMA ABSOLUTA (SISTEMA DE SEGURANÇA):
    1. VOCÊ É O MUNDO, NÃO O JOGADOR. É terminantemente PROIBIDO narrar qualquer ação, fala, pensamento ou sentimento de {char_name}.
    2. NÃO use verbos onde {char_name} seja o sujeito (Ex: "Você percebe", "Findariel entra", "Findariel diz", "Findariel sente").
    3. FOQUE 100% NAS REAÇÕES DO AMBIENTE E DOS NPCs.
       - EXEMPLO RUIM (PROIBIDO): "Findariel entra na taverna e pergunta ao barman onde fica a loja."
       - EXEMPLO BOM (CORRETO): "A taverna está barulhenta. O barman limpa um copo e olha na direção de {char_name}, aguardando uma palavra."
    4. NUNCA invente diálogos para {char_name}. Se {char_name} quiser falar, o JOGADOR escreverá isso no chat.
    
    TAREFA:
    - Narre o ambiente (clima, sons, cheiros).
    - Narre a presença e comportamento dos NPCs listados.
    - O GANCHO: O encerramento DEVE ser um evento externo que force {char_name} a agir. Termine com "O que você faz?"
    
    REGRA DE DIÁLOGO OBRIGATÓRIA: Use aspas duplas ("...") e tags [VOICE:...] para NPCs.
    """
    
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response.content})
    state["next_node"] = "END"
    return state
