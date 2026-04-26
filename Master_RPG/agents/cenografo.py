from state import GameState
from llm import get_llm
import re

def cenografo_node(state: GameState) -> GameState:
    """
    Agente 2: Cria ambiente cinematográfico. Descreve o ambiente, o que tem à disposição e quem está perto.
    """
    print("--> [Agente Cenógrafo] Descrevendo o ambiente...")
    llm = get_llm()
    
    char_name = state.get("char_name", "Aventureiro")
    lore = state.get("world_lore", "")
    current_loc = state.get("current_location", "um local desconhecido")
    
    scene_npcs_data = []
    if state.get("current_scene_npcs"):
        for npc_id in state["current_scene_npcs"]:
            if npc_id in state.get("active_npcs", {}):
                n_data = state["active_npcs"][npc_id]
                clean_name = re.sub(r'\(.*?\)', '', n_data['nome']).strip()
                scene_npcs_data.append(f"- {clean_name} ({n_data['tipo']}): {n_data['historia']}")
    
    npcs_str = "\n".join(scene_npcs_data)
    
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
    - Narre o ambiente (clima, sons, cheiros) de forma breve.
    - Narre a presença e comportamento dos NPCs listados. (NUNCA cite os metadados técnicos como "(npc_02)", use apenas os nomes de forma natural).
    - INCIDENTE INCITANTE: Imediatamente introduza a TRAMA PRINCIPAL ou a QUEST INICIAL. Algo perigoso e urgente DEVE acontecer e ATINGIR DIRETAMENTE o herói {char_name}. O herói NÃO PODE ser um mero espectador; o perigo ou mistério deve envolvê-lo de forma INEVITÁVEL (ex: um artefato misterioso cai em suas mãos, ele é atacado de surpresa, a magia atinge a sua mesa, ou o alvo principal olha diretamente para ele exigindo algo).
    - O GANCHO: O encerramento DEVE forçar {char_name} a tomar uma decisão de sobrevivência ou ação imediata. Termine com "O que você faz?"
    
    REGRA DE VOZ DOS NPCs (DETERMINÍSTICA):
    Sempre que QUALQUER personagem falar (NPCs, monstros, criaturas), você DEVE colocar a tag da voz imediatamente ANTES da abertura das aspas. 
    - Aliados/Neutros: pt-PT-DuarteNeural (Masc) ou pt-BR-ThalitaNeural (Fem)
    - Inimigos/Monstros (mesmo sem gênero definido): pt-PT-DuarteNeural (Voz Grave/Masc) ou pt-PT-RaquelNeural (Fem)
    - NUNCA use pt-BR-AntonioNeural para NPCs. NENHUMA fala pode ficar sem a tag [VOICE:...].
    EXEMPLO OBRIGATÓRIO: [VOICE:pt-PT-DuarteNeural] "Encontrei você!" sussurra a criatura sombria.
    """
    
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response.content})
    state["next_node"] = "END"
    return state
