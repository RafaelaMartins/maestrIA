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
    
    prompt = f"""
    Você é o Agente Cenógrafo do RPG.
    Mundo: {lore}
    O personagem {char_name} acaba de entrar em: {current_loc}.
    
    Descreva o ambiente de forma cinematográfica, épica e profunda. Dê vida ao cenário com detalhes únicos (iluminação, cheiros, atmosfera densa).
    
    Elementos Obrigatórios:
    - O que {char_name} vê, ouve e sente de forma vívida.
    - Figuras INTRIGANTES e ESPECÍFICAS presentes no local (ex: não descreva uma multidão genérica, descreva "uma guerreira de armadura rachada contando moedas", "um velho cego que parece saber demais" ou "uma feiticeira misteriosa que entra repentinamente").
    
    IMPORTANTE - O GANCHO: O encerramento da sua narrativa deve SEMPRE focar em um EVENTO ESPECÍFICO E INTENSO que exija uma reação de {char_name}. Não use ganchos genéricos como "ouviu-se um barulho". Faça com que alguém interaja com ele, ou que um evento dramático comece na sua frente (ex: uma briga de espadas se inicia, a feiticeira misteriosa o encara e fala um enigma, etc).
    
    REGRA DE IMERSÃO: NUNCA use as palavras "jogador", "usuário", "personagens" ou "NPCs". Refira-se a ele sempre pelo nome ({char_name}) ou pronomes.
    ATENÇÃO: A SUA RESPOSTA INTEIRA DEVE SER ESCRITA EM PORTUGUÊS DO BRASIL. NÃO USE NENHUMA PALAVRA EM INGLÊS (COMO "SUDDENLY").
    
    Responda diretamente com a narração (sem metalinguagem).
    """
    
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response.content})
    state["next_node"] = "END" # Returns to user
    return state
