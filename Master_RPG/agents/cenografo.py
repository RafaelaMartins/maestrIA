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
    
    Descreva cinematograficamente o ambiente. Inclua:
    - O que o personagem vê, ouve e sente.
    - Objetos à disposição.
    - Pessoas ou criaturas próximas (sem dar nomes específicos ainda, deixe genérico para o NPC Creator).
    
    Responda diretamente com a narração (sem metalinguagem).
    """
    
    response = llm.invoke(prompt)
    
    state["messages"] = [{"role": "assistant", "content": response.content}]
    state["next_node"] = "END" # Returns to user
    return state
