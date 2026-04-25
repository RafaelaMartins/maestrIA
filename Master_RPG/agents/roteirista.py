from state import GameState
from llm import get_llm

def roteirista_node(state: GameState) -> GameState:
    """
    Agente 1: Cria a sinopse da história, mundo, locais, cronologia e objetivo final.
    Roda no início para popular o 'world_lore' e 'main_goal'.
    """
    if state.get("world_lore"):
        return state # Already initialized
        
    print("--> [Agente Roteirista] Criando o mundo...")
    llm = get_llm()
    
    prompt = f"""
    Você é o Roteirista Chefe de um RPG de mesa.
    Crie um mundo sucinto, cronologia básica e um objetivo final épico para um herói chamado {state.get('char_name', 'Aventureiro')}.
    
    Formato desejado:
    MUNDO: [Nome e descrição]
    LOCAIS: [2 ou 3 locais chave]
    CRONOLOGIA: [O que aconteceu recentemente]
    OBJETIVO FINAL: [O grande desafio]
    """
    
    response = llm.invoke(prompt)
    
    # Store in hidden state
    state["world_lore"] = response.content
    state["main_goal"] = "Completar a jornada estabelecida pelo roteirista."
    
    return state
