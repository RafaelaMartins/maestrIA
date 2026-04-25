from state import GameState
from llm import get_llm
import json
import re

def diretor_combate_node(state: GameState) -> GameState:
    """
    Agente 4: Decide o momento de combate, chama criação de NPC de combate se necessário.
    """
    print("--> [Diretor de Combate] Analisando tensão...")
    # Lógica simplificada: Se a tensão for alta, define um estado para combate.
    state["next_node"] = "npc_creator"
    return state

def mestre_geral_node(state: GameState) -> GameState:
    """
    Agente 5: Decide o tipo da rodada (Diálogo, Exploração, Combate).
    """
    print("--> [Mestre Geral] Conduzindo a cena...")
    llm = get_llm()
    prompt = f"O jogador fez a ação: {state.get('current_input')}. Narre a consequência e pergunte o que ele faz."
    response = llm.invoke(prompt)
    
    state["messages"] = [{"role": "assistant", "content": response.content}]
    state["next_node"] = "END"
    return state

def avaliador_acoes_node(state: GameState) -> GameState:
    """
    Agente 6: Avalia ações sem rolagem e verifica consequência narrativa.
    """
    print("--> [Avaliador de Ações] Avaliando ação trivial...")
    llm = get_llm()
    prompt = f"""
    O personagem tenta: {state.get('current_input')}. 
    Esta é uma ação que não requer dados. Narre o que acontece, mantendo a coerência com a história.
    """
    response = llm.invoke(prompt)
    
    state["messages"] = [{"role": "assistant", "content": response.content}]
    state["next_node"] = "END"
    return state

def avaliador_testes_node(state: GameState) -> GameState:
    """
    Agente 7: Avalia ações com rolagem de dados (teste de proficiência).
    """
    print("--> [Avaliador de Testes] Resolvendo rolagem...")
    
    # Se o dado ainda não foi rolado, avisa a UI para pedir rolagem
    if not state.get("roll_result"):
        # Solicitar rolagem via UI
        msg = f"A ação requer um teste de {state['roll_details']['skill']} (CD {state['roll_details']['dc']}). Role o dado!"
        state["messages"] = [{"role": "assistant", "content": msg}]
        state["next_node"] = "END"
        return state
        
    # Se o dado FOI rolado
    llm = get_llm()
    roll = state["roll_result"]["total"]
    dc = state["roll_details"]["dc"]
    sucesso = roll >= dc
    
    resultado_str = "SUCESSO" if sucesso else "FALHA"
    
    prompt = f"""
    O personagem tentou uma ação e rolou {roll} contra CD {dc}. Resultado: {resultado_str}.
    Narre as consequências de forma épica.
    """
    response = llm.invoke(prompt)
    
    state["messages"] = [{"role": "assistant", "content": f"🎲 **Resultado:** {roll} vs CD {dc}\n\n{response.content}"}]
    state["requires_roll"] = False # Reset
    state["roll_result"] = None
    state["next_node"] = "END"
    return state
