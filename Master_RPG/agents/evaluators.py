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
    
    # Extrair contexto recente (últimas 3 mensagens)
    recent_msgs = state.get("messages", [])[-3:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Contexto da cena: {context}
    
    O personagem {state.get('char_name')} fez a ação: {state.get('current_input')}. 
    Narre a consequência imediata.
    
    IMPORTANTE - O GANCHO: O encerramento da sua narrativa deve SEMPRE focar em um EVENTO ATIVO, UMA PISTA CLARA ou UMA AMEAÇA que puxe a história para frente. Não encerre de forma passiva (ex: "ele se pergunta o que fazer"). Faça algo acontecer no mundo que exija uma reação imediata de {state.get('char_name')} (ex: alguém esbarra nele na rua com um mapa, ele ouve um grito, ou percebe que está sendo seguido).
    
    REGRA DE IMERSÃO: NUNCA use a palavra "jogador" ou "usuário" no seu texto. Chame-o sempre de {state.get('char_name')}.
    REGRA DE OURO ABSOLUTA: VOCÊ É O MESTRE, NÃO O JOGADOR. VOCÊ ESTÁ ESTRITAMENTE PROIBIDO de escrever pensamentos, emoções, reações físicas ou falas do personagem {state.get('char_name')}. Pare a sua narrativa assim que o mundo ou NPC reagir. NUNCA ESCREVA POR ELE!
    """
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response.content})
    state["next_node"] = "END"
    return state

def avaliador_acoes_node(state: GameState) -> GameState:
    """
    Agente 6: Avalia ações sem rolagem e verifica consequência narrativa.
    """
    print("--> [Avaliador de Ações] Avaliando ação trivial...")
    llm = get_llm()
    
    recent_msgs = state.get("messages", [])[-3:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Contexto da cena:
    {context}
    
    O personagem {state.get('char_name')} tenta: {state.get('current_input')}. 
    Esta é uma ação que não requer dados. Narre o que acontece com base APENAS nisso e no cenário. 
    Mantenha a coerência. Não invente batalhas do nada se não for o caso.
    
    IMPORTANTE - O GANCHO: Nunca deixe a cena "morta" ou passiva. O encerramento da sua narrativa deve SEMPRE apresentar um elemento novo e intrigante no ambiente que direcione {state.get('char_name')} para o seu objetivo ou para um mistério (ex: ele repara em uma placa suspeita, encontra um item brilhante no chão, ou alguém bloqueia o seu caminho).
    
    REGRA DE IMERSÃO: NUNCA use a palavra "jogador" ou "usuário" no seu texto. Chame-o sempre de {state.get('char_name')}.
    REGRA DE OURO ABSOLUTA: VOCÊ É O MESTRE, NÃO O JOGADOR. VOCÊ ESTÁ ESTRITAMENTE PROIBIDO de escrever pensamentos, emoções, reações físicas ou falas do personagem {state.get('char_name')}. Pare a sua narrativa assim que o mundo ou NPC reagir à ação dele. NUNCA ESCREVA POR ELE!
    """
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response.content})
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
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"].append({"role": "assistant", "content": msg})
        state["next_node"] = "END"
        return state
        
    # Se o dado FOI rolado
    llm = get_llm()
    roll = state["roll_result"]["total"]
    dc = state["roll_details"]["dc"]
    diff = roll - dc
    
    if diff >= 3:
        resultado_str = "SUCESSO CRÍTICO/ESPETACULAR"
    elif diff >= 0:
        resultado_str = "SUCESSO"
    elif diff >= -3:
        resultado_str = "FALHA SIMPLES"
    else:
        resultado_str = "FALHA CRÍTICA/DESASTROSA"
        
    recent_msgs = state.get("messages", [])[-3:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Contexto da cena: 
    {context}
    
    Ação do jogador que gerou esse teste: "{state.get('current_input')}"
    
    Nível de Sucesso/Falha: {resultado_str}.
    
    Instruções para a Narração (CINEMATOGRÁFICA E IMERSIVA):
    1. NÃO FOQUE APENAS NO HERÓI: Descreva também o que os inimigos ou alvos estão fazendo. Como eles reagem? Eles atacam de volta? Eles gritam de dor?
    2. AMBIENTE E NPCs: Inclua o cenário ao redor. Cadeiras quebram? As pessoas gritam e fogem? Algum aliado ou NPC reage à sua ação? Deixe a cena viva!
    3. NÃO mencione os números do dado, a Dificuldade (CD), nem termos sistêmicos como "Falha Crítica" no seu texto. Concentre-se APENAS na ficção.
       - Se for SUCESSO CRÍTICO: Narre algo lendário, perfeito, com consequências incríveis no ambiente.
       - Se for SUCESSO: Narre ele conseguindo o que queria com competência e o impacto imediato no inimigo.
       - Se for FALHA SIMPLES: Narre o inimigo desviando, bloqueando ou a situação fugindo levemente do controle.
       - Se for FALHA CRÍTICA/DESASTROSA: Narre uma CATASTROFE (arma cai no chão, tropeça, piora tudo).
    
    IMPORTANTE - O GANCHO FINAL: O encerramento da sua narrativa nunca pode ser passivo. Se a ameaça foi neutralizada ou se o teste acabou, não pare por aí! O que acontece em seguida? Um item misterioso cai do inimigo morto? O guarda chega? A pessoa que ele salvou diz algo vital? SEMPRE jogue a história para frente com um evento, fala ou mistério no final que obrigue o jogador a agir.
    
    Mantenha a coerência estrita com a intenção original: "{state.get('current_input')}".
    
    REGRA DE IMERSÃO: NUNCA use a palavra "jogador" ou "usuário" no seu texto. Chame-o sempre de {state.get('char_name')}.
    REGRA DE OURO ABSOLUTA: VOCÊ É O MESTRE, NÃO O JOGADOR. VOCÊ ESTÁ ESTRITAMENTE PROIBIDO de escrever pensamentos, emoções, reações físicas ou falas do personagem {state.get('char_name')} APÓS o resultado da rolagem. Descreva o que aconteceu visualmente e PARE. NUNCA ESCREVA POR ELE!
    """
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": f"🎲 **Resultado:** {roll} vs CD {dc}\n\n{response.content}"})
    state["requires_roll"] = False # Reset
    state["roll_result"] = None
    state["next_node"] = "END"
    return state
