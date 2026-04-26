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
    
    # Contexto de NPCs na cena
    scene_npcs_data = []
    if state.get("current_scene_npcs"):
        for npc_id in state["current_scene_npcs"]:
            if npc_id in state.get("active_npcs", {}):
                scene_npcs_data.append(state["active_npcs"][npc_id])
    npcs_context = "\n".join([f"- {n['nome']} ({n['tipo']}) [Voz: {n.get('voz_escolhida')}]" for n in scene_npcs_data])
    
    prompt = f"""
    Contexto da cena: 
    {context}
    
    Você é o Mestre Geral do RPG.
    Local Atual: {state.get('current_location')}
    DIRETRIZ DE NARRATIVA: Conduza {state.get('char_name')} para a QUEST PRINCIPAL.
    
    REGRA MÁXIMA ABSOLUTA (CONTROLE DO JOGADOR):
    1. VOCÊ SÓ NARRA A REAÇÃO DO MUNDO E DOS NPCs. 
    2. É EXPRESSAMENTE PROIBIDO narrar {state.get('char_name')} fazendo, dizendo, sentindo ou percebendo algo.
    3. NUNCA use frases como: "{state.get('char_name')} caminha", "{state.get('char_name')} responde", "{state.get('char_name')} entra".
    4. EXEMPLO RUIM: "Findariel agradece e sai da taverna." (PROIBIDO)
    5. EXEMPLO BOM: "O barman acena com a cabeça. As portas da taverna rangem conforme {state.get('char_name')} se retira para a rua noturna." (CORRETO)
    
    REGRA DE VOZ DOS NPCs (DETERMINÍSTICA):
    Sempre que um NPC falar (texto entre aspas), você DEVE colocar a tag da voz dele imediatamente ANTES da abertura das aspas. Use as vozes fornecidas no contexto. Se for um NPC sem voz listada, escolha uma das seguintes:
    - Aliados/Neutros: pt-PT-DuarteNeural (Masc) ou pt-BR-ThalitaNeural (Fem)
    - Inimigos/Monstros: pt-PT-DuarteNeural (Masc) ou pt-PT-RaquelNeural (Fem)
    - NUNCA use pt-BR-AntonioNeural para NPCs (voz exclusiva do narrador).
    EXEMPLO OBRIGATÓRIO: [VOICE:pt-BR-ThalitaNeural] "Cuidado com isso!" avisa a maga.
    
    O herói {state.get('char_name')} acabou de tentar: {state.get('current_input')}.
    Narre o resultado físico e a reação dos NPCs. 
    
    REGRA DE MISTÉRIO DE NPCs:
    Você PODE revelar o nome de NPCs novos se fizer sentido o herói já conhecê-los de vista. CONTUDO, mantenha uma proporção: para cada 2 NPCs conhecidos, faça 1 ser totalmente ENIGMÁTICO. Para NPCs enigmáticos, oculte o nome na narração e refira-se a eles apenas por características visuais (ex: "O homem encapuzado", "A elfa com a cicatriz") até que eles decidam se apresentar.
    
    IMPORTANTE - O GANCHO: O encerramento DEVE envolver o jogador DIRETAMENTE. Faça com que os acontecimentos afetem o herói: um NPC olha diretamente para ele e faz uma pergunta, algo cai em sua direção, ou um perigo surge subitamente. Termine SEMPRE com "O que você faz?" ou "Como você reage?"
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
    
    Você é o Mestre Narrador.
    DIRETRIZ: Conduza {state.get('char_name')} para a QUEST através de reações do mundo.
    
    REGRA MÁXIMA: NÃO NARRAR AÇÕES DO JOGADOR. 
    O jogador tentou: {state.get('current_input')}.
    Narre APENAS como o cenário e os NPCs reagem a isso. 
    EXEMPLO: Se o jogador diz "Vou para a loja", você narra: "As ruas de Novaria estão úmidas. A placa da loja de facas balança ao vento logo adiante." (E NÃO: "Você caminha até a loja").
    
    REGRA DE VOZ DOS NPCs (DETERMINÍSTICA):
    Sempre que um NPC falar (texto entre aspas), você DEVE colocar a tag da voz dele imediatamente ANTES da abertura das aspas. Use as vozes fornecidas no contexto. Se for um NPC sem voz listada, escolha uma das seguintes:
    - Aliados/Neutros: pt-PT-DuarteNeural (Masc) ou pt-BR-ThalitaNeural (Fem)
    - Inimigos/Monstros: pt-PT-DuarteNeural (Masc) ou pt-PT-RaquelNeural (Fem)
    - NUNCA use pt-BR-AntonioNeural para NPCs (voz exclusiva do narrador).
    EXEMPLO OBRIGATÓRIO: [VOICE:pt-BR-ThalitaNeural] "Não vá por ali!" grita Elyria.
    
    REGRA DE MISTÉRIO DE NPCs:
    Você PODE revelar o nome de NPCs novos se fizer sentido o herói já conhecê-los. CONTUDO, em média, faça 1 a cada 3 NPCs ser ENIGMÁTICO. Para os enigmáticos, oculte o nome e use apenas características visuais (ex: "A figura sombria") até que decidam se apresentar.
    
    IMPORTANTE - O GANCHO: O encerramento DEVE envolver o jogador DIRETAMENTE. Faça com que os acontecimentos afetem o herói de forma ativa ou crie um perigo imediato. Termine SEMPRE com "O que você faz?" ou "Como você reage?"
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
    roll_d20 = state["roll_result"]["d20"]
    dc = state["roll_details"]["dc"]
    attr_name = state["roll_details"].get("attr", "FOR")
    
    # Busca o valor do atributo, se não existir, é 0
    attr_val = 0
    if state.get("char_data") and state["char_data"].get("attributes"):
        attr_val = state["char_data"]["attributes"].get(attr_name, 0)
        
    total_roll = roll_d20 + attr_val
    diff = total_roll - dc
    
    if roll_d20 == 20:
        resultado_str = "SUCESSO CRÍTICO/ESPETACULAR (20 Natural!)"
    elif roll_d20 == 1:
        resultado_str = "FALHA CRÍTICA/DESASTROSA (1 Natural!)"
    elif diff >= 0:
        resultado_str = "SUCESSO"
    else:
        resultado_str = "FALHA SIMPLES"
        
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
       
    REGRA DE VOZ DOS NPCs (DETERMINÍSTICA):
    Sempre que um NPC falar (texto entre aspas), você DEVE colocar a tag da voz dele imediatamente ANTES da abertura das aspas. Use as vozes fornecidas no contexto. Se for um NPC sem voz listada, escolha uma das seguintes:
    - Aliados/Neutros: pt-PT-DuarteNeural (Masc) ou pt-BR-ThalitaNeural (Fem)
    - Inimigos/Monstros: pt-PT-DuarteNeural (Masc) ou pt-PT-RaquelNeural (Fem)
    - NUNCA use pt-BR-AntonioNeural para NPCs (voz exclusiva do narrador).
    EXEMPLO OBRIGATÓRIO: [VOICE:pt-PT-DuarteNeural] "Argh! Meu braço!" o bandido berra.
    
    IMPORTANTE - O GANCHO FINAL: O encerramento da sua narrativa nunca pode ser passivo ou distante. Crie uma consequência direta para o herói: o monstro o encara com ódio, um teto ameaça desabar sobre sua cabeça, ou uma revelação chocante é dita olhando em seus olhos. SEMPRE jogue a história para frente exigindo reação. Termine SEMPRE o texto com: "O que você faz?" ou "Como você reage?"
    
    REGRA MÁXIMA ABSOLUTA: VOCÊ NUNCA ASSUME O QUE O PERSONAGEM {state.get('char_name')} DIZ OU FAZ APÓS O RESULTADO. É EXPRESSAMENTE PROIBIDO inventar falas, pensamentos ou ações para o jogador. O jogador tem 100% de autonomia. NÃO CONFUNDA o herói com os NPCs.
    
    REGRA DE IMERSÃO: NUNCA use a palavra "jogador" ou "usuário" no seu texto. Chame-o sempre de {state.get('char_name')}.
    REGRA DE OURO ABSOLUTA: VOCÊ É O MESTRE, NÃO O JOGADOR. VOCÊ ESTÁ ESTRITAMENTE PROIBIDO de escrever pensamentos, emoções, reações físicas ou falas do personagem {state.get('char_name')} APÓS o resultado da rolagem. Descreva o que aconteceu visualmente e PARE. NUNCA ESCREVA POR ELE!
    """
    response = llm.invoke(prompt)
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    
    msg_resultado = f"🎲 **Resultado:** {roll_d20} (d20) + {attr_val} ({attr_name}) = {total_roll} vs CD {dc}"
    state["messages"].append({"role": "assistant", "content": f"{msg_resultado}\n\n{response.content}"})
    state["requires_roll"] = False # Reset
    state["roll_result"] = None
    state["next_node"] = "END"
    return state
