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
    Você é o Roteirista Chefe de um RPG de mesa. Sua tarefa é criar uma experiência ÚNICA, um enredo original e um mundo do zero para um herói chamado {state.get('char_name', 'Aventureiro')}.
    
    Seja extremamente criativo. Crie um mistério, uma ameaça ou uma missão que fuja de clichês básicos.
    
    Formato desejado (Gere textos dinâmicos e originais):
    MUNDO: [Nome do mundo e uma descrição vívida da atmosfera atual]
    TRAMA PRINCIPAL: [Qual é o grande mistério, vilão ou evento catastrófico que move a história?]
    NPCs CHAVE: [Crie 2 ou 3 NPCs originais que guardam segredos sobre a Trama Principal. Descreva o nome deles e como eles se ligam à trama]
    GANCHOS INICIAIS (PISTAS): [Crie 2 ou 3 rumores, itens estranhos ou eventos que devem acontecer na Taverna Inicial para "puxar" o jogador para a Trama Principal]
    """
    
    response = llm.invoke(prompt)
    
    # Store in hidden state
    state["world_lore"] = response.content
    state["main_goal"] = "Completar a jornada estabelecida pelo roteirista."
    
    return state

def world_updater_node(state: GameState) -> GameState:
    """
    Agente do Mundo: Atualiza o estado global com base nas ações recentes.
    Roda a cada N turnos.
    """
    print("--> [Agente do Mundo] Atualizando consequências globais...")
    llm = get_llm()
    
    # Pega os últimos 10 turnos (aproximadamente as últimas 20 mensagens)
    recent_msgs = state.get("messages", [])[-20:]
    history = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
    
    prompt = f"""
    Você é o Agente do Mundo de um RPG.
    Sua tarefa é observar os eventos recentes e reescrever a Lore/Estado Atual do Mundo para refletir as ações do jogador.
    
    Lore Atual:
    {state.get('world_lore')}
    
    Histórico Recente de Ações:
    {history}
    
    Baseado no histórico, a lore precisa mudar? Alguma facção foi destruída? Uma cidade está em alerta? Um novo boato surgiu?
    Reescreva a Lore Atual de forma sucinta (máximo 3 parágrafos) incorporando essas consequências, mas mantendo a base original intacta.
    """
    
    response = llm.invoke(prompt)
    state["world_lore"] = response.content
    
    # Note: It does not add a message to the chat. It updates the background state silently.
    return state
