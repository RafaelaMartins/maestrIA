from state import GameState
from llm import get_llm
import json
import re

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
    
    MUNDO: [Nome e atmosfera]
    TRAMA PRINCIPAL: [O grande mistério ou objetivo final]
    
    ESTRUTURA DE QUESTS (O CICLO DO HERÓI):
    1. QUEST INICIAL: [Objetivo X, Y e Z que o jogador deve cumprir para o NPC inicial]
    2. RECOMPENSA: [O que ele ganha ao voltar para o NPC após os desafios]
    3. GANCHO PARA PRÓXIMA QUEST: [Como a vitória na Quest 1 leva à Quest 2]
    
    POOL_NPC_JSON: [Gere 10 NPCs, garantindo que pelo menos 2 sejam 'Quest Givers' com objetivos ligados à estrutura acima]
    ... (formato JSON já estabelecido)
    {{
        "npcs": [
            {{
                "id": "npc_X",
                "nome": "Nome",
                "genero": "Masculino ou Feminino (NUNCA os dois)",
                "voz_escolhida": "pt-BR-ThalitaNeural (Fem Aliada) | pt-PT-DuarteNeural (Masc) | pt-PT-RaquelNeural (Fem Inimiga) | pt-BR-FranciscaNeural (Fem Neutra)",
                "tipo": "Amigo|Neutro|Inimigo|Indiferente",
                "dificuldade": "Alta (para aliados imortais e chefes) | Média | Baixa (capangas/comuns)",
                "objetivo": "O que ele quer",
                "historia": "Background curto"
            }},
            ...
        ]
    }}
    """
    
    response = llm.invoke(prompt)
    
    # Extrair Lore e Pool
    content = response.content
    lore_parts = content.split("POOL_NPC_JSON:")
    
    state["world_lore"] = lore_parts[0].strip()
    state["main_goal"] = "Completar a jornada estabelecida pelo roteirista."
    
    if len(lore_parts) > 1:
        try:
            json_match = re.search(r"\{.*\}", lore_parts[1], re.DOTALL)
            if json_match:
                pool_data = json.loads(json_match.group(0))
                state["global_npc_pool"] = pool_data.get("npcs", [])
                print(f"    Casting completo: {len(state['global_npc_pool'])} NPCs prontos.")
        except Exception as e:
            print(f"    Erro ao gerar Pool de NPCs: {e}")
            state["global_npc_pool"] = []
    
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
