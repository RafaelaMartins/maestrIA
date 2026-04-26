from state import GameState
from llm import get_llm
import json
import re

def npc_creator_node(state: GameState) -> GameState:
    """
    Agente 3: Cria NPCs com história, dificuldade, objetivos, pontos fortes e fracos.
    """
    print("--> [Agente Criador de NPCs] Gerando NPC...")
    llm = get_llm()
    
    prompt = f"""
    Você é o Criador de NPCs do RPG.
    O contexto atual exige a criação de um novo NPC (Inimigo, Amigo ou Informante).
    Crie um NPC que se encaixe no mundo atual: {state.get("world_lore")}
    
    Retorne APENAS um JSON válido com a estrutura:
    {{
        "id": "npc_1",
        "nome": "Nome",
        "genero": "Masculino|Feminino",
        "voz_escolhida": "pt-BR-ThalitaMultilingualNeural|pt-PT-DuarteNeural|pt-BR-FranciscaNeural",
        "tipo": "Inimigo|Amigo|Informante",
        "dificuldade": "Fácil|Médio|Difícil",
        "objetivo": "O que ele quer",
        "pontos_fortes": "Força, magia, etc",
        "pontos_fracos": "Fraquezas",
        "historia": "Background curto"
    }}
    
    DICA DE VOZ:
    - Para Mulheres: pt-BR-ThalitaMultilingualNeural ou pt-BR-FranciscaNeural.
    - Para Homens: pt-PT-DuarteNeural (Sotaque lusitano para diferenciar do narrador).
    """
    
    response = llm.invoke(prompt)
    
    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            npc_data = json.loads(json_match.group(0))
            if "active_npcs" not in state:
                state["active_npcs"] = {}
            state["active_npcs"][npc_data["id"]] = npc_data
            print(f"    NPC Criado: {npc_data['nome']} ({npc_data['tipo']})")
    except Exception as e:
        print(f"    Erro ao parsear NPC: {e}")
        
    return state
