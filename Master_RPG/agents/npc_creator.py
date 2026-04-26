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
            # Validação de Voz vs Gênero (Garantia de imersão)
            genero = npc_data.get("genero", "Masculino")
            voz = npc_data.get("voz_escolhida", "pt-PT-DuarteNeural")
            
            if genero == "Feminino" and "Duarte" in voz:
                npc_data["voz_escolhida"] = "pt-BR-ThalitaMultilingualNeural"
            elif genero == "Masculino" and ("Thalita" in voz or "Francisca" in voz):
                npc_data["voz_escolhida"] = "pt-PT-DuarteNeural"
                
            if "active_npcs" not in state:
                state["active_npcs"] = {}
            state["active_npcs"][npc_data["id"]] = npc_data
            print(f"    NPC Criado: {npc_data['nome']} ({npc_data['tipo']}) - Voz: {npc_data['voz_escolhida']}")
    except Exception as e:
        print(f"    Erro ao parsear NPC: {e}")
        
    return state

def populador_cena_node(state: GameState) -> GameState:
    """
    Agente que seleciona quem está na cena atual a partir do Pool Global.
    """
    if not state.get("global_npc_pool"):
        return state
        
    print(f"--> [Populador] Selecionando elenco para {state.get('current_location')}...")
    
    # Se já tem NPCs na cena e não mudou de lugar, mantém
    if state.get("current_scene_npcs") and len(state.get("current_scene_npcs")) >= 2:
        return state
        
    llm = get_llm()
    pool_str = json.dumps(state["global_npc_pool"], indent=2)
    
    prompt = f"""
    Você é o Diretor de Elenco.
    Local Atual: {state.get('current_location')}
    Lore: {state.get('world_lore')}
    
    Escolha EXATAMENTE 2 NPCs do Pool abaixo que façam mais sentido estar neste local AGORA.
    POOL:
    {pool_str}
    
    Retorne APENAS um JSON com os IDs:
    {{
        "selecionados": ["npc_id1", "npc_id2"]
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            ids = data.get("selecionados", [])
            state["current_scene_npcs"] = ids
            
            # Adiciona os detalhes dos selecionados ao active_npcs com correção de voz
            if "active_npcs" not in state: state["active_npcs"] = {}
            for npc in state["global_npc_pool"]:
                if npc["id"] in ids:
                    # Correção de Voz vs Gênero no pool também
                    gen = npc.get("genero", "Masculino")
                    v = npc.get("voz_escolhida", "pt-PT-DuarteNeural")
                    if gen == "Feminino" and "Duarte" in v:
                        npc["voz_escolhida"] = "pt-BR-ThalitaMultilingualNeural"
                    elif gen == "Masculino" and ("Thalita" in v or "Francisca" in v):
                        npc["voz_escolhida"] = "pt-PT-DuarteNeural"
                        
                    state["active_npcs"][npc["id"]] = npc
            
            print(f"    Elenco da cena: {', '.join(ids)}")
    except Exception as e:
        print(f"    Erro ao popular cena: {e}")
        
    return state
