from state import GameState
from llm import get_llm
import re

def guardrail_autoplay_node(state: GameState) -> GameState:
    """
    Agente de Guardrail: Verifica e corrige qualquer tentativa de autoplay (IA jogando pelo jogador).
    Roda após qualquer agente narrador.
    """
    if not state.get("messages"):
        return state
        
    last_msg = state["messages"][-1]
    if last_msg["role"] != "assistant":
        return state
        
    print("--> [Guardrail] Verificando integridade da narrativa (Autoplay Check)...")
    
    char_name = state.get("char_name", "Aventureiro")
    content = last_msg["content"]
    
    # Lista de gatilhos suspeitos (verbos ou construções que indicam que a IA está agindo pelo jogador)
    triggers = [
        f"{char_name} diz", f"{char_name} fala", f"{char_name} pergunta",
        f"{char_name} responde", f"{char_name} percebe", f"{char_name} sente",
        f"{char_name} entra", f"{char_name} caminha", f"{char_name} decide",
        f"{char_name} nota", f"{char_name} agradece", f"{char_name} sorri",
        f"Você diz", f"Você fala", f"Você pergunta", f"Você percebe"
    ]
    
    # Verifica se algum gatilho está presente
    violation = False
    for trigger in triggers:
        if re.search(rf"\b{re.escape(trigger)}\b", content, re.IGNORECASE):
            violation = True
            break
            
    # Verifica se o Narrador (Antonio) está sendo usado indevidamente para falas entre aspas
    if re.search(r'\[VOICE:Antonio.*?\]\s*"', content, re.IGNORECASE):
        violation = True
        print("    [ALERTA] Antonio usado para diálogo de NPC! Corrigindo...")
            
    if violation:
        print(f"    [ALERTA] Violação detectada no texto! Corrigindo...")
        llm = get_llm()
        
        prompt = f"""
        Você é o Filtro de Autonomia e Imersão do RPG. Sua missão é corrigir o texto da IA.
        
        REGRAS CRÍTICAS:
        1. NUNCA use a voz 'AntonioNeural' para falas entre aspas ("..."). Antônio é APENAS o narrador.
        2. Se houver falas sem tag de voz, adicione [VOICE:Thalita] (Mulher) ou [VOICE:Duarte] (Homem) baseando-se no NPC que fala.
        3. REMOVA qualquer ação, fala ou sentimento do herói {char_name}.
        4. O resultado deve ser: [VOICE:NPC] "Fala" e narração externa.
        
        TEXTO ORIGINAL:
        "{content}"
        
        Responda APENAS com o texto higienizado.
        """
        
        response = llm.invoke(prompt)
        state["messages"][-1]["content"] = response.content.strip()
        print("    [Guardrail] Texto higienizado e vozes corrigidas.")
    else:
        print("    [Guardrail] Narrativa limpa.")
        
    return state
