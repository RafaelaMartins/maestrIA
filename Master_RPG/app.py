import streamlit as st
import os
import random
import sys
import re

# FIX: Windows Event Loop Issue
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from orchestrator import build_graph
from utils import CharacterParser, FICHAS_DIR, generate_tts_audio

st.set_page_config(page_title="MestrIA - The Role-Playing Game", layout="wide")

# Setup LangGraph
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "game_started" not in st.session_state:
    st.session_state.game_started = False

if "last_audio_played_idx" not in st.session_state:
    st.session_state.last_audio_played_idx = -1

if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "session_id": "session_1",
        "messages": [],
        "current_input": "",
        "turn_count": 0,
        "char_name": "Aventureiro",
        "char_data": None,
        "current_location": "A Taverna do Sábio Sonolento",
        "active_npcs": {},
        "interacting_npc": None,
        "world_lore": "",
        "main_goal": "",
        "current_plot_stage": 0,
        "next_node": "",
        "requires_roll": False,
        "roll_details": {},
        "roll_result": None
    }

# Theming based on MestrIA v3
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #ffc107; }
    /* Forçar a cor das narrações e do chat para branco puro */
    .stChatMessage .stMarkdown p { color: #ffffff; }
    .stChatMessage .stMarkdown li { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🐉 MestrIA - The Role-Playing Game")

# Sidebar - Ficha
with st.sidebar:
    st.header("🔊 Narração")
    voice_toggle = st.toggle("Habilitar Voz do Mestre", value=False)
    st.session_state.voice_enabled = voice_toggle
    
    st.divider()
    
    st.header("Ficha do Personagem")
    uploaded_file = st.file_uploader("Upload PDF da Ficha", type="pdf")
    
    if uploaded_file and not st.session_state.game_state["char_data"]:
        path = os.path.join(FICHAS_DIR, "temp_ficha.pdf")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("Lendo ficha com IA..."):
            data = CharacterParser.parse_pdf(path)
            st.session_state.game_state["char_data"] = data
            st.session_state.game_state["char_name"] = data.get("name", "Aventureiro")
            st.success("Ficha processada!")
            st.rerun()
            
    if st.session_state.game_state["char_data"]:
        c = st.session_state.game_state["char_data"]
        
        if st.session_state.game_started:
            new_name = st.text_input("Nome do Personagem", value=c.get("name", "Aventureiro"))
            if new_name != c.get("name"):
                c["name"] = new_name
                st.session_state.game_state["char_name"] = new_name
                st.rerun()
            
        st.subheader(c.get("name", "Sem Nome"))
        st.write(f"**Classe:** {c.get('class', 'N/A')}")
        st.json(c.get("attributes", {}))

# CONFIRMATION SCREEN
if st.session_state.game_state["char_data"] and not st.session_state.game_started:
    st.info("Ficha carregada com sucesso! Confirme os dados para iniciar.")
    
    c = st.session_state.game_state["char_data"]
    col1, col2 = st.columns([3, 1])
    
    with col1:
        confirm_name = st.text_input("Nome do Personagem (Edite se necessário):", value=c.get("name", "Aventureiro"))
    
    with col2:
        st.write("") # Spacer
        st.write("") 
        if st.button("Confirmar e Iniciar 🚀", type="primary"):
            st.session_state.game_state["char_data"]["name"] = confirm_name
            st.session_state.game_state["char_name"] = confirm_name
            st.session_state.game_started = True
            st.rerun()

# GAME AREA
if st.session_state.game_started:
    # Intro Auto-start if no messages
    if not st.session_state.game_state["messages"]:
        with st.spinner("Mestre preparando o início da aventura..."):
            st.session_state.game_state["current_input"] = "INTRO_START"
            result = st.session_state.graph.invoke(st.session_state.game_state)
            st.session_state.game_state = result
            st.rerun()

    # Display Chat
    for idx, msg in enumerate(st.session_state.game_state.get("messages", [])):
        if msg.get("content"):
            with st.chat_message(msg["role"]):
                # Limpa tags de voz do texto visível
                display_text = re.sub(r'\[VOICE:.*?\]', '', msg["content"])
                st.markdown(display_text)
                
                # Se for a última mensagem do assistente, e a voz estiver ligada, e ainda não tocou:
                if msg["role"] == "assistant" and st.session_state.get("voice_enabled", False):
                    if idx == len(st.session_state.game_state["messages"]) - 1:
                        if st.session_state.last_audio_played_idx != idx:
                            with st.spinner("Gerando vozes (Atores e Narrador)..."):
                                audio_path = "temp_voice.mp3"
                                
                                npc_voice = None
                                interacting_npc_id = st.session_state.game_state.get("interacting_npc")
                                if interacting_npc_id and "active_npcs" in st.session_state.game_state:
                                    npc_data = st.session_state.game_state["active_npcs"].get(interacting_npc_id, {})
                                    npc_voice = npc_data.get("voz_escolhida")
                                
                                generate_tts_audio(msg["content"], audio_path, npc_voice)
                                st.audio(audio_path, format="audio/mp3", autoplay=True)
                                st.session_state.last_audio_played_idx = idx

    # Action Area
    if st.session_state.game_state.get("requires_roll") and not st.session_state.game_state.get("roll_result"):
        req = st.session_state.game_state["roll_details"]
        with st.container(border=True):
            st.warning(f"🎲 Rolagem necessária: **{req.get('skill')}** (CD {req.get('dc')})")
            if st.button("Rolar D10"):
                roll = random.randint(1, 10)
                st.toast(f"Você rolou {roll}!")
                st.session_state.game_state["roll_result"] = {"total": roll}
                st.session_state.game_state["current_input"] = "Continuar após rolagem"
                
                with st.spinner("Avaliador de Testes calculando..."):
                    st.session_state.game_state = st.session_state.graph.invoke(st.session_state.game_state)
                    st.rerun()

    else:
        st.caption("💡 **Dica:** Use aspas duplas para falas do seu personagem (ex: `\"Quem é você?\"`). Declare intenções ao mestre fora das aspas (ex: `Quero usar persuasão para extrair informações`).")
        prompt = st.chat_input("O que você faz?")
        if prompt:
            new_msg = {"role": "user", "content": prompt}
            if "messages" not in st.session_state.game_state or st.session_state.game_state["messages"] is None:
                st.session_state.game_state["messages"] = [new_msg]
            else:
                st.session_state.game_state["messages"].append(new_msg)
                
            with st.chat_message("user"):
                st.markdown(prompt)
                
            st.session_state.game_state["current_input"] = prompt
            st.session_state.game_state["turn_count"] = st.session_state.game_state.get("turn_count", 0) + 1
            
            with st.spinner("Mestre (e Agentes) pensando..."):
                result = st.session_state.graph.invoke(st.session_state.game_state)
                st.session_state.game_state = result
                st.rerun()
