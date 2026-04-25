import streamlit as st
import os
import random
import sys

# FIX: Windows Event Loop Issue
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from orchestrator import build_graph

# Setup LangGraph
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "session_id": "session_1",
        "messages": [],
        "current_input": "",
        "char_name": "Herói",
        "char_data": {},
        "current_location": "A Taverna Inicial",
        "active_npcs": {},
        "world_lore": "",
        "main_goal": "",
        "current_plot_stage": 0,
        "next_node": "",
        "requires_roll": False,
        "roll_details": {},
        "roll_result": None
    }

st.set_page_config(page_title="Master_RPG (LangGraph)", layout="wide")

# Theming based on MestrIA v3
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

st.title("🐉 Master_RPG (Multi-Agente)")

# Intro Auto-start if no messages
if not st.session_state.game_state["messages"] and not st.session_state.game_state.get("world_lore"):
    with st.spinner("Roteirista criando o mundo..."):
        st.session_state.game_state["current_input"] = "Começar aventura"
        result = st.session_state.graph.invoke(st.session_state.game_state)
        st.session_state.game_state = result
        st.rerun()

# Display Chat
for msg in st.session_state.game_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Action Area
if st.session_state.game_state.get("requires_roll") and not st.session_state.game_state.get("roll_result"):
    req = st.session_state.game_state["roll_details"]
    with st.container(border=True):
        st.warning(f"🎲 Rolagem necessária: **{req.get('skill')}** (CD {req.get('dc')})")
        if st.button("Rolar D20"):
            roll = random.randint(1, 20)
            st.toast(f"Você rolou {roll}!")
            st.session_state.game_state["roll_result"] = {"total": roll}
            # Continue the graph execution (it will resume at Orquestrador or directly to Evaluator)
            # Para o LangGraph, a gente invoca novamente com o input vazio e o state atualizado
            st.session_state.game_state["current_input"] = "Continuar após rolagem"
            
            with st.spinner("Avaliador de Testes calculando..."):
                st.session_state.game_state = st.session_state.graph.invoke(st.session_state.game_state)
                st.rerun()

else:
    prompt = st.chat_input("O que você faz?")
    if prompt:
        # User message
        new_msg = {"role": "user", "content": prompt}
        # Safely append to avoid overwriting issues
        if "messages" not in st.session_state.game_state or st.session_state.game_state["messages"] is None:
            st.session_state.game_state["messages"] = [new_msg]
        else:
            # LangGraph lists with reducers sometimes expect just the new elements when invoking, 
            # but since we modify state manually here, we just append to the state.
            st.session_state.game_state["messages"].append(new_msg)
            
        with st.chat_message("user"):
            st.markdown(prompt)
            
        st.session_state.game_state["current_input"] = prompt
        
        with st.spinner("Mestre (e Agentes) pensando..."):
            # Invoke LangGraph
            result = st.session_state.graph.invoke(st.session_state.game_state)
            st.session_state.game_state = result
            st.rerun()
