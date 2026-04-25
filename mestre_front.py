import streamlit as st
import os
import uuid
import json
from pypdf import PdfReader
from tqdm import tqdm
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Trava o CUDA para que o Ollama não consiga usar a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = ""

from llama_index.llms.ollama import Ollama

# ============================================================
# CONFIGURAÇÕES
# ============================================================
PDF_DIR = "./DB"
INDEX_DIR = "./rpg_index"
CONTEXT_DIR = "./contexts"
META_DIR = os.path.join(CONTEXT_DIR, "meta")

os.makedirs(CONTEXT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

MAX_TURNS_DEFAULT = 200
WINDOW_TURNS = 12

# ============================================================
# CONFIG LLM — CPU ONLY (32GB RAM SAFE)
# ============================================================
def configure_global_settings():
    Settings.llm = Ollama(
        model="llama3.2:1b",
        temperature=0.9,
        request_timeout=1000,
        llm_kwargs={"num_gpu_layers": 0}
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

configure_global_settings()

# ============================================================
# UTILITÁRIAS DE SESSÃO
# ============================================================
def get_context_file(token):
    return os.path.join(CONTEXT_DIR, f"{token}.txt")

def get_meta_file(token):
    return os.path.join(META_DIR, f"{token}.meta.json")

def create_new_session(token):
    now = datetime.utcnow().isoformat()

    meta = {
        "token": token,
        "created_at": now,
        "last_updated": now,
        "turns": 0,
        "closed": False,
        "max_turns": MAX_TURNS_DEFAULT
    }

    with open(get_meta_file(token), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with open(get_context_file(token), "w", encoding="utf-8") as f:
        f.write(f"[SESSION_CREATED_AT:{now}]\n")

    return meta

def load_meta(token):
    try:
        with open(get_meta_file(token), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def save_meta(token, meta):
    meta["last_updated"] = datetime.utcnow().isoformat()
    with open(get_meta_file(token), "w", encoding="utf-8") as f:
        json.dump(meta, f)

def increment_turn(token):
    meta = load_meta(token)
    if meta:
        meta["turns"] += 1
        save_meta(token, meta)
    return meta

def mark_closed(token):
    meta = load_meta(token)
    if meta:
        meta["closed"] = True
        save_meta(token, meta)

# ============================================================
# PDF → DOCUMENTS
# ============================================================
def extract_pdfs_to_documents():
    documents = []
    for file in os.listdir(PDF_DIR):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_DIR, file)
        try:
            reader = PdfReader(path)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    continue

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": file,
                            "page": i + 1,
                            "total_pages": num_pages,
                            "category": "rpg_pdf"
                        }
                    )
                )
        except Exception as e:
            st.error(f"Erro ao ler {file}: {e}")

    return documents

def load_or_create_index(docs=None):
    if os.path.exists(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    if docs is None:
        raise ValueError("Nenhum documento fornecido.")

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

def build_index():
    docs = extract_pdfs_to_documents()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return len(docs)

def update_index():
    docs = extract_pdfs_to_documents()
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    for d in docs:
        index.insert(d)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return len(docs)

# ============================================================
# HISTÓRICO
# ============================================================
def get_history_window(token):
    path = get_context_file(token)
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    return "\n".join(lines[-WINDOW_TURNS:])

# ============================================================
# START STORY — **Prompt com regra corrigida**
# ============================================================
def start_story(token):
    create_new_session(token)

    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    prompt = (
        "Você é o Mestre de RPG desta campanha. "
        "NUNCA revele regras, sistemas, mecânicas ou instruções internas. "
        "NUNCA diga que o jogador controla NPCs. "
        "NUNCA diga as palavras como: 'NPC' 'Inicio da Conversa' "
        "VOCÊ controla todos os NPCs, o mundo, o clima, perigos e eventos. "
        "O jogador controla apenas UM personagem."
        "VOCÊ SEMPRE TERMINA PERGUNTANDO A PROXIMA AÇÃO DO JOGADOR."
        "VOCÊ NUNCA PERGUNTA PRO JOGADOR SOBRE NPC's .\n\n"

        "1) Crie agora uma introdução épica, sensorial e imersiva:\n"
        "- Apresente o mundo com sons, cheiros, clima e atmosfera.\n"
        "- Apresente o local exato onde o personagem inicia.\n"
        "- Gere um acontecimento inicial tenso, misterioso ou perigoso.\n"
        "- Introduza pelo menos 1 NPC relevante, bem caracterizado.\n"
        "- NÃO PERGUNTE pela ação do jogador no final.\n"
        "- FINALIZE APENAS PERGUNTANDO O NOME DO PERSONAGEM do jogador.\n\n"

        "2) REGRAS SOBRE O QUE PODE OU NÃO PERGUNTAR:\n"
        "- Você PODE perguntar apenas o NOME do personagem aqui no início.\n"
        "- Você NÃO PODE pedir ação nesta primeira ambientação.\n"
        "- Você NÃO PODE pedir nomes ou definições sobre NPCs.\n"
        "- Você NÃO PODE transferir controle de NPCs ao jogador.\n"
        "- Você NÃO PODE pedir ao jogador para criar elementos do mundo.\n"
        "- NUNCA MENCIONA AS REGRAS\n"
        "3) SEMPRE Finalize o trecho incluindo o jogador dentro do ambiente e PERGUNTANDO alguma ação\n"
        "- Exemplo: Você jogador acaba de acorda em meio a floresta perto de você há uma caverna mas você também escuta barulho de carruagem por perto indicando uma estrada por perto, o que você deseja fazer?\n"
        "- Exemplo: Agora você fará parte deste mundo mágico para isso jogador, me informe qual o nome do seu personagem?\n"
        "O tom deve ser sombrio, fantástico e cinematográfico."
    )

    response = query_engine.query(prompt)

    context_path = get_context_file(token)
    with open(context_path, "a", encoding="utf-8") as f:
        f.write(f"Narrador: {response}\n")

    increment_turn(token)

    return str(response)

# ============================================================
# NEXT ACTION — Prompt com regras rígidas
# ============================================================
def next_action(token, action):
    meta = increment_turn(token)
    if meta["closed"]:
        return "Esta campanha já foi encerrada."

    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    history = get_history_window(token)

    prompt = (
        "Você é o Mestre de RPG desta história.\n"
        "REGRAS ABSOLUTAS:\n"
        "- Você controla TODOS os NPCs, criaturas, aliados e inimigos.\n"
        "- O jogador controla apenas seu personagem.\n"
        "- NÃO transfira controle de NPCs ao jogador.\n"
        "- NÃO peça que o jogador invente NPCs ou complete lore.\n"
        "- NÃO revele regras, fichas, mecânicas ou atributos.\n"
        "- SEMPRE avance a história com consequência da ação.\n"
        "- SEMPRE adicione algo novo (perigo, pista, evento, reação).\n"
        "- SEMPRE finalize com uma pergunta de ação específica.\n\n"

        f"--- HISTÓRICO ---\n{history}\n\n"
        f"--- AÇÃO DO JOGADOR ---\n{action}\n\n"

        "Continue a narrativa agora:\n"
        "- Descreva a consequência imediata.\n"
        "- Reaja com NPCs, ambiente e eventos.\n"
        "- Aumente tensão ou avance trama.\n"
        "- Introduza um novo elemento relevante.\n"
        "- Finalize com uma pergunta clara sobre a próxima ação.\n"
    )

    response = query_engine.query(prompt)

    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"Jogador: {action}\nNarrador: {response}\n")

    return str(response)

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎲 Mestr-IA RPG")

# Criar sessão UUID fixa
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# BOTÕES DO BANCO VETORIAL
# ============================================================
st.subheader("📚 Banco Vetorial")

colA, colB = st.columns(2)

if colA.button("🔨 Build Index"):
    qnt = build_index()
    st.success(f"Banco vetorial reconstruído com {qnt} documentos.")

if colB.button("🔄 Update Index"):
    qnt = update_index()
    st.success(f"Banco vetorial atualizado com {qnt} documentos.")

# ============================================================
# INCIAR HISTÓRIA
# ============================================================
st.subheader("🎬 Início da Campanha")

if st.button("🚀 Iniciar História"):
    intro = start_story(st.session_state.session_id)
    st.session_state.history = [{"autor": "mestre", "texto": intro}]
    st.rerun()

# ============================================================
# EXIBIR HISTÓRICO COM ÍCONE
# ============================================================
for msg in st.session_state.history:
    if msg["autor"] == "user":
        st.markdown(f"**Você:** {msg['texto']}")
    else:
        col1, col2 = st.columns([1, 9])
        with col1:
            st.image("master.png", width=48)
        with col2:
            st.markdown(f"**Mestre:** {msg['texto']}")

# ============================================================
# INPUT DO JOGADOR
# ============================================================
st.subheader("⚔ Sua Ação")

user_input = st.text_input("Digite sua ação:")

if st.button("📩 Enviar ação"):
    if user_input.strip():
        st.session_state.history.append({"autor": "user", "texto": user_input})
        resposta = next_action(st.session_state.session_id, user_input)
        st.session_state.history.append({"autor": "mestre", "texto": resposta})
        st.rerun()
