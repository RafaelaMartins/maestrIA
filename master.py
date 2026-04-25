# mestre_front.py
import streamlit as st
import os
import uuid
import json
import re
import random
import time
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
from llama_index.llms.ollama import Ollama

# ---------------------------
# FORÇAR USO DE RAM (DESATIVA GPU)
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = ""

# ---------------------------
# CONFIGURAÇÕES PATHS
# ---------------------------
PDF_DIR = "./DB"
INDEX_DIR = "./rpg_index"
CONTEXT_DIR = "./contexts"
META_DIR = os.path.join(CONTEXT_DIR, "meta")
FICHAS_DIR = os.path.join(CONTEXT_DIR, "fichas")  # guarda pdfs e jsons de ficha

os.makedirs(CONTEXT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(FICHAS_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

MAX_TURNS_DEFAULT = 200
WINDOW_TURNS = 12

# ---------------------------
# CONFIG LLM (CPU ONLY)
# ---------------------------
def configure_global_settings():
    # garante que o LLM não tente GPU (num_gpu_layers=0)
    Settings.llm = Ollama(
        model="llama3.2:1b",           # mantenha seu modelo escolhido; substitua por mistral se preferir
        temperature=0.9,
        request_timeout=1000,
        llm_kwargs={"num_gpu_layers": 0}
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

configure_global_settings()

# ---------------------------
# UTILITÁRIAS SESSÃO / ARQUIVOS
# ---------------------------
def get_context_file(token):
    return os.path.join(CONTEXT_DIR, f"{token}.txt")

def get_meta_file(token):
    return os.path.join(META_DIR, f"{token}.meta.json")

def ficha_json_path(token):
    return os.path.join(FICHAS_DIR, f"{token}.character.json")

def ficha_pdf_path(token):
    return os.path.join(FICHAS_DIR, f"{token}.pdf")

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

# ---------------------------
# PDF -> DOCUMENTS (banco vetorial)
# ---------------------------
def extract_pdfs_to_documents(pdf_dir=PDF_DIR):
    documents = []
    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, file)
        try:
            reader = PdfReader(path)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    continue
                metadata = {"source": file, "page": i+1, "total_pages": num_pages, "category": "rpg_pdf"}
                documents.append(Document(text=text, metadata=metadata))
        except Exception as e:
            st.error(f"Erro ao ler {file}: {e}")
    return documents

def load_or_create_index(docs=None):
    if os.path.exists(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)
    if docs is None:
        raise ValueError("Nenhum documento fornecido para criar índice.")
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

# ---------------------------
# FICHA (PDF) PARSING — heurística simples
# ---------------------------
ATTRIBUTES = ["FOR", "STR", "FORÇA", "DES", "DEX", "DESTREZA", "CON", "CONSTITUIÇÃO", "INT", "INTELIGÊNCIA", "SAB", "WIS", "SABEDORIA", "CAR", "CHA", "CARISMA"]

def parse_ficha_pdf_to_json(pdf_path):
    """
    Lê PDF e tenta extrair nome do personagem e atributos (valores numéricos).
    Retorna dict com 'name' e 'attributes' (mapa atributo->int).
    Heurística simples: procura palavras-chaves próximas de números.
    """
    result = {"name": None, "attributes": {}}
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += "\n" + t
        # Normaliza
        txt = text.upper()
        # tenta achar nome: procura "NOME" ou "PERSONAGEM"
        m = re.search(r"NOME[:\s]+([A-Z0-9 \-']{2,60})", txt)
        if not m:
            m = re.search(r"PERSONAGEM[:\s]+([A-Z0-9 \-']{2,60})", txt)
        if m:
            name = m.group(1).strip().title()
            result["name"] = name
        else:
            # fallback: pega primeira linha com duas palavras
            first_lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if first_lines:
                candidate = first_lines[0]
                # pega até 5 primeiras palavras
                result["name"] = " ".join(candidate.split()[:5]).title()

        # procura atributos com regex: palavra + número (ex: FOR 16 ou FORÇA 12)
        for token in ATTRIBUTES:
            pattern = rf"{token}\s*[:\-]?\s*(\d{{1,2}})"
            mm = re.search(pattern, txt)
            if mm:
                try:
                    val = int(mm.group(1))
                    # normaliza token para forma curta
                    key = token[:3].upper()
                    if key in ["FOR","DES","CON","INT","SAB","CAR"]:
                        result["attributes"][key] = val
                    else:
                        # map full names
                        if "FOR" in token:
                            result["attributes"]["FOR"] = val
                        elif token in ["DES","DEX","DESTREZA"]:
                            result["attributes"]["DES"] = val
                        elif "CON" in token:
                            result["attributes"]["CON"] = val
                        elif "INT" in token:
                            result["attributes"]["INT"] = val
                        elif "SAB" in token:
                            result["attributes"]["SAB"] = val
                        elif token in ["CAR","CHA","CARISMA"]:
                            result["attributes"]["CAR"] = val
                except:
                    pass
        # se falta algum atributo, deixa vazio (pode ser completado depois manualmente)
        return result
    except Exception as e:
        st.error(f"Erro parseando ficha: {e}")
        return result

# ---------------------------
# REFEREE / AGENTE PASSIVO
# ---------------------------
# Mapeamentos simples keyword->atributo/perícia
SKILL_MAP = {
    # palavras chaves : (atributo, exemplo de perícia)
    "ATIRAR": ("DES", "Tiro com Arma"),
    "ARCO": ("DES", "Tiro com Arco"),
    "ESCALAR": ("FOR", "Escalada"),
    "PULAR": ("FOR", "Acrobacia"),
    "CROPAR": ("INT", "Conhecimento"),
    "DISCUSS": ("CAR", "Persuasão"),
    "CONVENCER": ("CAR", "Persuasão"),
    "FEITIÇO": ("INT", "Magia"),
    "LANÇAR": ("INT", "Magia"),
    "FURTAR": ("DES", "Furtividade"),
    "ESCONDER": ("DES", "Furtividade"),
    "PERCEP": ("SAB", "Percepção"),
    "OBSER": ("SAB", "Percepção"),
    "ATACAR": ("FOR", "Ataque"),
    "BLOQUEAR": ("FOR", "Defesa"),
    "CURAR": ("SAB", "Medicina"),
}

def detect_check_from_action(action_text):
    """
    Verifica se a ação do jogador provavelmente precisa de teste.
    Retorna (needs_check:bool, attribute:str, skill:str, suggested_dc:int)
    """
    a = action_text.upper()
    for kw, (attr, skill) in SKILL_MAP.items():
        if kw in a:
            # heurística de DC: se verbo simples -> DC médio 12, se verbo tenso -> DC 15, se palavra 'difícil' -> 18
            dc = 12
            if "DIF" in a or "DIFÍCIL" in a or "DIFICIL" in a:
                dc = 18
            elif "MUITO" in a or "PERIGOS" in a or "PERIGO" in a:
                dc = 15
            return True, attr, skill, dc
    # fallback: se contém 'TEST' ou 'ROLAR'...
    if "ROL" in a or "TESTE" in a or "ROLAR" in a:
        return True, "DES", "Teste Genérico", 12
    return False, None, None, None

def attribute_mod(score):
    """Calcula modificador D&D-like: floor((score-10)/2)"""
    try:
        return (int(score) - 10) // 2
    except:
        return 0

def resolve_check(ficha, attr, dc):
    """
    Resolve o roll: rola d20, soma mod do atributo, retorna result dict:
    { roll, mod, total, dc, degree, message }
    degree: 'crit_fail', 'fail', 'success', 'crit_success'
    """
    roll = random.randint(1, 20)
    base = ficha.get("attributes", {}).get(attr, None)
    mod = attribute_mod(base) if base is not None else 0
    total = roll + mod
    degree = "fail"
    if roll == 1:
        degree = "crit_fail"
    elif roll == 20:
        degree = "crit_success"
    else:
        if total >= dc + 5:
            degree = "crit_success"
        elif total >= dc:
            degree = "success"
        else:
            degree = "fail"
    msg = f"Rolou {roll} + mod {mod} = {total} contra DC {dc} → {degree.replace('_',' ')}"
    return {"roll": roll, "mod": mod, "total": total, "dc": dc, "degree": degree, "message": msg}

# ---------------------------
# HISTÓRICO
# ---------------------------
def get_history_window(token):
    path = get_context_file(token)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return "\n".join(lines[-WINDOW_TURNS:])

# ---------------------------
# START STORY / NEXT ACTION (integra referee)
# ---------------------------
def start_story(token):
    create_new_session(token)
    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    # prompt inicial para o mestre - ele pode usar nome do personagem se já houver ficha carregada
    ficha = None
    if os.path.exists(ficha_json_path(token)):
        with open(ficha_json_path(token), "r", encoding="utf-8") as f:
            ficha = json.load(f)

    player_name = ficha.get("name") if ficha else "Jogador"

    prompt = (
        f"Você é o Mestre de RPG desta campanha. O personagem do jogador se chama: {player_name}.\n\n"
        "Produza uma introdução épica, sensorial e cinematográfica. Apresente o local, um acontecimento incitante e um NPC relevante.\n"
        "Ao final, inclua o jogador no ambiente e faça UMA pergunta clara pedindo apenas o NOME do personagem (apenas nesta primeira cena) ou informe como deseja prosseguir.\n"
        "Não revele regras; mantenha tom sombrio e fantástico."
    )

    response = query_engine.query(prompt)

    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"Narrador: {response}\n")

    increment_turn(token)
    return str(response)

def next_action(token, action):
    meta = increment_turn(token)
    if meta and meta.get("closed", False):
        return "Esta campanha já foi encerrada."

    # carrega ficha se existir
    ficha = {}
    if os.path.exists(ficha_json_path(token)):
        with open(ficha_json_path(token), "r", encoding="utf-8") as f:
            ficha = json.load(f)

    # decide se precisa de check
    needs, attr, skill, dc = detect_check_from_action(action)
    # se precisa, setamos um estado aguardando roll no session_state
    if needs:
        # salva no arquivo de contexto um pedido de roll (para referência)
        # marcamos em session_state: awaiting_roll, roll_meta
        st.session_state.awaiting_roll = True
        st.session_state.roll_meta = {"attr": attr, "skill": skill, "dc": dc, "player_token": token}
        # escreve no histórico que o mestre solicitou roll — mas vamos notificar apenas via UI
        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"Jogador: {action}\nMestre: Solicita teste de {skill} ({attr}) contra DC {dc}.\n")
        return f"O Mestre solicitou um teste de **{skill}** (atributo {attr}) contra DC {dc}. Clique em 'Rolar D20' para tentar."
    else:
        # sem check -> manda direto para o mestre gerar consequência
        index = load_or_create_index()
        query_engine = index.as_query_engine(similarity_top_k=3)
        history = get_history_window(token)

        prompt = (
            "Você é o Mestre de RPG. Continue a narrativa SEMPRE avançando a história.\n"
            "Regras: não revele mecânicas; reaja ao que o jogador fez; introduza consequência; finalize com pergunta clara.\n\n"
            f"Histórico:\n{history}\n\n"
            f"Ação do jogador:\n{action}\n\n"
            "Continue a história descrevendo a consequência imediata e finalizando com uma pergunta sobre a próxima ação."
        )
        response = query_engine.query(prompt)

        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"Jogador: {action}\nNarrador: {response}\n")
        return str(response)

# ---------------------------
# FUNÇÃO: aplicar resultado do roll (quando jogador clica rolar)
# ---------------------------
def apply_roll_and_inform_master(token):
    """
    Usa st.session_state.roll_meta para resolver check, grava resultado e chama mestre pedindo narração baseada no resultado.
    """
    if not st.session_state.get("roll_meta"):
        return "Nenhum teste pendente."

    meta = st.session_state.roll_meta
    attr = meta["attr"]
    dc = meta["dc"]
    skill = meta["skill"]

    # carrega ficha
    ficha = {}
    if os.path.exists(ficha_json_path(token)):
        with open(ficha_json_path(token), "r", encoding="utf-8") as f:
            ficha = json.load(f)

    res = resolve_check(ficha, attr, dc)
    # grava resultado no contexto
    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"RESULTADO_TESTE: {res['message']}\n")

    # agora chama o mestre LLM para descrever consequência com base no resultado
    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)
    history = get_history_window(token)

    # inclua ficha resumida para contexto (sem expor mecânicas detalhadas)
    ficha_resumo = ""
    if ficha:
        name = ficha.get("name", "personagem")
        attrs = ", ".join([f"{k}:{v}" for k, v in ficha.get("attributes", {}).items()])
        ficha_resumo = f"Personagem: {name}. Atributos: {attrs}."

    prompt = (
        "Você é o Mestre de RPG. Um teste acabou de ser rolado e o resultado abaixo deve ser usado para narrar a consequência.\n\n"
        f"Histórico:\n{history}\n\n"
        f"Ficha resumida:\n{ficha_resumo}\n\n"
        f"Teste: {skill} ({attr}) contra DC {dc}.\n"
        f"Resultado do teste: {res['message']}\n\n"
        "Com base nisso, descreva a consequência imediata, reações de NPCs/ambiente e finalize com uma pergunta clara ao jogador."
    )
    response = query_engine.query(prompt)

    # grava narração
    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"RESULTADO_TESTE_DETALHADO: {res['message']}\nNarrador: {response}\n")

    # limpa estado de roll
    st.session_state.awaiting_roll = False
    st.session_state.roll_meta = None

    return {"narration": str(response), "roll_result": res}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Mestre-IA RPG", page_icon="🎲", layout="wide")

st.title("🎲 Mestre-IA RPG — Long Shot com Referee (D20)")

# sessão id fixa por usuário
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "awaiting_roll" not in st.session_state:
    st.session_state.awaiting_roll = False
if "roll_meta" not in st.session_state:
    st.session_state.roll_meta = None

st.sidebar.markdown(f"**Sessão:** `{st.session_state.session_id}`")

# ---------------------------
# UPLOAD DE FICHA (PDF)
# ---------------------------
st.sidebar.subheader("📄 Upload de Ficha (PDF)")
uploaded = st.sidebar.file_uploader("Envie a ficha do personagem (PDF)", type=["pdf"])
if uploaded is not None:
    token = st.session_state.session_id
    pdf_path = ficha_pdf_path(token)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())
    # parse
    ficha = parse_ficha_pdf_to_json(pdf_path)
    with open(ficha_json_path(token), "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False)
    st.sidebar.success(f"Ficha salva. Personagem: {ficha.get('name')}. Atributos: {ficha.get('attributes')}")

# ---------------------------
# CONTROLES DO BANCO VETORIAL
# ---------------------------
st.sidebar.subheader("📚 Banco Vetorial")
if st.sidebar.button("🔨 Build Index"):
    qnt = build_index()
    st.sidebar.success(f"Banco vetorial reconstruído com {qnt} docs.")
if st.sidebar.button("🔄 Update Index"):
    qnt = update_index()
    st.sidebar.success(f"Banco vetorial atualizado com {qnt} docs.")

# ---------------------------
# INICIAR HISTÓRIA
# ---------------------------
st.subheader("🎬 Início da Campanha")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🚀 Iniciar História"):
        intro = start_story(st.session_state.session_id)
        st.session_state.history = [{"autor": "mestre", "texto": intro}]
        st.rerun()


# ---------------------------
# EXIBIR HISTÓRICO COM ÍCONE
# ---------------------------
st.markdown("---")
for msg in st.session_state.history:
    if msg["autor"] == "user":
        st.markdown(f"**Você:** {msg['texto']}")
    else:
        cols = st.columns([0.5, 9.5])
        with cols[0]:
            if os.path.exists("master.png"):
                st.image("master.png", width=48)
            else:
                st.write("🧙")
        with cols[1]:
            st.markdown(f"**Mestre:** {msg['texto']}")
st.markdown("---")

# ---------------------------
# SE HOUVER ROLL PENDENTE -> MOSTRA BOTAO ANIMADO
# ---------------------------
if st.session_state.awaiting_roll:
    st.info("O Mestre solicitou um teste. Clique abaixo para rolar D20.")
    # animação simples: mostra loading por 0.8s
    if st.button("🎲 Rolar D20"):
        with st.spinner("Jogando o dado..."):
            time.sleep(0.8)
            out = apply_roll_and_inform_master(st.session_state.session_id)
            # anexa ao histórico a narração do mestre
            if isinstance(out, dict):
                st.session_state.history.append({"autor": "user", "texto": f"(Rolagem feita) {out['roll_result']['message']}"})
                st.session_state.history.append({"autor": "mestre", "texto": out["narration"]})
            else:
                st.session_state.history.append({"autor": "mestre", "texto": str(out)})
            st.experimental_rerun()

# ---------------------------
# INPUT DO JOGADOR (AÇÕES)
# ---------------------------
st.subheader("⚔ Sua Ação")
user_input = st.text_input("Digite sua ação (ex: 'Atacar com arco', 'Tentar convencer o guarda'):")

if st.button("📩 Enviar ação"):
    if user_input.strip():
        st.session_state.history.append({"autor": "user", "texto": user_input})
        resposta = next_action(st.session_state.session_id, user_input)
        # se mestre solicitou roll, next_action já colocou awaiting_roll e escreveu instrução
        # se resposta for texto do mestre, adicionamos
        st.session_state.history.append({"autor": "mestre", "texto": resposta})
        st.rerun()


# ---------------------------
# DEBUG / INFO
# ---------------------------
with st.expander("🔧 Debug / Estado"):
    st.write("awaiting_roll:", st.session_state.awaiting_roll)
    st.write("roll_meta:", st.session_state.roll_meta)
    token = st.session_state.session_id
    if os.path.exists(ficha_json_path(token)):
        with open(ficha_json_path(token), "r", encoding="utf-8") as f:
            st.write("Ficha carregada:", json.load(f))
    st.write("Context file preview:")
    if os.path.exists(get_context_file(token)):
        with open(get_context_file(token), "r", encoding="utf-8") as f:
            st.text("\n".join(f.readlines()[-30:]))

st.caption("Qualquer coisa, vai nos avisando.")
