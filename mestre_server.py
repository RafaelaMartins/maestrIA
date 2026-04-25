# rpg_server.py
import os
import uuid
import json
from flask import Flask, request, jsonify
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


# ==============================
# CONFIGURAÇÕES
# ==============================
PDF_DIR = "./DB"
INDEX_DIR = "./rpg_index"
CONTEXT_DIR = "./contexts"
META_DIR = os.path.join(CONTEXT_DIR, "meta")
os.makedirs(CONTEXT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

MAX_TURNS_DEFAULT = 10000
WINDOW_TURNS = 12
LANG = "português"

app = Flask(__name__)


# ==============================
# CONFIG GLOBAL (CPU ONLY)
# ==============================
def configure_global_settings():
    Settings.llm = Ollama(
        model="mistral:7b",
        temperature=0.9,
        request_timeout=600,
        llm_kwargs={"num_gpu_layers": 0}
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

configure_global_settings()


# ==============================
# UTILITÁRIAS DE SESSÃO / META
# ==============================
def get_context_file(token):
    return os.path.join(CONTEXT_DIR, f"{token}.txt")

def get_meta_file(token):
    return os.path.join(META_DIR, f"{token}.meta.json")

def create_new_session(token, max_turns=MAX_TURNS_DEFAULT):
    ctx_path = get_context_file(token)
    meta_path = get_meta_file(token)
    now = datetime.utcnow().isoformat()
    initial_meta = {
        "token": token,
        "created_at": now,
        "last_updated": now,
        "turns": 0,
        "closed": False,
        "max_turns": max_turns
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(initial_meta, f)
    with open(ctx_path, "w", encoding="utf-8") as f:
        f.write(f"[SESSION_CREATED_AT:{now}]\n")
    return initial_meta

def load_meta(token):
    meta_path = get_meta_file(token)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(token, meta):
    meta["last_updated"] = datetime.utcnow().isoformat()
    with open(get_meta_file(token), "w", encoding="utf-8") as f:
        json.dump(meta, f)

def increment_turn(token):
    meta = load_meta(token)
    if meta is None:
        return None
    meta["turns"] += 1
    save_meta(token, meta)
    return meta

def mark_closed(token):
    meta = load_meta(token)
    if meta is None:
        return None
    meta["closed"] = True
    save_meta(token)
    return meta

def is_closed(token):
    meta = load_meta(token)
    return (meta is not None) and meta.get("closed", False)


# ==============================
# PDF -> DOCUMENTS
# ==============================
def extract_pdfs_to_documents(pdf_dir):
    documents = []
    for file in tqdm(os.listdir(pdf_dir), desc="Extraindo PDFs"):
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
                metadata = {
                    "source": file,
                    "page": i + 1,
                    "total_pages": num_pages,
                    "category": "revista_rpg",
                }
                documents.append(Document(text=text, metadata=metadata))
        except Exception as e:
            print(f"❌ Erro ao ler {file}: {e}")
    return documents

def load_or_create_index(docs=None):
    if os.path.exists(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    else:
        if docs is None:
            raise ValueError("Nenhum documento fornecido para criar o índice.")
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    return index


# ==============================
# HISTORY & PROMPTS
# ==============================
def get_history_window(token, max_lines=WINDOW_TURNS):
    path = get_context_file(token)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    snippet = "\n".join(lines[-max_lines:])
    return snippet


# ==============================
# ROTAS
# ==============================
@app.route("/build_index", methods=["POST"])
def build_index():
    docs = extract_pdfs_to_documents(PDF_DIR)
    load_or_create_index(docs)
    return jsonify({"message": f"Índice criado com {len(docs)} documentos."})

@app.route("/update_index", methods=["POST"])
def update_index():
    docs = extract_pdfs_to_documents(PDF_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    for doc in docs:
        index.insert(doc)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return jsonify({"message": "Índice atualizado com novos PDFs."})

@app.route("/start_story", methods=["GET"])
def start_story():
    token = str(uuid.uuid4())
    create_new_session(token)

    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    prompt = (
        "Responda sempre em português.\n"
        "Você é um mestre de RPG que inicia uma aventura épica. "
        "Descreva o cenário inicial e apresente o herói ou grupo. "
        "Ao final, faça UMA pergunta clara ao jogador, com 2-4 opções e também resposta aberta."
    )

    response = query_engine.query(prompt)

    context_path = get_context_file(token)
    with open(context_path, "a", encoding="utf-8") as f:
        f.write(f"Narrador: {response}\n")

    increment_turn(token)

    return jsonify({"token": token, "intro": str(response)})


@app.route("/next_action", methods=["POST"])
def next_action():
    data = request.get_json(force=True)
    token = data.get("token")
    action = data.get("action")

    if not token or not action:
        return jsonify({"error": "Campos obrigatórios: token, action"}), 400

    meta = load_meta(token)
    if meta is None:
        return jsonify({"error": "Token de sessão inválido."}), 404
    if meta.get("closed", False):
        return jsonify({"error": "Sessão já finalizada."}), 400

    meta = increment_turn(token)
    if meta is None:
        return jsonify({"error": "Falha ao atualizar meta."}), 500

    # ================= FINAL ÉPICO =================
    if meta["turns"] >= meta.get("max_turns", MAX_TURNS_DEFAULT):
        index = load_or_create_index()
        query_engine = index.as_query_engine(similarity_top_k=3)

        history_snippet = get_history_window(token, max_lines=WINDOW_TURNS * 2)
        final_prompt = (
            "Gere um DESFECHO ÉPICO desta campanha.\n"
            "Considere o histórico:\n"
            f"{history_snippet}\n\n"
            f"Ação final do jogador: {action}"
        )

        final_response = query_engine.query(final_prompt)

        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"Jogador: {action}\nNarrador (DESFECHO): {final_response}\n")
        mark_closed(token)

        return jsonify({
            "response": str(final_response),
            "token": token,
            "ended": True
        })

    # =============== FLUXO NORMAL ===============
    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    history = get_history_window(token, max_lines=WINDOW_TURNS)

    # 🔥 PROMPT PEDIDO POR VOCÊ 🔥
    prompt = (
        "Você é o mestre de RPG desta campanha. Continue a história SEMPRE avançando a "
        "narrativa, independentemente da ação do jogador ser vaga, curta ou pouco específica.\n\n"

        "Regras obrigatórias:\n"
        "1. Nunca peça a mesma ação novamente.\n"
        "2. Nunca devolva a pergunta ao jogador. Interprete a intenção dele e avance.\n"
        "3. Use o conhecimento do índice como referência de mundo, mas mantenha a fluidez.\n"
        "4. Sempre gere um acontecimento novo.\n"
        "5. No final de cada resposta, faça UMA pergunta que avance a trama.\n"
        "6. A narrativa deve sempre ter consequência, risco, recompensa ou descoberta.\n"
        "7. Mantenha tom cinematográfico e ação contínua.\n"
        "8. Responda sempre em português.\n\n"

        "História até aqui:\n"
        f"{history}\n\n"
        "Ação do jogador:\n"
        f"{action}\n\n"

        "Agora continue a história com base na ação acima."
    )

    response = query_engine.query(prompt)

    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"Jogador: {action}\nNarrador: {response}\n")

    return jsonify({"response": str(response), "token": token, "ended": False})


# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
