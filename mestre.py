# rpg_server.py
import os
import uuid
from flask import Flask, request, jsonify
from pypdf import PdfReader
from tqdm import tqdm

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

os.makedirs(CONTEXT_DIR, exist_ok=True)
app = Flask(__name__)


# ==============================
# CONFIG GLOBAL (CPU ONLY)
# ==============================
def configure_global_settings():
    Settings.llm = Ollama(
        model="mistral:7b",
        temperature=0.9,
        request_timeout=600,
        llm_kwargs={"num_gpu_layers": 0}  # força CPU
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

configure_global_settings()


# ==============================
# FUNÇÕES UTILITÁRIAS
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


def get_context_file(token):
    return os.path.join(CONTEXT_DIR, f"{token}.txt")


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

    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    prompt = (
        "Responda sempre em português. "
        "Você é um mestre de RPG que inicia uma aventura épica em um mundo de fantasia. "
        "Descreva o cenário inicial, o clima e apresente o herói ou o grupo de forma envolvente."
    )

    response = query_engine.query(prompt)

    # Salvar história inicial
    context_path = get_context_file(token)
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(f"Narrador: {response}\n")

    return jsonify({"token": token, "intro": str(response)})


@app.route("/next_action", methods=["POST"])
def next_action():
    data = request.get_json()
    token = data.get("token")
    action = data.get("action")

    if not token or not action:
        return jsonify({"error": "Campos obrigatórios: token, action"}), 400

    context_path = get_context_file(token)
    if not os.path.exists(context_path):
        return jsonify({"error": "Token de sessão inválido."}), 404

    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Carrega histórico anterior
    with open(context_path, "r", encoding="utf-8") as f:
        history = f.read()

    prompt = (
        "Continue a narrativa de RPG abaixo, mantendo coerência, clima, personagens e estilo.\n\n"
        "Responda sempre em português.\n\n"
        f"História até aqui:\n{history}\n\n"
        f"Ação do jogador: {action}"
    )

    response = query_engine.query(prompt)

    # Atualiza histórico
    with open(context_path, "a", encoding="utf-8") as f:
        f.write(f"Jogador: {action}\nNarrador: {response}\n")

    return jsonify({"response": str(response), "token": token})


# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
