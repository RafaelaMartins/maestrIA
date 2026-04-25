import os
from pypdf import PdfReader
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, StorageContext, Document, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama  # ✅ o correto


PDF_DIR = "./DB"
INDEX_DIR = "./rpg_index"

# ========= 1) Extração dos PDFs =========
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


# ========= 2) Criação ou carregamento do índice =========
def build_or_load_index(docs):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ sempre HuggingFace

    if os.path.exists(INDEX_DIR):
        print("🔁 Carregando índice existente...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)  # ✅ passa explicitamente o modelo
    else:
        print("⚙️ Criando novo índice vetorial...")
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)

    return index


# ========= 3) Loop de chat =========
def start_rpg_chat(index):
    print("\n🎲 Bem-vindo ao RPG Narrativo!")
    print("Você pode digitar comandos como:")
    print("  → 'explorar a floresta', 'falar com o mago', 'lutar com o dragão', etc.")
    print("  → 'sair' para encerrar o jogo.\n")

    llm = Ollama(model="mistral:7b", temperature=0.9, request_timeout=600)
    query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)

    history = []
    context_prompt = (
        "Você é um mestre de RPG que cria histórias imersivas e consistentes. "
        "Use as informações dos PDFs como referência de ambientação, criaturas e enredos. "
        "Continue a narrativa com base nas decisões do jogador."
    )

    while True:
        user_input = input("🗡️ Sua ação: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("🏁 Fim da aventura!")
            break

        full_prompt = (
            context_prompt
            + "\nHistórico da história até agora:\n"
            + "\n".join(history[-5:])
            + "\nNova ação do jogador: "
            + user_input
        )

        response = query_engine.query(full_prompt)
        print(f"\n📖 {response}\n")

        history.append(f"Jogador: {user_input}")
        history.append(f"Narrador: {response}")


# ========= 4) Execução =========
if __name__ == "__main__":
    docs = extract_pdfs_to_documents(PDF_DIR)
    print(f"✅ Total de documentos extraídos: {len(docs)}")
    index = build_or_load_index(docs)
    start_rpg_chat(index)
