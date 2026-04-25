import streamlit as st
import os
import json
import re
import sys
import random
import asyncio
from typing import Union
from datetime import datetime
from pypdf import PdfReader

# FIX: Windows Event Loop Issue
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
    Context
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ---------------------------
# CONFIGURAÇÃO GLOBAL
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "DB")
INDEX_DIR = os.path.join(BASE_DIR, "rpg_index_v2")
CONTEXT_DIR = os.path.join(BASE_DIR, "contexts_v2")
FICHAS_DIR = os.path.join(CONTEXT_DIR, "fichas")

for d in [DB_DIR, INDEX_DIR, CONTEXT_DIR, FICHAS_DIR]:
    os.makedirs(d, exist_ok=True)

# CONFIGURAÇÃO GLOBAL DO LLAMAINDEX
Settings.llm = Ollama(model="mistral:7b", request_timeout=1200.0, additional_kwargs={"num_gpu": 0})
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def init_llm():
    """Re-init LLM to bind to current async loop"""
    Settings.llm = Ollama(model="mistral:7b", request_timeout=1200.0, additional_kwargs={"num_gpu": 0})

# ---------------------------
# 1. LORE ENGINE (RAG)
# ---------------------------
class LoreEngine:
    """
    Gerencia o conhecimento do mundo (Regras + Lore) usando RAG.
    """
    def __init__(self):
        self.index = self._load_or_create_index()
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    def _load_or_create_index(self):
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
                return load_index_from_storage(storage_context)
            except Exception as e:
                print(f"Erro ao carregar index: {e}. Recriando...")
        
        return self._build_index()

    def _build_index(self):
        documents = []
        # Tenta ler PDFs do diretório DB
        if os.path.exists(DB_DIR):
            files = [f for f in os.listdir(DB_DIR) if f.endswith(".pdf")]
            if files:
                print(f"Iniciando indexação de {len(files)} arquivos...")
                # Mostra progresso no Streamlit se estiver rodando lá
                progress_bar = st.progress(0, text="Indexando Biblioteca de RPG (Isso pode demorar na primeira vez)...")
                
                for i, f in enumerate(files):
                    path = os.path.join(DB_DIR, f)
                    try:
                        reader = PdfReader(path)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        documents.append(Document(text=text, metadata={"source": f}))
                    except Exception as e:
                        print(f"Erro lendo {f}: {e}")
                    
                    # Atualiza barra
                    progress = (i + 1) / len(files)
                    progress_bar.progress(progress, text=f"Lendo {f} ({i+1}/{len(files)})")
                
                progress_bar.empty()

        if not documents:
            documents.append(Document(text="Regras básicas de RPG: D20 para testes. 10 é médio.", metadata={"source": "dummy"}))

        print("Criando embeddings...")
        with st.spinner("Gerando Embeddings (Finalizando)..."):
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=INDEX_DIR)
        return index

    def query_lore(self, query: str) -> str:
        response = self.query_engine.query(query)
        return str(response)

# ---------------------------
# 2. CHARACTER PARSER
# ---------------------------
class CharacterParser:
    """
    Extrai dados da ficha PDF com heurísticas melhoradas e LLM para correção.
    """
    @staticmethod
    def parse_pdf(pdf_path: str) -> dict:
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            return {"name": "Desconhecido", "attributes": {}, "error": str(e)}

        # Extração via LLM para garantir precisão no Nome e Atributos
        prompt = f"""
        Analise o texto cru de uma ficha de RPG abaixo e extraia um JSON.
        
        TEXTO DA FICHA:
        {text[:3000]}
        
        OBJETIVO:
        Retorne APENAS um JSON válido com este formato:
        {{
            "name": "Nome do Personagem (se não achar, invente um baseado na classe/raça)",
            "attributes": {{
                "FOR": valor_inteiro,
                "DES": valor_inteiro,
                "CON": valor_inteiro,
                "INT": valor_inteiro,
                "SAB": valor_inteiro,
                "CAR": valor_inteiro
            }},
            "class": "Classe/Raça se houver",
            "background": "Resumo breve do histórico se houver"
        }}
        
        Se algum atributo não estiver claro, assuma 10.
        """
        
        try:
            response = Settings.llm.complete(prompt).text
            # Limpeza básica para pegar o JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except Exception as e:
            print(f"Erro no LLM Parse: {e}")

        return {"name": "Aventureiro", "attributes": {"FOR":10, "DES":10, "CON":10, "INT":10, "SAB":10, "CAR":10}}

# ---------------------------
# 3. EVENTS (WORKFLOW)
# ---------------------------
class UserActionEvent(Event):
    action: str
    session_id: str
    char_name: str

class AnalysisEvent(Event):
    """Resultado da análise do Árbitro"""
    requires_roll: bool
    skill: str
    attr: str
    dc: int
    reason: str
    action_text: str
    session_id: str
    char_name: str

class RollRequestEvent(Event):
    """Solicitação de rolagem para a UI"""
    skill: str
    attr: str
    dc: int
    reason: str
    action_text: str
    session_id: str
    char_name: str
    message_to_player: str

class RollResultEvent(Event):
    """Resultado da rolagem vindo da UI"""
    roll_total: int
    roll_natural: int
    dc: int
    skill: str
    action_text: str
    session_id: str
    char_name: str

class NarrativeEvent(Event):
    """Gatilho para o Narrador gerar a resposta final"""
    input_text: str
    context_type: str  # 'direct_action' ou 'roll_result'
    roll_details: dict = None
    session_id: str
    char_name: str

class GameResponseEvent(Event):
    """Resposta final para a UI"""
    text: str
    session_id: str

# ---------------------------
# 4. GAME WORKFLOW
# ---------------------------
class GameWorkflow(Workflow):
    def __init__(self, timeout=1200, verbose=True):
        super().__init__(timeout=timeout, verbose=verbose)
        # Lazy load or managed externally, but for now init here
        # We will cache the LoreEngine instance using st.cache_resource wrapper outside
        self.lore = get_lore_engine() 
    
    @step
    async def analyze_action(self, ev: StartEvent) -> Union[NarrativeEvent, StopEvent]:
        """
        AGENTE ÁRBITRO: Analisa a intenção e decide se precisa de teste.
        Entrada: StartEvent com 'action' e 'session_id'.
        """
        action = ev.get("action")
        session_id = ev.get("session_id")
        char_name = ev.get("char_name", "Aventureiro")
        
        if not action:
            return None

        # AUTO-START LOGIC
        if action == "INTRO_START":
            return NarrativeEvent(
                input_text="INTRO_START",
                context_type="intro",
                session_id=session_id,
                char_name=char_name
            )

        # MANUAL ROLL DETECTION
        # Detects: "[SISTEMA] O jogador rolou um dado manualmente e tirou: 15."
        manual_roll_match = re.search(r"\[SISTEMA\].*rolou.*tirou:\s*(\d+)", action)
        if manual_roll_match:
            roll_val = int(manual_roll_match.group(1))
            print(f"--> Rolagem Manual Detectada: {roll_val}")
            return NarrativeEvent(
                input_text=action,
                context_type="manual_roll",
                roll_details={"roll_total": roll_val, "dc": 0, "skill": "Manual"},
                session_id=session_id,
                char_name=char_name
            )

        print(f"--> Árbitro analisando: {action}")
        
        prompt = f"""
        Você é o ÁRBITRO do RPG. Analise a ação do jogador {char_name}: "{action}".
        
        Decida se é necessário um teste de dados (D20).
        - Ações triviais (olhar, falar, andar) -> NÃO requer teste.
        - Ações de risco/combate/perícia (atacar, escalar, mentir, magia) -> REQUER teste.
        
        FORMATO DE RESPOSTA (JSON PURO, SEM MARKDOWN):
        {{
            "requires_roll": true,
            "skill": "Nome da Perícia",
            "attr": "FOR/DES/CON/INT/SAB/CAR",
            "dc": 15,
            "reason": "Explicação",
            "message_to_player": "Mensagem pedindo a rolagem"
        }}
        
        OU
        
        {{
            "requires_roll": false,
            "skill": "",
            "attr": "",
            "dc": 0,
            "reason": "Ação simples",
            "message_to_player": ""
        }}
        """
        
        response = await Settings.llm.acomplete(prompt)
        text_response = response.text
        print(f"--> Resposta do Árbitro: {text_response}") # Debug
        
        try:
            # Try to find JSON block
            json_match = re.search(r"\{.*\}", text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix common LLM JSON errors
                json_str = json_str.replace("'", '"') # Replace single quotes
                json_str = json_str.replace("True", "true").replace("False", "false") # Fix booleans
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
        except Exception as e:
            print(f"Erro parseando JSON do Árbitro: {e}")
            # Fallback: If text contains "teste" or "CD", assume roll needed
            if "teste" in text_response.lower() or "cd" in text_response.lower():
                 data = {
                     "requires_roll": True, 
                     "skill": "Sorte", 
                     "attr": "DES", 
                     "dc": 10, 
                     "reason": "Fallback de detecção", 
                     "message_to_player": "O Mestre pede um teste, mas houve um erro de comunicação. Faça um teste de Sorte (DES) CD 10."
                 }
            else:
                data = {"requires_roll": False, "skill": "", "attr": "", "dc": 0, "reason": "Erro parse", "message_to_player": ""}

        if data.get("requires_roll"):
            # Wrap in StopEvent so the workflow returns this object to the UI
            req = RollRequestEvent(
                skill=data.get("skill", "Geral"),
                attr=data.get("attr", "DES"),
                dc=int(data.get("dc", 10)),
                reason=data.get("reason", "Ação incerta"),
                action_text=action,
                session_id=session_id,
                char_name=char_name,
                message_to_player=data.get("message_to_player", f"Teste de {data.get('skill', 'Geral')} necessário.")
            )
            return StopEvent(result=req)
        else:
            return NarrativeEvent(
                input_text=action,
                context_type="direct_action",
                session_id=session_id,
                char_name=char_name
            )

    @step
    async def generate_narrative(self, ev: NarrativeEvent) -> StopEvent:
        """
        AGENTE NARRADOR: Gera a resposta final.
        """
        print(f"--> Narrador gerando para: {ev.context_type}")
        
        lore_context = self.lore.query_lore(f"Contexto relevante para: {ev.input_text}")
        char_name = ev.char_name
        
        # Get History Context
        history_context = ""
        try:
            if "history" in st.session_state:
                # Get last 3 messages to understand context
                msgs = st.session_state.history[-3:]
                history_context = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in msgs])
        except:
            pass

        # COMMON RULES
        base_instructions = f"""
        REGRAS DE OURO:
        1. NUNCA use a palavra "NPC". Use "personagem", "figura", "homem", "mulher" ou o nome deles.
        2. Seja imersivo. Descreva cheiros, sons, clima e sensações.
        3. SEMPRE termine com uma situação que exija ação do jogador ({char_name}). Pergunte "O que você faz?" ou coloque-o em perigo imediato.
        4. O nome do protagonista é **{char_name}**. Use-o.
        5. RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.
        6. SEJA DECISIVO. Se o jogador falhou, ele FALHOU. Não diga "talvez". Descreva a consequência ruim.
        """

        if ev.context_type == "intro":
            prompt = f"""
            Você é o MESTRE NARRADOR. Inicie a aventura para {char_name}.
            
            Lore do Mundo: {lore_context}
            
            Instruções:
            1. Descreva onde {char_name} está (ambiente, clima, hora do dia).
            2. Introduza um evento inicial ou mistério.
            3. Se houver outros personagens, descreva-os organicamente (SEM usar "NPC").
            4. Crie uma tensão imediata.
            {base_instructions}
            """

        elif ev.context_type == "roll_result":
            r = ev.roll_details
            diff = r['roll_total'] - r['dc']
            
            outcome = "SUCESSO" if diff >= 0 else "FALHA"
            nuance = ""
            if diff >= 5: nuance = "Sucesso Crítico/Espetacular"
            elif diff >= 0: nuance = "Sucesso Marginal/Com Custo"
            elif diff >= -5: nuance = "Falha Simples"
            else: nuance = "Falha Crítica/Desastrosa"

            prompt = f"""
            Você é o MESTRE NARRADOR.
            Personagem: {char_name}
            Ação Tentada: "{ev.input_text}"
            Resultado do Dado: {outcome} ({nuance}).
            Detalhes: Rolou {r['roll_total']} contra CD {r['dc']}.
            
            Lore Relevante: {lore_context}
            
            Instruções:
            1. Narre a ação de {char_name} baseada ESTRITAMENTE no resultado.
            2. SE FALHA ({outcome} == FALHA): O personagem NÃO CONSEGUE o que queria. Descreva a falha e uma consequência negativa imediata (dano, queda, ser descoberto, perder algo). NÃO AMENIZE.
            3. SE SUCESSO ({outcome} == SUCESSO): O personagem CONSEGUE. Descreva a ação sendo realizada com êxito.
            4. Avance a história a partir desse novo estado.
            {base_instructions}
            """
        
        elif ev.context_type == "manual_roll":
            r = ev.roll_details
            prompt = f"""
            Você é o MESTRE NARRADOR.
            Personagem: {char_name}
            
            CONTEXTO RECENTE (O que estava acontecendo):
            {history_context}
            
            AÇÃO: O jogador decidiu rolar um dado manualmente para resolver a situação acima.
            RESULTADO DA ROLAGEM: {r['roll_total']} (D20).
            
            Lore Relevante: {lore_context}
            
            Instruções:
            1. Analise o CONTEXTO RECENTE para entender o que o jogador estava tentando fazer.
            2. Julgue o resultado:
               - 1 a 9: FALHA / AZAR. A ação dá errado. Descreva a consequência ruim.
               - 10 a 20: SUCESSO / SORTE. A ação dá certo. Descreva o êxito.
            3. Narre a cena de forma conclusiva.
            {base_instructions}
            """

        else:
            prompt = f"""
            Você é o MESTRE NARRADOR.
            Personagem: {char_name}
            Ação do Jogador: "{ev.input_text}"
            
            Lore Relevante: {lore_context}
            
            Instruções:
            1. Responda à ação de {char_name} de forma imersiva.
            2. Descreva o ambiente e reações dos personagens.
            3. Avance a história.
            {base_instructions}
            """
            
        response = await Settings.llm.acomplete(prompt)
        return StopEvent(result=str(response.text))

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="MestrIA v2", layout="wide")

def apply_custom_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@400;700&display=swap');

        /* GLOBAL THEME */
        .stApp {
            background-color: #0e1117;
            background-image: radial-gradient(circle at center, #1c1f26 0%, #0e1117 100%);
            color: #e0e0e0;
            font-family: 'Lato', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Cinzel', serif;
            color: #ffc107 !important;
            text-shadow: 2px 2px 4px #000000;
        }

        /* HEADER */
        header[data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        
        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #161920;
            border-right: 2px solid #2d333b;
        }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(to bottom, #2d333b, #1c1f26);
            color: #ffc107;
            border: 1px solid #ffc107;
            font-family: 'Cinzel', serif;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(to bottom, #3e4652, #2d333b);
            box-shadow: 0 0 10px #ffc107;
            border-color: #ffffff;
            color: #ffffff;
        }

        /* CHAT BUBBLES */
        [data-testid="stChatMessage"] {
            background-color: #1c1f26;
            border: 1px solid #2d333b;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        }
        [data-testid="stChatMessage"][data-testid="user"] {
            background-color: #232830;
            border-left: 3px solid #4a90e2;
        }
        [data-testid="stChatMessage"][data-testid="assistant"] {
            background-color: #1a1d24;
            border-left: 3px solid #ffc107;
        }

        /* INPUT AREA */
        .stTextInput > div > div > input {
            background-color: #1c1f26;
            color: #e0e0e0;
            border: 1px solid #2d333b;
        }
        
        /* TOAST */
        div[data-baseweb="toast"] {
            background-color: #2d333b !important;
            color: #ffc107 !important;
            border: 1px solid #ffc107;
        }
        
        /* SPINNER & LOADING TEXT */
        .stSpinner > div > div {
            color: #ffffff !important;
        }
        
        /* FILE UPLOADER */
        [data-testid="stFileUploader"] {
            color: #ffffff;
        }
        [data-testid="stFileUploader"] small {
            color: #ffffff !important;
        }
        [data-testid="stFileUploader"] span {
            color: #ffffff !important;
        }
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #1c1f26;
            border: 1px dashed #4a90e2;
        }
        section[data-testid="stFileUploaderDropzone"] button {
            color: #000000 !important; /* Browse button text black */
            background-color: #e0e0e0 !important; /* Button background light */
            border: none;
        }
        
        /* GENERAL TEXT OVERRIDES */
        .stMarkdown, .stText, p, label {
            color: #e0e0e0 !important;
        }
        
        /* SPECIFIC ICON FIX (Deploy/Loading/Cache) */
        .st-emotion-cache-scp8yw, .e3g0k5y6, .st-emotion-cache-1wbqy5l, .stStatusWidget {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        
        /* Force all SVGs in the header/status area to be white */
        header svg, [data-testid="stStatusWidget"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }
        
        /* STOP BUTTON & HEADER BUTTONS */
        header button {
            color: #ffffff !important;
        }
        
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

@st.cache_resource(show_spinner=False)
def get_lore_engine():
    return LoreEngine()

if "workflow" not in st.session_state:
    # Custom spinner with white text
    with st.spinner("Inicializando MestrIA (Carregando Lore...)..."):
        # Force load lore engine first to show progress
        get_lore_engine()
        st.session_state.workflow = GameWorkflow(timeout=1200, verbose=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "char_data" not in st.session_state:
    st.session_state.char_data = None
if "pending_roll" not in st.session_state:
    st.session_state.pending_roll = None # Stores RollRequestEvent
if "game_started" not in st.session_state:
    st.session_state.game_started = False

st.title("🐉 Maestr-IA RPG ")

# Sidebar - Ficha
with st.sidebar:
    st.header("Ficha do Personagem")
    uploaded_file = st.file_uploader("Upload PDF da Ficha", type="pdf")
    if uploaded_file and not st.session_state.char_data:
        path = os.path.join(FICHAS_DIR, "temp_ficha.pdf")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("Lendo ficha com IA..."):
            data = CharacterParser.parse_pdf(path)
            st.session_state.char_data = data
            st.success("Ficha processada!")
            st.rerun() # Rerun to trigger confirmation UI
            
    if st.session_state.char_data:
        c = st.session_state.char_data
        
        # Only show sidebar edit if game started, otherwise main area handles it
        if st.session_state.game_started:
            new_name = st.text_input("Nome do Personagem", value=c.get("name", "Aventureiro"))
            if new_name != c.get("name"):
                c["name"] = new_name
                st.session_state.char_data = c
                st.rerun()
            
        st.subheader(c.get("name", "Sem Nome"))
        st.write(f"**Classe:** {c.get('class', 'N/A')}")
        st.json(c.get("attributes", {}))

# CONFIRMATION SCREEN
if st.session_state.char_data and not st.session_state.game_started:
    st.info("Ficha carregada com sucesso! Confirme os dados para iniciar.")
    
    c = st.session_state.char_data
    col1, col2 = st.columns([3, 1])
    
    with col1:
        confirm_name = st.text_input("Nome do Personagem (Edite se necessário):", value=c.get("name", "Aventureiro"))
    
    with col2:
        st.write("") # Spacer
        st.write("") 
        if st.button("Confirmar e Iniciar 🚀", type="primary"):
            st.session_state.char_data["name"] = confirm_name
            st.session_state.game_started = True
            st.rerun()

# AUTO-START LOGIC (Only if game_started)
if st.session_state.game_started and not st.session_state.history:
    with st.spinner("Mestre preparando o início da aventura..."):
        async def start_intro():
            handler = st.session_state.workflow
            char_name = st.session_state.char_data.get("name", "Aventureiro")
            result = await handler.run(action="INTRO_START", session_id="session_1", char_name=char_name)
            return result
        
        intro_text = asyncio.run(start_intro())
        st.session_state.history.append({"role": "assistant", "content": intro_text})
        st.rerun()

# GAME AREA (Only if game_started)
if st.session_state.game_started:
    # Chat Area
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Handling
    if st.session_state.pending_roll:
        # ROLL MODE - ACTION AREA
        req = st.session_state.pending_roll
        
        # Show the GM's request in the chat history (if not already there)
        # Note: We added it to history in the previous turn, but let's ensure the user sees the context
        # Actually, the previous turn added the "message_to_player" to history? 
        # No, the previous turn returned the RollRequestEvent.
        # So we should display the GM's message as the last message in history if it's not there.
        # But wait, the loop above displays history.
        # Let's check if we should add the GM message to history NOW so it persists.
        # The current code adds it AFTER the roll. That's fine, but maybe we want to see it now?
        # Let's keep the current flow: GM message is part of the "Action Request" UI, then added to history after resolution.
        
        with st.container(border=True):
            st.markdown(f"### 🎲 Desafio: {req.skill}")
            st.markdown(req.message_to_player)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Atributo:** {req.attr} | **Dificuldade (CD):** {req.dc}")
            
            with col2:
                if st.button(f"Rolar 🎲 (CD {req.dc})", type="primary", use_container_width=True):
                    roll = random.randint(1, 20)
                    mod = 0
                    if st.session_state.char_data:
                        attr_val = st.session_state.char_data["attributes"].get(req.attr, 10)
                        mod = (attr_val - 10) // 2
                    
                    total = roll + mod
                    
                    # Show result immediately
                    st.toast(f"Rolou {roll} + {mod} = {total}", icon="🎲")
                    
                    async def resume_roll():
                        handler = st.session_state.workflow
                        char_name = st.session_state.char_data.get("name", "Aventureiro")
                        # Resume with specific event
                        evt = NarrativeEvent(
                            input_text=req.action_text,
                            context_type="roll_result",
                            roll_details={
                                "roll_total": total,
                                "dc": req.dc,
                                "skill": req.skill
                            },
                            session_id="session_1",
                            char_name=char_name
                        )
                        res = await handler.run(item=evt)
                        return res

                    with st.spinner("Narrando consequência..."):
                        result = asyncio.run(resume_roll())
                        # Commit to history
                        st.session_state.history.append({"role": "assistant", "content": req.message_to_player}) 
                        st.session_state.history.append({"role": "assistant", "content": f"🎲 **Rolagem de {req.skill}:** {roll} + {mod} = **{total}** vs CD {req.dc}"})
                        st.session_state.history.append({"role": "assistant", "content": result})
                        st.session_state.pending_roll = None
                        st.rerun()

    else:
        # NORMAL CHAT MODE
        
        # Quick Actions
        col_q1, col_q2 = st.columns([1, 5])
        with col_q1:
            if st.button("🎲 Rolar D20", help="Rola um dado manualmente e envia para o Mestre"):
                roll = random.randint(1, 20)
                st.toast(f"Você rolou: {roll}", icon="🎲")
                
                # Send as action
                prompt = f"[SISTEMA] O jogador rolou um dado manualmente e tirou: {roll}."
                st.session_state.history.append({"role": "user", "content": f"🎲 *Rolo um D20...* **{roll}**"})
                
                with st.spinner("Mestre analisando rolagem..."):
                    async def run_manual_roll():
                        handler = st.session_state.workflow
                        char_name = st.session_state.char_data.get("name", "Aventureiro") if st.session_state.char_data else "Aventureiro"
                        result = await handler.run(action=prompt, session_id="session_1", char_name=char_name)
                        return result

                    output = asyncio.run(run_manual_roll())
                    
                    if isinstance(output, RollRequestEvent):
                        st.session_state.pending_roll = output
                        st.rerun()
                    else:
                        st.session_state.history.append({"role": "assistant", "content": output})
                        st.rerun()

        prompt = st.chat_input("Sua ação...")
        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.spinner("Mestre pensando..."):
                async def run_game():
                    handler = st.session_state.workflow
                    char_name = st.session_state.char_data.get("name", "Aventureiro") if st.session_state.char_data else "Aventureiro"
                    # Start workflow with StartEvent (kwargs passed to run)
                    result = await handler.run(action=prompt, session_id="session_1", char_name=char_name)
                    return result

                output = asyncio.run(run_game())
                
                if isinstance(output, RollRequestEvent):
                    st.session_state.pending_roll = output
                    st.rerun()
                else:
                    st.session_state.history.append({"role": "assistant", "content": output})
                    st.rerun()
