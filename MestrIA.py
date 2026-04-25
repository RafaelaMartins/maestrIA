# app.py / mestrIA.py — Mestre / Árbitro / Narrador — Arquivo único (Mistral:7b)
# Requer: streamlit, pypdf, tqdm, llama_index, ollama runtime disponível localmente
# Uso: streamlit run mestrIA.py

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
from llama_index.embeddings.fastembed import FastEmbedEmbedding
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
# CONFIG LLM (CPU ONLY) - Mistral chosen as requested
# ---------------------------
def configure_global_settings():
    Settings.llm = Ollama(
        model="mistral:7b",
        temperature=0.9,
        request_timeout=1200,
        llm_kwargs={"num_gpu_layers": 0}
    )

    # FIX CRÍTICO PARA O ERRO DO META TENSOR
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",              # força CPU
        embed_batch_size=8         # mais leve na RAM
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
# Helpers para rerun/estado
# ---------------------------
def safe_rerun():
    """
    Tenta chamar st.experimental_rerun() (presente em algumas versões do Streamlit).
    Se não existir (AttributeError), alterna uma flag em session_state para forçar rerun.
    """
    try:
        st.experimental_rerun()
    except AttributeError:
        st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)
    except Exception:
        # fallback silencioso (não queremos quebrar UI por causa do rerun)
        st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)

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
            # no UI blocking here — return partial
            print(f"Erro ao ler {file}: {e}")
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
# FICHA (PDF) PARSING — heurística reforçada
# ---------------------------
ATTRIBUTES = [
    "FOR", "STR", "FORÇA",
    "DES", "DEX", "DESTREZA",
    "CON", "CONSTITUIÇÃO", "CONSTITUICAO",
    "INT", "INTELIGÊNCIA", "INTELLIGENCIA",
    "SAB", "WIS", "SABEDORIA",
    "CAR", "CHA", "CARISMA"
]

def clean_candidate_name(s: str) -> str:
    if not s:
        return None
    s = s.strip()
    # remove common trailing slashes/spurious chars
    s = s.replace("/", " ").strip()
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9 \-']", " ", s)  # keep letters (incl. accents), numbers, dash, apostrophe
    s = re.sub(r"\s{2,}", " ", s)
    s = s.title()
    if len(s) < 2:
        return None
    return s

def parse_ficha_pdf_to_json(pdf_path):
    """
    Heurística reforçada para extrair:
    - nome (tentativa por campos de formulário, labels comuns, proximidade, fallback filename)
    - atributos (valores numéricos, incluindo negativos)
    """
    result = {"name": None, "attributes": {}}
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # 1) tentar extrair campos de formulário (se o PDF for preenchível)
        try:
            form = reader.get_form_text_fields()  # pypdf helper
            if isinstance(form, dict) and form:
                # heurística: procurar chaves que contenham 'nome' ou 'personagem'
                for k, v in form.items():
                    if v and isinstance(v, str):
                        key_l = k.lower()
                        if "nome" in key_l or "personagem" in key_l:
                            candidate = clean_candidate_name(v)
                            if candidate:
                                result["name"] = candidate
                                break
                # também procurar campos que pareçam atributos
                for k, v in form.items():
                    if not v:
                        continue
                    kk = k.upper()
                    for token in ATTRIBUTES:
                        if token in kk:
                            try:
                                val = int(re.sub(r"[^\-0-9]", "", v))
                                # normalize token
                                if "FOR" in token:
                                    result["attributes"]["FOR"] = val
                                elif token in ["DES","DEX","DESTREZA"]:
                                    result["attributes"]["DES"] = val
                                elif "CON" in token:
                                    result["attributes"]["CON"] = val
                                elif "INT" in token:
                                    result["attributes"]["INT"] = val
                                elif "SAB" in token or "WIS" in token:
                                    result["attributes"]["SAB"] = val
                                elif token in ["CAR","CHA","CARISMA"]:
                                    result["attributes"]["CAR"] = val
                            except:
                                pass
        except Exception:
            # reader may not have form fields
            pass

        # 2) concat text from pages
        for p in reader.pages:
            try:
                t = p.extract_text()
            except Exception:
                t = None
            if t:
                text += "\n" + t

        # normalize and uppercase for matching, but keep original lines for candidate extraction
        txt = text or ""
        txt_up = txt.upper()

        # 3) try label-based name extraction from text
        if not result["name"]:
            # patterns to try (Portuguese common labels)
            patterns = [
                r"NOME DO PERSONAGEM[:\s]*([A-Z0-9 \-\/'áéíóúãõâêôàèùç]{2,80})",
                r"NOME[:\s]*([A-Z0-9 \-\/'áéíóúãõâêôàèùç]{2,80})",
                r"PERSONAGEM[:\s]*([A-Z0-9 \-\/'áéíóúãõâêôàèùç]{2,80})",
                r"CHARACTER[:\s]*([A-Z0-9 \-\/'áéíóúãõâêôàèùç]{2,80})"
            ]
            for p in patterns:
                m = re.search(p, txt_up, flags=re.IGNORECASE)
                if m:
                    candidate = clean_candidate_name(m.group(1))
                    if candidate and candidate != "/":
                        result["name"] = candidate
                        break

        # 4) fallback: look for lines that look like a name (first page header lines)
        if not result["name"]:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            # pick first non-empty line that is not "FICHA", "PERSONAGEM", etc and is not too short
            for ln in lines[:20]:
                ln_clean = ln.strip()
                if len(ln_clean) < 3:
                    continue
                if re.search(r"\b(FICHA|PERSONAGEM|Atributos|ATRIBUTOS|RPG|FICHA T20)\b", ln_clean, flags=re.IGNORECASE):
                    continue
                candidate = clean_candidate_name(ln_clean)
                if candidate and not re.search(r"\d", candidate):  # avoid lines that are numeric-heavy
                    result["name"] = candidate
                    break

        # 5) another fallback: look for patterns like 'Nome - X' or 'Nome: X' in entire text
        if not result["name"]:
            m = re.search(r"Nome\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ \-']{2,60})", txt, flags=re.IGNORECASE)
            if m:
                candidate = clean_candidate_name(m.group(1))
                if candidate:
                    result["name"] = candidate

        # 6) attribute extraction: more robust regex (allow negative, allow + or - signs, etc)
        for token in ATTRIBUTES:
            # look for token near number; use both uppercase and common forms
            pattern = rf"({token})\s*[:\-]?\s*([+\-]?\d{{1,2}})"
            mm = re.search(pattern, txt_up, flags=re.IGNORECASE)
            if mm:
                try:
                    val = int(re.sub(r"[^\-0-9]", "", mm.group(2)))
                    if "FOR" in token:
                        result["attributes"]["FOR"] = val
                    elif token in ["DES","DEX","DESTREZA"]:
                        result["attributes"]["DES"] = val
                    elif "CON" in token:
                        result["attributes"]["CON"] = val
                    elif "INT" in token:
                        result["attributes"]["INT"] = val
                    elif "SAB" in token or "WIS" in token:
                        result["attributes"]["SAB"] = val
                    elif token in ["CAR","CHA","CARISMA"]:
                        result["attributes"]["CAR"] = val
                except:
                    pass

        # 7) final fallbacks: if name still None or equals '/', try filename
        if not result["name"] or result["name"].strip() in ["", "/"]:
            try:
                fname = os.path.basename(pdf_path)
                candidate = os.path.splitext(fname)[0]
                candidate = re.sub(r"[_\-]+", " ", candidate)
                candidate = clean_candidate_name(candidate)
                if candidate:
                    result["name"] = candidate
            except:
                pass

        # Normalize attribute defaults: ensure keys for all main attrs exist (avoid missing later)
        for k in ["FOR","DES","CON","INT","SAB","CAR"]:
            if k not in result["attributes"]:
                # default neutral 10? original system expected small numbers; use 10 as neutral
                # but to keep behavior consistent with earlier code, default to 10
                result["attributes"][k] = 10

        # final clean: if name like single char or just numbers, null it
        if result["name"]:
            if re.fullmatch(r"[\W_0-9]+", result["name"]):
                result["name"] = None

        return result
    except Exception as e:
        print(f"Erro parseando ficha: {e}")
        # ensure defaults
        for k in ["FOR","DES","CON","INT","SAB","CAR"]:
            if k not in result["attributes"]:
                result["attributes"][k] = 10
        return result

# ---------------------------
# Helpers para FICHA persistente / consistência do nome
# ---------------------------
def load_ficha(token):
    """
    Carrega o JSON de ficha se existir. Retorna dict ou None.
    """
    path = ficha_json_path(token)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_ficha(token, ficha):
    """
    Salva o JSON de ficha (substitui/reescreve).
    """
    path = ficha_json_path(token)
    # garante pasta
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False)

def ensure_ficha_name_consistency(token):
    """
    Garante que o nome da ficha JSON esteja sincronizado com st.session_state.player_name, se houver.
    Se houver player_name em session_state e ficha existir -> atualiza JSON (remove e recria).
    Retorna a ficha atualizada (ou None).
    """
    ficha = load_ficha(token)
    manual = st.session_state.get("player_name", "").strip()
    if ficha is None:
        return None
    if manual:
        # sobrescreve no dicionário e regrava conforme pedido (apagar e recriar)
        ficha["name"] = manual
        path = ficha_json_path(token)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        save_ficha(token, ficha)
    return ficha

def get_effective_name(token, ficha=None):
    """
    Retorna o nome efetivo que deve ser usado nas prompts e contextos:
    prioridade: st.session_state.player_name (manual, se preenchido) -> ficha['name'] -> 'Personagem Sem Nome'
    """
    manual = st.session_state.get("player_name", "").strip()
    if manual:
        return manual
    if ficha and isinstance(ficha, dict):
        return ficha.get("name") or "Personagem Sem Nome"
    ficha_loaded = load_ficha(token)
    if ficha_loaded:
        return ficha_loaded.get("name") or "Personagem Sem Nome"
    return "Personagem Sem Nome"

# ---------------------------
# REFEREE / ÁRBITRO (Agente que decide se rolar e qual atributo)
# ---------------------------
SKILL_MAP = {
    "ATIRAR": ("DES", "Pontaria"),
    "ARCO": ("DES", "Pontaria"),
    "ESCALAR": ("FOR", "Escalada"),
    "PULAR": ("FOR", "Acrobacia"),
    "SALTA": ("FOR", "Acrobacia"),
    "SALTO": ("FOR", "Acrobacia"),
    "CONVENCER": ("CAR", "Persuasão"),
    "PERSUADIR": ("CAR", "Persuasão"),
    "FEITIÇO": ("INT", "Magia"),
    "FEITICO": ("INT", "Magia"),
    "LANÇAR": ("INT", "Magia"),
    "LANCAR": ("INT", "Magia"),
    "FURTAR": ("DES", "Furtividade"),
    "ESCONDER": ("DES", "Furtividade"),
    "PERCEP": ("SAB", "Percepção"),
    "OBSER": ("SAB", "Percepção"),
    "ATACAR": ("FOR", "Ataque"),
    "BLOQUEAR": ("FOR", "Defesa"),
    "CURAR": ("SAB", "Medicina"),
    "ROUBAR": ("DES", "Furtividade"),
    "ARROMBAR": ("FOR", "Força"),
    "FUGIR": ("DES", "Acrobacia"),
    "INVADIR": ("FOR", "Força"),
    "INTIMIDAR": ("CAR", "Intimidação"),
}

def classify_dc_by_description(text):
    t = text.upper()
    dc = 12
    if any(x in t for x in ["FÁCIL","FACIL","SIMPLES"]):
        dc = 8
    if any(x in t for x in ["DIFÍCIL","DIFICIL","DIF"]):
        dc = 16
    if any(x in t for x in ["MUITO","PERIGOS","PERIGO","ARRISCADO"]):
        dc = 15
    if any(x in t for x in ["IMPOSSÍVEL","IMPOSSIVEL"]):
        dc = 20
    return dc

def referee_agent_decide(action_text):
    a = action_text.upper()
    # keywords
    for kw, (attr, skill) in SKILL_MAP.items():
        if kw in a:
            dc = classify_dc_by_description(action_text)
            reason = f"Ação detectada: {skill}."
            return {"requires_roll": True, "attr": attr, "skill": skill, "dc": dc, "reason": reason}
    # explicit test keywords
    if any(x in a for x in ["TENTAR", "TENTO", "TENTATIVA", "ROLAR", "TESTE", "DESAFIO"]):
        return {"requires_roll": True, "attr": "DES", "skill": "Teste Genérico", "dc": 12, "reason": "Ação explícita de teste."}
    # social
    if any(x in a for x in ["CONVENCER", "PERSUADIR", "INTIMIDAR", "BLEFAR", "NEGOCIAR"]):
        return {"requires_roll": True, "attr": "CAR", "skill": "Persuasão", "dc": 12, "reason": "Ação social."}
    # trivial actions
    if any(x in a for x in ["ANDAR", "OLHAR", "FALAR", "RESPIRAR", "ESCUTAR", "OBSERVAR"]):
        return {"requires_roll": False, "attr": None, "skill": None, "dc": None, "reason": "Ação trivial, sem teste."}
    # risky defaults
    risky_words = ["ATIRAR", "PULAR", "ESCALAR", "SALTAR", "FUGIR", "INVADIR", "ROUBAR", "ATAQUE", "ATACAR"]
    if any(x in a for x in risky_words):
        return {"requires_roll": True, "attr": "FOR", "skill": "Teste Arriscado", "dc": 13, "reason": "Ação arriscada detectada."}
    return {"requires_roll": False, "attr": None, "skill": None, "dc": None, "reason": "Ação comum, sem teste necessário."}

def attribute_mod(score):
    try:
        return (int(score) - 10) // 2
    except:
        return 0

def resolve_check_with_roll(ficha, attr, dc, roll_value=None):
    if roll_value is None:
        roll = random.randint(1, 20)
    else:
        try:
            roll = int(roll_value)
            roll = max(1, min(20, roll))
        except:
            roll = random.randint(1, 20)

    base = ficha.get("attributes", {}).get(attr, None) if ficha else None
    mod = attribute_mod(base) if base is not None else 0
    total = roll + mod
    diff = total - dc
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
    return {"roll": roll, "mod": mod, "total": total, "dc": dc, "degree": degree, "diff": diff, "message": msg}

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
# NARRADOR (RAG/LLM) — produz narração baseada no resultado
# ---------------------------
def narrator_generate_description(token, ficha, action_text, check_result):
    """
    Gera narrativa cinematográfica robusta tanto para ações narrativas quanto para resolvidas com teste.
    """
    index = load_or_create_index() if os.path.exists(INDEX_DIR) else None
    query_engine = index.as_query_engine(similarity_top_k=3) if index else None

    history = get_history_window(token)
    player_name = ficha.get("name", "O personagem")

    degree = check_result.get("degree", "auto")
    diff = check_result.get("diff", 0)
    message = check_result.get("message", "")

    # -----------------------------------------
    # MODO 1 — AÇÃO SEM TESTE
    # -----------------------------------------
    if degree == "auto":
        prompt = f"""
Você é o MESTRE NARRADOR.
Produza apenas a NARRAÇÃO CINEMATOGRÁFICA final, sem explicar regras.

🎭 CENA:
- Continue a partir do histórico abaixo.
- Reaja diretamente à ação: "{action_text}" tomada por {player_name}.
- Descreva como o ambiente reage (luzes, poeira, ruído, vento, magia, ecos).
- Mostre ao menos 1 reação clara de personagens presentes.
- Avance a história com um novo detalhe significativo (uma ameaça, pista, mudança, revelação).
- Mantenha o tom sombrio, tenso, físico, concreto — nada vago.
- Termine apenas com: "O que você faz agora?"

Histórico recente:
{history}

Agora gere somente a narrativa final.
"""

    # -----------------------------------------
    # MODO 2 — AÇÃO COM TESTE
    # -----------------------------------------
    else:
        prompt = f"""
Você é o MESTRE NARRADOR.
Traduza o resultado do teste em CONSEQUÊNCIA NARRATIVA pura.

🎭 INSTRUÇÕES (não dizer ao jogador):
- {player_name} tomou esta ação: "{action_text}"
- Resultado do teste (não mencionar termos técnicos): {message}

Regras de narrativa:
- Falha → detalhe negativo imediato (queda, alerta inimigo, barulho, dor, oportunidade perdida)
- Sucesso → avanço claro, resultado tangível
- Sucesso no limite → sucesso com custo (ferimento leve, barulho, complicação)
- Sucesso alto → execução primorosa com vantagem ou impacto visual forte
- Crítico → transformação extrema da cena (muito positivo ou muito ruim)
- Reaja sempre com ambiente + personagens
- Avance a cena em direção a algo maior
- Finalize com: "O que você faz agora?"

Histórico recente:
{history}

Agora gere SOMENTE a narrativa final.
"""

    try:
        if query_engine:
            narration = str(query_engine.query(prompt)).strip()
        else:
            narration = Settings.llm.call(prompt).strip()

        # sanitize
        narration = re.sub(r"\b(NPC|PLAYER|JOGADOR)\b", player_name, narration, flags=re.IGNORECASE)

        if not narration.lower().strip().endswith("o que você faz agora?"):
            narration += "\n\nO que você faz agora?"

        return narration

    except Exception as e:
        print("Erro:",str(e))
        return f"{player_name} age e o mundo responde — algo muda de forma intensa.\n\nO que você faz agora?"



# ---------------------------
# ORCHESTRATOR & AGENTS (Refactoring)
# ---------------------------

class RefereeAgent:
    """
    Agente Árbitro: Decide se uma ação precisa de teste e qual atributo usar.
    """
    def decide(self, action_text):
        return referee_agent_decide(action_text)

class ActionResolver:
    """
    Agente de Resolução: Processa a mecânica do teste (rolagem + modificadores).
    """
    def resolve(self, ficha, attr, dc, roll_value=None):
        return resolve_check_with_roll(ficha, attr, dc, roll_value)

class NarratorAgent:
    """
    Agente Narrador: Gera a descrição da cena baseada no resultado ou ação direta.
    """
    def narrate(self, token, ficha, action_text, check_result):
        return narrator_generate_description(token, ficha, action_text, check_result)

class GameOrchestrator:
    """
    Orquestrador: Gerencia o fluxo entre os agentes (Árbitro -> Resolução -> Narrador).
    """
    def __init__(self):
        self.referee = RefereeAgent()
        self.resolver = ActionResolver()
        self.narrator = NarratorAgent()

    def process_action(self, token, action_text):
        """
        Fluxo principal de uma ação do jogador.
        1. Verifica se campanha está fechada.
        2. Carrega ficha.
        3. Chama Árbitro para decidir se precisa de teste.
        4. Se precisar de teste, configura estado e retorna solicitação.
        5. Se não precisar, chama Narrador diretamente.
        """
        meta = increment_turn(token)
        if meta and meta.get("closed", False):
            return "Esta campanha já foi encerrada."

        ficha = load_ficha(token) or {}
        if ficha:
            ficha = ensure_ficha_name_consistency(token) or ficha

        # 1. Árbitro decide
        decision = self.referee.decide(action_text)

        # Grava decisão
        effective_name = get_effective_name(token, ficha)
        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"Jogador ({effective_name}): {action_text}\nREFEREE_DECISION: {decision}\n")

        # 2. Se precisa de rolagem
        if decision["requires_roll"]:
            return self._setup_roll_state(token, decision, action_text)
        
        # 3. Se não precisa de rolagem (Narrativa direta)
        check_result = {
            "roll": None, "mod": 0, "total": None, "dc": None,
            "degree": "auto", "diff": 0,
            "message": "Ação resolvida narrativamente."
        }
        narration = self.narrator.narrate(token, ficha, action_text, check_result)
        
        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"Mestre: {narration}\n")
        
        return narration

    def process_roll(self, token, roll_override=None):
        """
        Fluxo de resolução de teste.
        1. Recupera estado do teste.
        2. Chama ActionResolver para calcular resultado.
        3. Chama Narrador para descrever consequência.
        4. Limpa estado.
        """
        meta_session = st.session_state.get("roll_meta")
        meta_file = load_meta(token) or {}
        meta_saved = meta_file.get("roll_state")
        meta = meta_session or meta_saved
        
        if not meta:
            return "Nenhum teste pendente."

        attr = meta["attr"]
        dc = meta["dc"]
        action_text = meta["action"]
        ficha = load_ficha(token) or {}

        # 1. Resolver Mecânica
        res = self.resolver.resolve(ficha, attr, dc, roll_value=roll_override)

        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"RESULTADO_TESTE: {res['message']}\n")

        # 2. Narrar Consequência
        narration = self.narrator.narrate(token, ficha, action_text, res)

        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"RESULTADO_TESTE_DETALHADO: {res['message']}\nNarrador: {narration}\n")

        # 3. Limpar Estado
        self._clear_roll_state(token, meta_file)

        return {"narration": narration, "roll_result": res}

    def _setup_roll_state(self, token, decision, action_text):
        attr = decision["attr"]
        skill = decision["skill"]
        dc = decision["dc"]
        reason = decision["reason"]

        roll_state = {
            "awaiting_roll": True,
            "attr": attr,
            "skill": skill,
            "dc": dc,
            "reason": reason,
            "action": action_text
        }
        meta = load_meta(token) or {}
        meta["roll_state"] = roll_state
        save_meta(token, meta)

        st.session_state.awaiting_roll = True
        st.session_state.roll_meta = {
            "attr": attr,
            "skill": skill,
            "dc": dc,
            "action": action_text,
            "player_token": token
        }

        msg = (
            f"Mestre: {reason}\n\n"
            f"→ **Teste necessário:** {skill} ({attr})\n"
            f"→ **Dificuldade (DC):** {dc}\n\n"
            "Role 1d20 (use o botão de rolagem automática abaixo) ou envie valor manual."
        )
        return msg

    def _clear_roll_state(self, token, meta_file):
        st.session_state.awaiting_roll = False
        st.session_state.roll_meta = None
        if "roll_state" in meta_file:
            meta_file.pop("roll_state")
            save_meta(token, meta_file)


# ---------------------------
# START STORY / NEXT ACTION (integra árbitro e narrador)
# ---------------------------
def start_story(token):
    ficha_json = ficha_json_path(token)
    if not os.path.exists(ficha_json):
        return "⚠️ Nenhuma ficha foi enviada ainda. Envie sua ficha em PDF antes de começar."

    with open(ficha_json, "r", encoding="utf-8") as f:
        ficha = json.load(f)

    player_name = ficha.get("name", "Personagem Sem Nome")
    create_new_session(token)

    prompt = f"""
    Você é o MESTRE NARRADOR de uma história de fantasia sombria.
    Produza SOMENTE a narrativa cinematográfica inicial, em português.

     Regras internas (NÃO citar na resposta final):
    - NÃO explique regras nem sistemas.
    - NÃO mencione que recebeu instruções.
    - NÃO use as palavras jogador, player ou NPC.
    - SEMPRE use o nome do personagem: {player_name}.
    - TOM: sombrio, cinético, sensorial, com tensão crescente.
    - A cena inicial deve introduzir:
    • o mundo ao redor (economia, clima, rumores, tensões, ruínas, forças sobrenaturais)
    • o local exato onde {player_name} está (detalhes físicos e atmosféricos)
    • eventos imediatos acontecendo ao redor (movimentos, sons, luzes, ameaças)
    • DOIS personagens relevantes com motivações claras
    • um problema ou risco imediato que obriga {player_name} a escolher
    • sensações físicas e emocionais de {player_name} (visão, audição, tato, cheiro, intuição)

    - Coloque {player_name} DENTRO da cena, percebendo e reagindo ao ambiente.
    - Conecte todos os elementos em uma narrativa fluida e cinematográfica.
    - Use frases curtas, impactantes e cheias de tensão.
    - Evite finais vagos ou reticências.
    - Finalize EXCLUSIVAMENTE com: "O que você faz?"
        "1) Crie agora uma introdução épica, sensorial e imersiva:\n"
        "- Apresente o mundo com sons, cheiros, clima e atmosfera.\n"
        "- Apresente o local exato onde o personagem inicia.\n"
        "- Gere um acontecimento inicial \n"
        "- Introduza pelo menos 1 NPC relevante, bem caracterizado.\n"


        "2) REGRAS SOBRE O QUE PODE OU NÃO PERGUNTAR:\n"
        "- Você NÃO PODE pedir nomes ou definições sobre NPCs.\n"
        "- Você NÃO PODE transferir controle de NPCs ao jogador/personagem {player_name}.\n"
        "- Você NÃO PODE pedir ao jogador para criar elementos do mundo.\n"
        "- NUNCA MENCIONA AS REGRAS\n"
        "3) SEMPRE Finalize o trecho incluindo o personagem {player_name} dentro do ambiente e PERGUNTANDO alguma ação\n"
        "- Exemplo: Você jogador acaba de acorda em meio a floresta perto de você há uma caverna mas você também escuta barulho de carruagem por perto indicando uma estrada por perto, o que você deseja fazer?\n"
        "- Exemplo: Agora você fará parte deste mundo mágico para isso jogador, me informe qual o nome do seu personagem?\n"
        
    """


    try:
        if os.path.exists(INDEX_DIR):
            index = load_or_create_index()
            response = index.as_query_engine(similarity_top_k=3).query(prompt)
            intro = str(response).strip()
        else:
            intro = Settings.llm.call(prompt).strip()

    except Exception as e:
        print("Erro:",str(e))
        intro = (
            f"{player_name} desperta entre ruínas antigas. O ar vibra com energia estática, "
            "duas figuras o observam e algo pulsa no chão à frente. O que você faz?"
        )

    # Sanitização
    intro = re.sub(r"\b(NPC|Jogador|Player)\b", player_name, intro, flags=re.IGNORECASE)
    if not intro.endswith("?"):
        intro += "\n\nO que você faz?"

    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"Mestre: {intro}\n")

    increment_turn(token)
    return intro


def next_action(token, action):
    orchestrator = GameOrchestrator()
    return orchestrator.process_action(token, action)

def apply_roll_and_inform_master(token, roll_override=None):
    orchestrator = GameOrchestrator()
    return orchestrator.process_roll(token, roll_override)

# ---------------------------
# STREAMLIT UI
# ---------------------------
try:
    st.set_page_config(page_title="Mestre-IA RPG", page_icon="🎲", layout="wide")
except Exception:
    pass

st.title("🎲 Mestre-IA RPG — Arquitetura A (Mestre / Árbitro / Narrador) — Híbrido Cinemático")

# sessão id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "awaiting_roll" not in st.session_state:
    st.session_state.awaiting_roll = False
if "roll_meta" not in st.session_state:
    st.session_state.roll_meta = None

# flag for safe rerun if necessary
if "_rerun_flag" not in st.session_state:
    st.session_state._rerun_flag = False

st.sidebar.markdown(f"**Sessão:** `{st.session_state.session_id}`")

token = st.session_state.session_id
ficha_path = ficha_json_path(token)

# ---------------------------
# INPUT DO NOME (persistente, aparece assim que a página carrega)
# ---------------------------
st.sidebar.subheader("📝 Nome do Personagem (persistente)")
# inicializa o player_name na session_state a partir de ficheiro se existir
if "player_name" not in st.session_state:
    # tenta carregar do ficheiro caso exista
    existing = load_ficha(token)
    if existing and existing.get("name"):
        st.session_state.player_name = existing.get("name")
    else:
        st.session_state.player_name = ""

player_name_input = st.sidebar.text_input("Digite o nome do personagem (persistente):", value=st.session_state.get("player_name", ""))

if st.sidebar.button("💾 Salvar Nome (Persistente)"):
    # salva no estado de sessão
    st.session_state.player_name = player_name_input.strip()
    # se ficha existir -> atualiza JSON (apaga e salva novo) e registra no contexto
    if os.path.exists(ficha_path):
        ficha_existing = load_ficha(token) or {"name": st.session_state.player_name, "attributes": {}}
        ficha_existing["name"] = st.session_state.player_name
        # remove e recria conforme pedido
        try:
            os.remove(ficha_path)
        except Exception:
            pass
        save_ficha(token, ficha_existing)
        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"NOME_ATUALIZADO: {st.session_state.player_name}\n")
    st.sidebar.success(f"Nome persistente salvo: **{st.session_state.player_name}**")
    safe_rerun()

# ---------------------------
# UPLOAD DE FICHA (PDF)
# ---------------------------
st.sidebar.subheader("📄 Upload de Ficha (PDF)")
uploaded = st.sidebar.file_uploader(
    "Envie a ficha do personagem (PDF) — obrigatória antes de iniciar a história",
    type=["pdf"]
)

if uploaded is not None:
    pdf_path = ficha_pdf_path(token)

    # salva o PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    # extrai dados do PDF → JSON (nome original + atributos)
    ficha = parse_ficha_pdf_to_json(pdf_path)

    # se houver player_name manual preenchido, sobrescreve o nome extraído
    manual = st.session_state.get("player_name", "").strip()
    if manual:
        ficha["name"] = manual

    # salva JSON inicial
    save_ficha(token, ficha)

    name_display = ficha.get("name") or "Personagem Sem Nome"
    st.sidebar.success(
        f"Ficha salva. Personagem: {name_display}. Atributos: {ficha.get('attributes')}"
    )

    # registra no contexto
    with open(get_context_file(token), "a", encoding="utf-8") as f:
        f.write(f"FICHA_CARREGADA: {name_display} — {ficha.get('attributes')}\n")

    # atualiza o campo persistente se ficha trouxe nome e não havia manual
    if not st.session_state.get("player_name") and ficha.get("name"):
        st.session_state.player_name = ficha.get("name")

# ---------------------------
# BLOCO DE ATUALIZAÇÃO DE NOME (SE JSON EXISTIR) - alternativa ao input persistente
# ---------------------------
st.subheader("📝 Atualizar Nome da Ficha (se houver JSON)")
if os.path.exists(ficha_path):
    ficha_loaded = load_ficha(token) or {}
    nome_atual = ficha_loaded.get("name", "")
    nome_digitado = st.text_input("Digite o nome do personagem (atualiza o JSON):", value=nome_atual, key="update_name_input")

    if st.button("💾 Atualizar Nome (JSON)"):
        new_name = nome_digitado.strip()
        ficha_loaded["name"] = new_name
        # remove o arquivo antigo e recria (conforme pedido)
        try:
            os.remove(ficha_path)
        except Exception:
            pass
        save_ficha(token, ficha_loaded)

        # também atualiza o nome persistente na sessão
        st.session_state.player_name = new_name

        # registra no contexto para manter coerência narrativa
        with open(get_context_file(token), "a", encoding="utf-8") as f:
            f.write(f"NOME_ATUALIZADO: {new_name}\n")
            f.write(f"NOME: {new_name}\n")
            f.write(f"Name: {new_name}\n")
            f.write(f"name: {new_name}\n")
            f.write(f"Personagem: {new_name}\n")

        st.success(f"Nome atualizado para: **{new_name}**")
        safe_rerun()

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
        try:
            safe_rerun()
        except Exception:
            st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)

# ---------------------------
# EXIBIR HISTÓRICO
# ---------------------------
st.markdown("---")
for msg in st.session_state.history:
    if msg["autor"] == "user":
        st.markdown(f"**Você:** {msg['texto']}")
    else:
        cols = st.columns([0.5, 9.5])
        with cols[0]:
            st.write("🧙")
        with cols[1]:
            st.markdown(f"**Mestre:** {msg['texto']}")
st.markdown("---")

# ---------------------------
# SE HOUVER ROLL PENDENTE -> BOTÃO AUTOMÁTICO (1-20) + MANUAL
# ---------------------------
if st.session_state.awaiting_roll:
    st.info("O Mestre solicitou um teste. Use rolagem automática ou envie um valor manual.")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        if st.button("🎲 Rolar D20 (Automático)"):
            with st.spinner("Jogando o dado..."):
                time.sleep(0.6)
                roll_val = random.randint(1, 20)
                out = apply_roll_and_inform_master(st.session_state.session_id, roll_override=roll_val)
                if isinstance(out, dict):
                    r = out["roll_result"]
                    st.session_state.history.append({"autor": "user", "texto": f"(Rolagem automática) 1d20 = {r['roll']} — {r['message']}"} )
                    st.session_state.history.append({"autor": "mestre", "texto": out["narration"]})
                else:
                    st.session_state.history.append({"autor": "mestre", "texto": str(out)})
            try:
                safe_rerun()
            except Exception:
                st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)
    with col_b:
        manual_roll = st.number_input("Enviar valor do D20 (1-20):", min_value=1, max_value=20, value=10, step=1)
        if st.button("Enviar roll manual"):
            with st.spinner("Enviando roll..."):
                time.sleep(0.3)
                out = apply_roll_and_inform_master(st.session_state.session_id, roll_override=int(manual_roll))
                if isinstance(out, dict):
                    r = out["roll_result"]
                    st.session_state.history.append({"autor": "user", "texto": f"(Rolagem manual) 1d20 = {r['roll']} — {r['message']}"} )
                    st.session_state.history.append({"autor": "mestre", "texto": out["narration"]})
                else:
                    st.session_state.history.append({"autor": "mestre", "texto": str(out)})
            try:
                safe_rerun()
            except Exception:
                st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)

# ---------------------------
# INPUT DO JOGADOR (AÇÕES)
# ---------------------------
st.subheader("⚔ Sua Ação")
user_input = st.text_input("Digite sua ação (ex: 'Atacar com arco', 'Tentar convencer o guarda'):")
if st.button("📩 Enviar ação"):
    if user_input.strip():
        # garante que ficha existe antes de aceitar ações
        token = st.session_state.session_id
        if not os.path.exists(ficha_json_path(token)):
            st.warning("Envie a Ficha (PDF) antes de enviar ações.")
        else:
            # registra ação do jogador
            st.session_state.history.append({"autor": "user", "texto": user_input})
            resposta = next_action(st.session_state.session_id, user_input)
            st.session_state.history.append({"autor": "mestre", "texto": resposta})
            try:
                safe_rerun()
            except Exception:
                st.session_state._rerun_flag = not st.session_state.get("_rerun_flag", False)

# ---------------------------
# DEBUG / INFO (expansível)
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
            st.text("\n".join(f.readlines()[-60:]))

st.caption("Qualquer coisa, vai nos avisando.")
