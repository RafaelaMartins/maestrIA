import os
import json
import re
import asyncio
import edge_tts
from pypdf import PdfReader
from llm import get_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FICHAS_DIR = os.path.join(BASE_DIR, "fichas")
os.makedirs(FICHAS_DIR, exist_ok=True)

class CharacterParser:
    @staticmethod
    def parse_pdf(pdf_path: str) -> dict:
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            return {"name": "Desconhecido", "attributes": {}, "error": str(e)}

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
            llm = get_llm()
            response = llm.invoke(prompt)
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except Exception as e:
            print(f"Erro no LLM Parse: {e}")

        return {"name": "Aventureiro", "attributes": {"FOR":10, "DES":10, "CON":10, "INT":10, "SAB":10, "CAR":10}}

VALID_VOICES = {
    "francisca": "pt-BR-FranciscaNeural",
    "thalita": "pt-BR-ThalitaNeural",
    "duarte": "pt-PT-DuarteNeural",
    "raquel": "pt-PT-RaquelNeural",
    "antonio": "pt-BR-AntonioNeural"
}

def get_valid_voice(voice_str: str, default: str) -> str:
    if not voice_str: return default
    v_lower = voice_str.lower()
    for k, v in VALID_VOICES.items():
        if k in v_lower:
            return v
    return default

async def _generate_audio_async(text: str, output_file: str, default_npc_voice: str = None, known_voices: dict = None):
    # known_voices: { "Nome": "Voz" }
    narrator_voice = "pt-BR-AntonioNeural"
    
    # Limpa markdown e emojis
    clean_text = re.sub(r'🎲.*?\n', '', text) 
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'_(.*?)_', r'\1', clean_text)
    
    # Divide em partes de narração e falas
    parts = re.split(r'("[^"]*")', clean_text)
    
    temp_files = []
    last_narration = ""
    last_voice_tag = None
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        voice = narrator_voice
        speech_text = part
        
        if part.startswith('"'):
            # É fala - NUNCA PODE SER ANTONIO
            npc_voice_to_use = default_npc_voice
            
            # 1. Se tem tag dentro da própria fala
            voice_match = re.search(r'\[VOICE:(.*?)\]', part, flags=re.IGNORECASE)
            if voice_match:
                npc_voice_to_use = get_valid_voice(voice_match.group(1).strip(), npc_voice_to_use)
            # 2. Se a narração logo antes da fala deixou uma tag
            elif last_voice_tag:
                npc_voice_to_use = get_valid_voice(last_voice_tag, npc_voice_to_use)
            
            if known_voices and not npc_voice_to_use:
                for name, v in known_voices.items():
                    if name.lower() in last_narration.lower():
                        npc_voice_to_use = v
                        break
            
            # Heurística de gênero se não achou nome ou voice id específico
            if not npc_voice_to_use or npc_voice_to_use == narrator_voice:
                if any(ind in last_narration.lower() for ind in ["ela ", " a ", "uma ", "feiticeira", "guerreira", "mira", "lady", "mulher"]):
                    voice = "pt-BR-ThalitaNeural"
                elif any(ind in last_narration.lower() for ind in ["ele ", " o ", "um ", "barman", "guarda", "homem", "mestre", "ancião", "velho"]):
                    voice = "pt-PT-DuarteNeural"
                else:
                    voice = "pt-BR-ThalitaNeural" # Fallback feminino
            else:
                voice = get_valid_voice(npc_voice_to_use, "pt-BR-ThalitaNeural")
            
            speech_text = part.strip('"')
            speech_text = re.sub(r'\[VOICE:.*?\]', '', speech_text, flags=re.IGNORECASE).strip()
            last_voice_tag = None # Consome a tag
        else:
            # É narração - SEMPRE ANTONIO
            voice = narrator_voice
            last_narration = part[-60:]
            
            # Procura tag na narração para salvar para a próxima fala
            voice_match = re.findall(r'\[VOICE:(.*?)\]', part, flags=re.IGNORECASE)
            if voice_match:
                last_voice_tag = voice_match[-1].strip()
                
            speech_text = re.sub(r'\[VOICE:.*?\]', '', part, flags=re.IGNORECASE).strip()
            speech_text = re.sub(r'\[VOZ:.*?\]', '', speech_text, flags=re.IGNORECASE).strip()
            
        if not speech_text.strip():
            continue
            
        temp_file = f"temp_chunk_{i}.mp3"
        try:
            communicate = edge_tts.Communicate(speech_text, voice)
            await communicate.save(temp_file)
            temp_files.append(temp_file)
        except Exception as e:
            print(f"Erro no edge-tts chunk {i}: {e}")

    # Concatena
    with open(output_file, 'wb') as outfile:
        for f in temp_files:
            try:
                with open(f, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(f)
            except:
                pass

def generate_tts_audio(text, output_file, voice=None, known_voices=None) -> bool:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate_audio_async(text, output_file, voice, known_voices))
        loop.close()
        return True
    except Exception as e:
        print(f"Erro fatal no TTS: {e}")
        return False
