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

async def _generate_audio_async(text: str, output_file: str, npc_voice: str = None):
    narrator_voice = "pt-BR-AntonioNeural"
    female_voice = "pt-BR-ThalitaMultilingualNeural"
    male_voice = "pt-PT-DuarteNeural"
    
    # Se não for passado um voice id específico, tentamos inferir
    default_npc_voice = npc_voice if npc_voice else female_voice
    
    # Limpa markdown e emojis
    clean_text = re.sub(r'🎲.*?\n', '', text) 
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'_(.*?)_', r'\1', clean_text)
    
    # Fatiar por aspas ("...") ou tags [VOICE:...]
    # Regex flexível para capturar [VOICE:XYZ] "Texto" (com ou sem espaço)
    parts = re.split(r'(\[VOICE:.*?\]\s*".*?"|".*?")', clean_text)
    
    temp_files = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        voice = narrator_voice
        speech_text = part
        
        # Detecta Tag de voz: [VOICE:Thalita] "..."
        tag_match = re.match(r'\[VOICE:(.*?)\]\s*"(.*?)"', part)
        if tag_match:
            voice_key = tag_match.group(1).strip()
            speech_text = tag_match.group(2).strip()
            # Mapeamento
            if "Thalita" in voice_key: voice = female_voice
            elif "Francisca" in voice_key: voice = "pt-BR-FranciscaNeural"
            elif "Duarte" in voice_key: voice = male_voice
            elif "Antonio" in voice_key: voice = narrator_voice
        elif part.startswith('"') and part.endswith('"'):
            # Aspas simples sem tag
            voice = default_npc_voice
            speech_text = part.strip('"')
        else:
            # É narração - LIMPA qualquer tag residual que o LLM possa ter deixado fora do lugar
            speech_text = re.sub(r'\[VOICE:.*?\]', '', part).strip()
            
        if not speech_text.strip():
            continue
            
        temp_file = f"temp_chunk_{i}.mp3"
        communicate = edge_tts.Communicate(speech_text, voice)
        await communicate.save(temp_file)
        temp_files.append(temp_file)
        
    # Concatena os MP3s
    with open(output_file, 'wb') as outfile:
        for f in temp_files:
            try:
                with open(f, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(f)
            except Exception as e:
                print(f"Erro concatenando audio: {e}")

def generate_tts_audio(text: str, output_file: str, npc_voice: str = None):
    # Roda o gerador assíncrono do edge-tts
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    loop.run_until_complete(_generate_audio_async(text, output_file, npc_voice))
