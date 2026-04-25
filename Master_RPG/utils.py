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

async def _generate_audio_async(text: str, output_file: str, voice: str):
    # Limpa markdown e emojis para a voz ficar mais natural
    clean_text = re.sub(r'🎲.*?\n', '', text) # Remove a linha do dado
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text) # Remove negrito
    clean_text = re.sub(r'_(.*?)_', r'\1', clean_text) # Remove italico
    
    communicate = edge_tts.Communicate(clean_text, voice)
    await communicate.save(output_file)
def generate_tts_audio(text: str, output_file: str, voice: str = "pt-BR-ThalitaMultilingualNeural"):
    '''
        voice: "pt-BR-AntonioNeural"
        voice: "pt-BR-FranciscaNeural"
        voice: "pt-BR-ThalitaMultilingualNeural"
    '''
    # Roda o gerador assíncrono do edge-tts
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    loop.run_until_complete(_generate_audio_async(text, output_file, voice))
