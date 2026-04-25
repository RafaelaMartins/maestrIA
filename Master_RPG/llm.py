from langchain_ollama import ChatOllama

# Shared Mistral 7B instance for all agents
def get_llm():
    return ChatOllama(
        model="mistral:7b",
        temperature=0.7, # A bit of creativity for RPG
        num_gpu=99 # GPU acceleration
    )
