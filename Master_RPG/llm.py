from langchain_ollama import ChatOllama

# Shared Qwen 2.5 14B instance for all agents
def get_llm():
    return ChatOllama(
        model="qwen2.5:14b",
        temperature=0.7, # A bit of creativity for RPG
        num_gpu=99 # GPU acceleration
    )
