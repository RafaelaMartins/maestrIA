# 🐉 MestrIA v2 - AI RPG Master

MestrIA v2 é um Mestre de RPG impulsionado por Inteligência Artificial, projetado para narrar aventuras dinâmicas, arbitrar regras e gerenciar fichas de personagens usando **LlamaIndex Workflows** e **Ollama**.

## 🚀 Funcionalidades

*   **Narrativa Imersiva**: Um Agente Narrador que cria descrições ricas e reage às suas ações.
*   **Árbitro de Regras**: Um Agente Árbitro que analisa suas ações e decide quando um teste de dados é necessário.
*   **Sistema de Rolagem**:
    *   **Automático**: O Mestre pede um teste, você clica e o resultado é integrado à história.
    *   **Manual**: Um botão "Rolar D20" para você testar sua sorte a qualquer momento.
*   **RAG (Retrieval-Augmented Generation)**: O sistema lê PDFs da pasta `DB/` para aprender regras e lore do seu mundo.
*   **Leitura de Ficha**: Upload de PDF da ficha do personagem, com extração automática de atributos via IA.
*   **Memória de Contexto**: O Narrador lembra das suas últimas ações para manter a coerência.

## 🛠️ Pré-requisitos

*   **Python 3.10+**
*   **Ollama** instalado e rodando (com o modelo `mistral:7b` baixado).
    *   Para baixar o modelo: `ollama pull mistral:7b`

## 📦 Instalação

1.  Clone ou baixe este repositório.
2.  Instale as dependências:

```bash
pip install streamlit llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pypdf
```

## 🎮 Como Jogar

1.  Certifique-se que o Ollama está rodando.
2.  Execute a aplicação:

```bash
streamlit run MestrIA_v2.py
```

3.  **No Navegador**:
    *   Faça upload do PDF da sua ficha na barra lateral.
    *   Confirme ou edite o nome do seu personagem.
    *   Clique em "Confirmar e Iniciar".
    *   Aproveite a aventura! Digite suas ações no chat ou use o botão de rolagem.

## 📂 Estrutura de Pastas

*   `MestrIA_v2.py`: Código principal.
*   `DB/`: Coloque seus PDFs de regras e lore aqui.
*   `contexts_v2/`: Diretório de armazenamento de índices e fichas.

## ⚙️ Configuração Técnica

*   **CPU vs GPU**: O sistema está configurado para rodar na **CPU** (`num_gpu=0`) para evitar erros de VRAM em placas com pouca memória, utilizando a RAM do sistema (32GB+ recomendado para fluidez total, mas roda com menos).
*   **Timeouts**: Os tempos limite foram aumentados para acomodar a inferência via CPU.

---
*Criado com ❤️ por Rafa & MestrIA Team*
