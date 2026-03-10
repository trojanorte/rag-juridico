# 📚 RAG Jurídico — Convenções Coletivas

Sistema de Retrieval-Augmented Generation (RAG) aplicado à análise de Convenções Coletivas de Trabalho.

O projeto realiza:

- 📄 Ingestão de documentos .doc/.docx
- 🧠 Extração estruturada por cláusulas
- 🔎 Embeddings semânticos locais (MiniLM)
- 📦 Indexação vetorial com FAISS
- 💬 Busca semântica
- 🤖 Geração de respostas fundamentadas com LLM local (Ollama)

---

## 🏗️ Arquitetura do Sistema

Fluxo do RAG:

Pergunta do usuário
↓
Interface Streamlit (app.py)
↓
Embedding da pergunta (MiniLM)
↓
Busca vetorial no FAISS
↓
Recuperação dos chunks relevantes
↓
Montagem do contexto
↓
LLM local (Ollama)
↓
Resposta fundamentada
↓
Registro de execução (telemetry)
↓
Persistência em SQLite (debug_store)
↓
Consulta via páginas Debug / Histórico

---

RAG/
│
├── convencoes_coletivas/      # Documentos fonte (.doc/.docx)
├── embeddings/                # Geração de embeddings
├── ingest/                    # Parsing e chunking por cláusula
├── prompts/                   # Templates de prompt
├── vectorstore/               # FAISS + persistência
│
├── core/                      # Guardrails e utilitários
│   └── guardrails.py
│
├── observability/             # Telemetria e persistência de logs
│   ├── telemetry.py
│   └── debug_store.py
│
├── pages/                     # Páginas Streamlit
│   ├── 1_Debug.py
│   └── 2_Histórico.py
│
├── app.py                     # Interface Streamlit
├── build_index.py             # Construção do índice vetorial
├── query.py                   # Retrieval sem geração
├── rag_generator.py           # Pipeline completo de RAG
│
├── requirements.txt
├── README.md
└── ARCHITECTURE.md

========================================
COMO EXECUTAR
========================================

1) Criar ambiente virtual
python -m venv venv
venv\Scripts\activate

2) Instalar dependências
pip install -r requirements.txt


3) Construir o índice vetorial
python build_index.py

Isso irá:
- gerar embeddings
- criar índice FAISS
- salvar faiss.index e metadata.pkl


4) Testar busca semântica
python query.py
Exemplo de pergunta:
Existe obrigação de seguro de vida empresarial?


========================================
EXECUTAR INTERFACE WEB
========================================

Após construir o índice vetorial, execute:
streamlit run app.py
A aplicação abrirá em:http://localhost:8501

========================================
OBSERVABILIDADE E DEBUG
========================================

O sistema possui um módulo de observabilidade para rastrear a execução do pipeline RAG.

Cada consulta registra:

- Trace ID da execução
- Pergunta do usuário
- Resposta gerada
- Fontes utilizadas
- Contexto enviado ao modelo
- Prompt final
- Métricas de tempo

Essas informações são armazenadas em um banco SQLite local.

Páginas disponíveis:

DEBUG

- visualização da execução atual
- contexto e prompt enviados ao modelo

HISTÓRICO

- registro persistente de consultas
- identificação por trace_id e session_id
- métricas de execução

========================================
DECISÕES TÉCNICAS
========================================

- Embeddings locais com sentence-transformers/all-MiniLM-L6-v2
- Indexação vetorial com FAISS
- LLM open-source rodando localmente via Ollama
- Interface interativa construída com Streamlit
- Observabilidade com módulo próprio de telemetry
- Persistência de consultas em SQLite

========================================
OBJETIVO
========================================

Demonstrar implementação completa de um pipeline RAG aplicado a documentos jurídicos, incluindo:

- recuperação semântica
- fundamentação textual
- independência de APIs externas
- arquitetura modular e escalável

========================================
POSSÍVEIS EVOLUÇÕES
========================================

- autenticação de usuários
- API REST com FastAPI
- deploy com Docker
- observabilidade com OpenTelemetry
- reranking de documentos
- extração estruturada de obrigações

========================================
AUTOR
========================================

Allyson Aires

Projeto desenvolvido como desafio técnico para implementação de sistema RAG aplicado a documentos jurídicos.


========================================
ARQUITETURA
========================================

Veja o desenho completo em:

ARCHITECTURE.md