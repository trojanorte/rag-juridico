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

## 🚀 Como Executar

### 1️⃣ Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
### 2️⃣ Instalar dependências
pip install -r requirements.txt
📦 Construir o índice vetorial
python build_index.py
Isso irá:

Gerar embeddings

Criar índice FAISS

Salvar faiss.index e metadata.pkl

🔎 Testar busca semântica
python query.py
Exemplo de pergunta:

Existe obrigação de seguro de vida empresarial?
🤖 Executar RAG completo (com geração)
É necessário ter o Ollama instalado.

Baixar modelo:
ollama pull qwen2.5:1.5b

---

# 4️⃣ Adicionar seção de Observabilidade

Adicione esta nova seção no README:

```markdown
## 📊 Observabilidade e Debug

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

### Páginas disponíveis

**Debug**
- Visualização da execução atual
- Contexto e prompt enviados ao modelo

**Histórico**
- Registro persistente de consultas
- Identificação por `trace_id` e `session_id`
- Métricas de execução

## 🖥️ Executar Interface Web

Após construir o índice vetorial, execute a interface:

```bash
streamlit run app.py

🧠 Decisões Técnicas
Embeddings locais com sentence-transformers/all-MiniLM-L6-v2
Indexação vetorial com FAISS (similaridade por produto interno)
LLM open-source rodando localmente via Ollama
Interface interativa construída com Streamlit
Observabilidade e rastreabilidade com módulo próprio de telemetry
Persistência de consultas em SQLite

🎯 Objetivo
Demonstrar implementação completa de um pipeline RAG aplicado a documentos jurídicos, com:
Recuperação semântica
Fundamentação textual
Independência de APIs externas
Arquitetura organizada e escalável

🔮 Possíveis Evoluções
Autenticação de usuários
API REST com FastAPI
Deploy em container Docker
Observabilidade com OpenTelemetry
Reranking de documentos
Extração estruturada de obrigações

👩‍💻 Autor
Allyson Aires
Projeto desenvolvido como desafio técnico para implementação de sistema RAG aplicado a documentos jurídicos.

## 📐 Arquitetura

Veja o desenho completo em [ARCHITECTURE.md]
---

# 🚀 Depois disso

Rode:

```powershell
git add README.md
git commit -m "docs: add project README"
git push