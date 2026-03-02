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
Embedding da pergunta (MiniLM)  
↓  
Busca vetorial no FAISS  
↓  
Recuperação das cláusulas mais relevantes  
↓  
Montagem de contexto  
↓  
LLM local (Ollama)  
↓  
Resposta fundamentada com base nas cláusulas

---

RAG/
│
├── convencoes_coletivas/     # Documentos fonte (.doc/.docx)
├── embeddings/               # Geração de embeddings
├── ingest/                   # Parsing e chunking por cláusula
├── prompts/                  # Templates de prompt
├── vectorstore/              # FAISS + persistência
├── core/                     # Guardrails e utilitários centrais
│   └── guardrails.py         # Filtros de entrada/saída + anti-jailbreak
│
├── build_index.py            # Indexação inicial
├── query.py                  # Busca semântica (somente retrieval)
├── rag_generator.py          # RAG completo (retrieval + LLM + citações)
├── requirements.txt
├── README.md                 # Como rodar
└── ARCHITECTURE.md

## 🚀 Como Executar

### 1️⃣ Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Instalar dependências
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
Executar:
python rag_generator.py

🧠 Decisões Técnicas
Embeddings locais com sentence-transformers/all-MiniLM-L6-v2
Indexação vetorial com FAISS (similaridade por produto interno)
LLM open-source rodando localmente via Ollama
Separação modular de camadas (ingest, embeddings, vectorstore)

🎯 Objetivo
Demonstrar implementação completa de um pipeline RAG aplicado a documentos jurídicos, com:
Recuperação semântica
Fundamentação textual
Independência de APIs externas
Arquitetura organizada e escalável

🔮 Possíveis Evoluções
Classificação automática de cláusulas
Extração estruturada de obrigações
API com FastAPI
Interface web com Streamlit
Filtros por sindicato e categoria

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