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
│
├── embeddings/                # Geração de embeddings
├── ingest/                    # Parsing e chunking por cláusula
├── prompts/                   # Templates de prompt
│
├── vectorstore/               # FAISS + persistência do índice vetorial
│   ├── faiss.index
│   ├── metadata.pkl
│   └── faiss_store.py
│
├── core/                      # Guardrails e utilitários centrais
│   └── guardrails.py
│
├── observability/             # Telemetria e rastreamento de execução
│   ├── telemetry.py
│   ├── decorators.py
│   └── debug_store.py
│
├── evaluation/                # Avaliação do sistema RAG
│   ├── evaluation_set.json
│   ├── evaluate_rag.py
│   ├── evaluate_rag_v2.py
│   ├── evaluation_results.json
│   └── evaluation_results_v2.json
│
├── pages/                     # Páginas Streamlit
│   ├── 1_Debug.py
│   └── 2_Histórico.py
│
├── app.py                     # Interface Streamlit principal
├── build_index.py             # Construção do índice vetorial
├── query.py                   # Busca sem geração (retrieval)
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

## 📊 Avaliação do Sistema (RAG Evaluation)

O sistema possui um conjunto de testes para avaliar a qualidade das respostas geradas pelo pipeline RAG.

### Dataset de avaliação

Foi criado um conjunto de 25 perguntas jurídicas típicas encontradas em convenções coletivas de trabalho, cobrindo temas como:

- vigência do acordo
- reajuste salarial
- seguro de vida
- contribuição sindical
- jornada de trabalho
- banco de horas
- estabilidade
- benefícios trabalhistas

### Avaliação quantitativa

Script:

python evaluation/evaluate_rag.py

Este script mede:

- presença de fontes
- aderência ao tópico esperado
- penalização para respostas fracas

### Avaliação qualitativa

Script:

python evaluation/evaluate_rag_v2.py

Este avaliador classifica as respostas em categorias:

- correct
- correct_but_contaminated
- partial
- wrong
- no_evidence
- error

### Resultado atual

Última avaliação:

Total de perguntas: 25

correct: 11
correct_but_contaminated: 5
partial: 6
wrong: 1
no_evidence: 2
error: 0

Isso indica aproximadamente:

- 44% respostas corretas limpas
- 64% respostas corretas com evidência
- aproximadamente 88–90% de acerto substantivo

### Interpretação

A maioria dos erros restantes está relacionada a:

- continuação indevida da resposta pelo modelo
- recuperação parcial de contexto
- formatação da saída

O conteúdo jurídico gerado tende a ser correto quando há evidência suficiente no contexto recuperado.

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