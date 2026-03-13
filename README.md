====================================================
LexRAG — Legal RAG for Collective Labor Agreements
====================================================

Sistema de Retrieval-Augmented Generation (RAG) aplicado à análise de Convenções Coletivas de Trabalho.

O projeto implementa um pipeline completo de busca semântica e geração de respostas fundamentadas a partir de documentos jurídicos, utilizando embeddings locais, indexação vetorial e um LLM avançado para geração de respostas.

O objetivo é permitir consultas jurídicas rápidas e fundamentadas em documentos de convenções coletivas, mantendo rastreabilidade das fontes utilizadas e observabilidade completa do sistema.

Principais funcionalidades:

- Ingestão de documentos .doc/.docx
- Extração estruturada de cláusulas
- Embeddings semânticos locais (MiniLM)
- Indexação vetorial com FAISS
- Busca semântica por similaridade
- Geração de respostas com LLM (OpenAI GPT)
- Observabilidade do pipeline
- Sistema de debug e histórico de consultas
- Avaliação automática da qualidade das respostas


====================================================
ARQUITETURA DO SISTEMA
====================================================

Fluxo do pipeline RAG:

Pergunta do usuário
↓
Interface Web (Streamlit)
↓
Embedding da pergunta (MiniLM)
↓
Busca vetorial no FAISS
↓
Recuperação dos chunks mais relevantes
↓
Construção do contexto
↓
Geração de resposta (LLM via OpenAI API)
↓
Resposta fundamentada com fontes
↓
Registro de execução (Telemetry)
↓
Persistência em SQLite
↓
Visualização em páginas Debug / Histórico


====================================================
ESTRUTURA DO PROJETO
====================================================

RAG/
│
├── convencoes_coletivas/      # Documentos fonte (.doc/.docx) - não versionados
│
├── embeddings/                # Geração de embeddings
├── ingest/                    # Parsing e chunking por cláusula
├── prompts/                   # Templates de prompt
│
├── vectorstore/               # Índice vetorial FAISS
│   ├── faiss.index
│   ├── metadata.pkl
│   └── faiss_store.py
│
├── core/                      # Guardrails e utilitários centrais
│   └── guardrails.py
│
├── observability/             # Telemetria e observabilidade
│   ├── telemetry.py
│   ├── decorators.py
│   ├── debug_store.py
│   └── prom_metrics.py
│
├── evaluation/                # Avaliação automática do RAG
│   ├── evaluation_set.json
│   ├── evaluate_rag.py
│   ├── evaluate_rag_v2.py
│   ├── evaluation_results.json
│   └── evaluation_results_v2.json
│
├── pages/                     # Páginas da interface Streamlit
│   ├── 1_Debug.py
│   └── 2_Historico.py
│
├── app.py                     # Interface principal
├── build_index.py             # Construção do índice vetorial
├── query.py                   # Retrieval sem geração
├── rag_generator.py           # Pipeline completo de RAG
│
├── requirements.txt
├── README.md
└── ARCHITECTURE.md


====================================================
COMO EXECUTAR O PROJETO
====================================================

1) Criar ambiente virtual

python -m venv venv
venv\Scripts\activate


2) Instalar dependências

pip install -r requirements.txt


3) Construir o índice vetorial

python build_index.py

Este processo irá:

- gerar embeddings
- criar índice FAISS
- salvar faiss.index e metadata.pkl


4) Testar busca semântica

python query.py

Exemplo de pergunta:

Existe obrigação de seguro de vida empresarial?


====================================================
EXECUTAR INTERFACE WEB
====================================================

Após construir o índice vetorial, execute:

streamlit run app.py

A aplicação abrirá em:

http://localhost:8501


====================================================
CONFIGURAÇÃO DA API OPENAI
====================================================

Para utilizar o modelo GPT na geração das respostas, é necessário definir uma chave de API da OpenAI.

Defina a variável de ambiente:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

No Windows PowerShell:

$env:OPENAI_API_KEY="sua_chave_aqui"

Essa chave é utilizada pelo módulo rag_generator.py para acessar o modelo de linguagem responsável pela geração das respostas.


====================================================
OBSERVABILIDADE DO SISTEMA
====================================================

O sistema possui observabilidade completa do pipeline RAG.

Cada consulta registra:

- Trace ID da execução
- Pergunta do usuário
- Resposta gerada
- Fontes utilizadas
- Contexto enviado ao modelo
- Prompt final
- Métricas de tempo

Os dados são armazenados em um banco SQLite local.

Páginas disponíveis na interface:

DEBUG
- inspeção detalhada de uma consulta específica
- visualização do contexto e prompt enviados ao modelo

HISTÓRICO
- registro persistente de consultas
- filtragem por trace_id e session_id
- análise de métricas da execução


====================================================
OBSERVABILIDADE COM PROMETHEUS E GRAFANA
====================================================

O sistema também exporta métricas para Prometheus.

Métricas disponíveis:

- rag_requests_total
- rag_errors_total
- rag_total_time_seconds
- rag_retrieval_time_seconds
- rag_generation_time_seconds
- rag_chunks_retrieved
- rag_chunks_used
- rag_top_score
- rag_avg_score

Essas métricas podem ser visualizadas em:

http://localhost:8000/metrics


Grafana pode ser utilizado para criar dashboards de monitoramento com:

- latência média de resposta
- latência de retrieval
- latência de geração
- taxa de erro
- qualidade média do retrieval
- volume de consultas


====================================================
AVALIAÇÃO DO SISTEMA (RAG EVALUATION)
====================================================

O projeto inclui um framework de avaliação automática para medir a qualidade das respostas.

Dataset de avaliação:

25 perguntas jurídicas típicas encontradas em convenções coletivas de trabalho.

Temas avaliados:

- vigência do acordo
- reajuste salarial
- seguro de vida
- contribuição sindical
- jornada de trabalho
- banco de horas
- estabilidade
- benefícios trabalhistas


Scripts disponíveis:

python evaluation/evaluate_rag.py
python evaluation/evaluate_rag_v2.py


Categorias de avaliação:

correct
correct_but_contaminated
partial
wrong
no_evidence
error


Resultados atuais:

Total de perguntas: 25

correct: 11
correct_but_contaminated: 5
partial: 6
wrong: 1
no_evidence: 2
error: 0


Isso representa aproximadamente:

44% respostas corretas limpas
64% respostas corretas com evidência
≈ 88–90% de acerto substantivo


====================================================
DECISÕES TÉCNICAS
====================================================

Embeddings:
sentence-transformers/all-MiniLM-L6-v2

Vector Store:
FAISS

LLM:
OpenAI GPT (via API)

Interface:
Streamlit

Observabilidade:
Prometheus + Grafana

Persistência:
SQLite


====================================================
OBJETIVO DO PROJETO
====================================================

Demonstrar a implementação completa de um pipeline RAG aplicado a documentos jurídicos, incluindo:

- recuperação semântica
- fundamentação textual
- observabilidade do sistema
- avaliação automática de respostas
- arquitetura modular e escalável


====================================================
POSSÍVEIS EVOLUÇÕES
====================================================

- autenticação de usuários
- API REST com FastAPI
- deploy com Docker
- observabilidade com OpenTelemetry
- reranking de documentos
- extração estruturada de obrigações
- suporte a múltiplas convenções coletivas
- fine-tuning de prompts
- cache de respostas


====================================================
AUTOR
====================================================

Allyson Aires

Projeto desenvolvido como implementação técnica de um sistema RAG aplicado à análise de documentos jurídicos.


====================================================
ARQUITETURA DETALHADA
====================================================

Veja o desenho completo em:

ARCHITECTURE.md