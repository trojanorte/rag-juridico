```mermaid
flowchart TB

%% ==================================================
%% ETAPA 1 — INDEXAÇÃO (OFFLINE)
%% ==================================================

subgraph INDEXACAO_OFFLINE["Indexação de Documentos (Offline)"]
direction TB

D1[Documentos .doc / .docx<br>Convenções Coletivas]
D2[Parsing de texto]
D3[Chunking por cláusula]
D4[Geração de embeddings]
D5[(FAISS Vector Index)]

D1 --> D2 --> D3 --> D4 --> D5

end


%% ==================================================
%% ETAPA 2 — CONSULTA (ONLINE)
%% ==================================================

subgraph CONSULTA_RAG["Pipeline RAG (Online)"]
direction TB

Q1[Usuário envia pergunta]
Q2[Guardrail de entrada<br>validação da pergunta]
Q3[Embedding da pergunta]
Q4[Busca vetorial Top-K]
Q5{Gate de confiança<br>score mínimo}
Q6[Construção do contexto]
Q7[Construção do prompt]
Q8[LLM - GPT]
Q9[Guardrail de saída]
Q10[Resposta final + citações]

Q1 --> Q2 --> Q3 --> Q4 --> Q5
Q5 -- Contexto suficiente --> Q6
Q5 -- Sem evidência --> Q10
Q6 --> Q7 --> Q8 --> Q9 --> Q10

end


%% ==================================================
%% ETAPA 3 — OBSERVABILITY
%% ==================================================

subgraph OBSERVABILITY["Observabilidade e Monitoramento"]
direction TB

M1[Captura de métricas]
M2[(SQLite / Database)]
M3[Dashboard Streamlit]

M1 --> M2 --> M3

end


%% ==================================================
%% CONEXÕES ENTRE SISTEMAS
%% ==================================================

D5 --> Q4
Q8 --> M1
Q10 --> M1
