# Melhorias do LexRAG — implementação de 30/09/2026

## Resumo

O fluxo web agora exige acesso configurado e seleção explícita de uma convenção; a busca não entrega chunks de outro documento e descarta resultados abaixo do limiar. O índice foi reconstruído localmente a partir de 34 DOCX para **1.988 chunks**, com metadados JSONL, manifesto e ponteiro atômico. Por conter texto das convenções, o índice novo permanece ignorado pelo Git; cada instalação precisa reindexar seu corpus autorizado. O prompt saiu do código, a resposta mostra o trecho citado e IDs inexistentes são rejeitados. Esta entrega melhora rastreabilidade; não certifica a interpretação jurídica do LLM. O índice antigo não é carregado.

## Problemas da auditoria corrigidos

| ID | Status | Solução |
|---|---|---|
| R01 | corrigido | Devcontainer usa proteção padrão do Streamlit; acesso do chat e admin exige senha configurada e helper comum. |
| R02 | corrigido | UI exige convenção, fachada exige `document_id` para consulta normativa e FAISS filtra por ele. |
| R03 | corrigido | Removido fallback de 3 resultados; sem score suficiente retorna recusa sem chamar LLM. |
| R04 | parcialmente corrigido | Fonte mostra trecho, cláusula, score e chunk ID; não há página confiável nem validação semântica por afirmação. |
| R05 | parcialmente corrigido | Dataset real de 18 casos e métricas de retrieval; falta revisão jurídica humana e gabarito de resposta. |
| R06 | parcialmente corrigido | Subchunks por sentença/parágrafo e ajuste ao tokenizer E5; estruturas irregulares ainda podem juntar temas. |
| R07 | parcialmente corrigido | IDs, hash, título, cláusula e posição; metadados jurídicos não identificados ficam `null`. |
| R08 | parcialmente corrigido | Novos logs minimizam texto e admin usa helper; registros antigos, senha única e retenção dependem de operação. |
| R09 | corrigido | Novo loader usa JSONL, sem `pickle.load`. |
| R10 | corrigido | Telemetria por contexto de execução via `ContextVar`. |
| R11 | corrigido | Índice e metadados versionados, manifesto validado, publicação via `CURRENT` atômico e cache por versão. |
| R12 | adiado | Conversão `.doc` ainda depende de Word/COM; README documenta o processo e a limitação Linux. |
| R13 | corrigido | CLI exige documento e usa dimensão do índice; guardrail básico foi reparado. |
| R14 | corrigido | Hash SHA-256 do conteúdo normalizado evita documentos idênticos duplicados; duplicatas são registradas. |
| R15 | corrigido | Parser registra arquivos vazios/falhas e interrompe indexação em erro, sem publicar índice parcial. |
| R16 | parcialmente corrigido | Filtro melhora retrieval; BM25 foi medido offline e não promovido por perda de Hit@1. |
| R17 | parcialmente corrigido | Instruções separadas e trechos tratados como dados; citações validadas estruturalmente, sem checagem factual. |
| R18 | parcialmente corrigido | Reescrita só em follow-ups; geração ainda é síncrona e sem streaming. |
| R19 | corrigido | UI registra sucesso e erro, mostra mensagem segura; falha do servidor de métricas vai para log. |
| R20 | parcialmente corrigido | Seleção, preview e cópia foram adicionados; sem upload público, exportação e retomada de conversa. |
| R21 | parcialmente corrigido | Testes, CI e `.env.example`; sem Docker de produção, health check ou migração de banco. |
| R22 | parcialmente corrigido | `observability/db.py` virou fachada do módulo SQL ativo; configuração/caminhos ainda podem ser centralizados. |
| R23 | corrigido | README descreve E5 768D, fluxo real, segurança e limites. |
| R24 | parcialmente corrigido | Status lê índice/manifesto reais; UI segue componentes padrão Streamlit, sem auditoria visual completa. |
| R25 | parcialmente corrigido | README e arquitetura atualizados, prompt ativo; sem screenshot validado ou licença existente. |

**Totais totalmente corrigidos:** P0 2/2; P1 4/10 (R03, R09, R10, R11); P2 4/10 (R13, R14, R15, R19); P3 1/3 (R23). Status parciais não foram contados como corrigidos.

## Arquitetura antes

DOCX → cláusula sem tamanho máximo → E5 → FAISS global + `metadata.pkl` → top-5 com fallback de score → prompt único inline → GPT → fontes indicativas. Chat sem acesso configurado e sem escolha da convenção; logs de conteúdo integral.

## Arquitetura depois

DOCX → cláusula/subchunks com IDs, hash e localização → E5 → FAISS + JSONL + manifesto versionado → escolha de convenção → busca filtrada → limiar 0,82 → contexto com IDs de fonte → instruções separadas na Responses API → validação estrutural de citações → resposta e preview. Telemetria por execução, logs minimizados e autenticação centralizada. Veja `ARCHITECTURE.md`.

## Retrieval antes/depois

Benchmark offline de **16 casos respondíveis e 2 negativos**, derivados das cláusulas do corpus local. As três variantes usam o **índice novo de 1.988 chunks**; A simula a política antiga de busca global, não é uma medição retrospectiva do artefato antigo de 1.653 vetores. Os números medem localização do chunk esperado; não medem correção da resposta jurídica. Tempo é apenas da etapa de busca/ranking, sem embedding ou LLM.

| Variante | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | Busca média |
|---|---:|---:|---:|---:|---:|---:|
| A — E5 global sem filtro | 12,50% | 18,75% | 25,00% | 25,00% | 0,1719 | 0,43 ms |
| B — E5 filtrado (produção) | 93,75% | 93,75% | 93,75% | 93,75% | 0,9375 | 1,54 ms |
| C — E5 filtrado + BM25 experimental | 68,75% | 100,00% | 100,00% | 100,00% | 0,8333 | 3,40 ms |

**Decisão:** manter B como padrão. C melhorou Hit@5 em um caso, mas reduziu Hit@1/MRR e aumentou tempo; não há ganho líquido claro para promoção. Reranker não foi testado: primeiro é preciso ampliar/revisar o dataset e corrigir casos de parsing. Com limiar provisório **0,82**, os 2/2 casos negativos exploratórios foram rejeitados; com 0,35 eram 0/2. Este ajuste em amostra pequena pode não generalizar. No caso Q012, o chunk esperado da cláusula 12x36 não apareceu no top-5 filtrado, e a cláusula parserizada contém texto de tema diferente em outro subchunk; requer revisão do documento/parsing.

## Segurança

O chat e as páginas admin falham fechados sem credenciais em ambiente ou `st.secrets`; CORS/XSRF voltaram ao padrão. O endpoint Prometheus agora escuta somente em `127.0.0.1`. Novo loader não desserializa pickle. Novos registros SQL não persistem pergunta, resposta, prompt nem contexto completos, apenas IDs/métricas/tipo de erro. O arquivo local de secrets e o corpus original permanecem ignorados pelo Git. Ainda faltam rate limiting, identidade individual e política operacional de retenção. Dados antigos de `query_logs` não foram apagados.

## Metadados

Cada chunk novo tem `chunk_id`, `document_id`, `document_hash`, `document_version`, `filename`, `document_title`, `clause_number`, `clause_title`, `section`, `page`, posições e conteúdo. `page` fica `null`. Categoria, sindicatos, abrangência e vigência ficam `null` porque a extração ainda não é confiável. A posição é calculada no texto normalizado; subdivisões por limite do modelo podem ter posição apenas aproximada. O manifesto registra versão, data, modelo, dimensão, número de documentos/chunks, hash do corpus e versão do parser.

## Citações

Antes: lista de arquivo/título/score, com fontes anexadas mesmo sem citação efetiva. Depois: resposta usa `[n]` do contexto e a UI mostra trecho, cláusula, convenção, score e `chunk_id`; citação ausente/inexistente ou de outro documento é rejeitada. Isso comprova **que a referência veio do contexto**, não que a frase gerada seja fiel ao trecho. Página não está disponível via `docx2txt`.

## Avaliação

`evaluation/gold_dataset.json` contém 18 perguntas, IDs de fonte esperados, documento e rótulo `answerable`; perguntas negativas não têm fonte esperada. `evaluate_retrieval.py` mede Hit@1/3/5, recall@5, MRR e tempo da busca, sem API externa. `evaluate_grounding.py` oferece verificações estruturais. Os avaliadores heurísticos antigos estão marcados como legados e desativados para novas execuções. Não há pontuação de precisão jurídica, faithfulness semântica nem revisão humana do conjunto.

## UX

Sidebar com convenção, nova conversa, sair e informações reais do índice; chat com estado de busca/geração, preview expansível e bloco copiável. Trocar convenção limpa a conversa atual. O usuário é solicitado a escolher uma convenção antes da consulta jurídica. Não houve captura de screenshot; teste visual em navegador e revisão mobile/acessibilidade ainda são necessários.

## Testes

- `pytest tests -q -p no:cacheprovider --basetemp .test_tmp`: **12 passed, 0 failed, 0 skipped** em execução local fora do sandbox (o sandbox bloqueou diretórios temporários; 3 avisos de depreciação de biblioteca nativa).
- `ruff check ... --select E4,E7,E9,F`: passou.
- `python -B -m compileall -q` nos módulos e testes Python: passou.
- Parse AST de 27 arquivos Python: passou.
- `evaluation/evaluate_retrieval.py --check-dataset`: 18 casos válidos.
- Benchmark offline: números acima; índice validado com 34 documentos e 1.988 chunks.
- Startup Streamlit em `127.0.0.1:8503`: endpoint de saúde HTTP **200**.
- AppTest: login, seleção de convenção, preview expansível e erro amigável sem exceção na UI; três páginas administrativas abriram com credencial de teste.
- Consulta real via OpenAI: mesma pergunta sobre adicional noturno nas convenções `5b8dcefe3de4e81d` e `4c965e5aa66fc82f`; respostas citaram fontes só do documento selecionado. A primeira cláusula consultada registra 25%, a segunda 20%, conferidos nos trechos do índice. Consulta negativa em `8325efcdd0b3a991` retornou recusa com **0** chamadas de geração; sem seleção retornou pedido de escolha.
- Typecheck estático separado não configurado; CI cobre compilação/imports. Navegação em navegador real e resolução menor não foram executadas por ausência de ferramenta de browser automatizado; não há screenshot.

## Limitações restantes

O benchmark usa poucos casos e nenhuma revisão jurídica humana. O parser ainda erra em formatações não padronizadas (Q012). O limiar é específico deste corpus/versão do E5. A validação de citações é estrutural, não semântica. O app não oferece autenticação individual, rate limiting, sessões persistentes, upload ou gestão de documentos. `.doc` requer conversão externa no Windows. O índice novo precisa estar presente para iniciar a aplicação. Alguns logs antigos ainda contêm conteúdo integral. Navegação mobile/teclado/leitor de tela precisa de validação visual futura.

## Próximas melhorias

1. Revisar manualmente Q012 e outras cláusulas com parsing irregular; ampliar gold dataset com gabarito jurídico por documento e casos negativos realistas.
2. Medir faithfulness por afirmação com revisão humana e testar limiar em corpus/consultas independentes.
3. Implementar autenticação por usuário e retenção de logs se houver disponibilização compartilhada; adicionar proteção de borda/rate limiting.
4. Validar UI em navegador real e resoluções menores, com acessibilidade e preview de documento completo quando possível.
5. Reavaliar BM25/reranking apenas quando o baseline filtrado falhar em casos revisados e houver ganho claro de qualidade/latência.
