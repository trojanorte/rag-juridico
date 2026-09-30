from prometheus_client import Counter, Gauge, Histogram, REGISTRY, start_http_server


def _get_or_create_counter(name: str, documentation: str):
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return Counter(name, documentation)


def _get_or_create_gauge(name: str, documentation: str):
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return Gauge(name, documentation)


def _get_or_create_histogram(name: str, documentation: str):
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return Histogram(name, documentation)


rag_requests_total = _get_or_create_counter(
    "rag_requests_total",
    "Total de consultas ao sistema RAG"
)

rag_errors_total = _get_or_create_counter(
    "rag_errors_total",
    "Total de erros do sistema RAG"
)

rag_total_time_seconds = _get_or_create_histogram(
    "rag_total_time_seconds",
    "Tempo total de resposta do RAG"
)

rag_retrieval_time_seconds = _get_or_create_histogram(
    "rag_retrieval_time_seconds",
    "Tempo da etapa de retrieval"
)

rag_generation_time_seconds = _get_or_create_histogram(
    "rag_generation_time_seconds",
    "Tempo da etapa de geração"
)

rag_chunks_retrieved = _get_or_create_gauge(
    "rag_chunks_retrieved",
    "Quantidade de chunks recuperados"
)

rag_chunks_used = _get_or_create_gauge(
    "rag_chunks_used",
    "Quantidade de chunks usados no contexto"
)

rag_top_score = _get_or_create_gauge(
    "rag_top_score",
    "Maior score entre os chunks recuperados"
)

rag_avg_score = _get_or_create_gauge(
    "rag_avg_score",
    "Score médio dos chunks recuperados"
)

_metrics_started = False


def start_metrics_server(port: int = 8000) -> None:
    global _metrics_started

    if _metrics_started:
        return

    try:
        start_http_server(port)
        _metrics_started = True
    except OSError:
        _metrics_started = True