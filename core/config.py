"""Runtime settings; relevance must be calibrated with the gold dataset."""
import os

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.82"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1200"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "350"))
MODEL_NAME = os.getenv("LEXRAG_MODEL", "gpt-4.1-mini")
NO_RELEVANT_CONTEXT = "Não encontrei evidência suficiente nesta convenção para responder com segurança. Reformule a pergunta ou consulte outra convenção."
UNVERIFIED_CITATION = "Não foi possível validar as fontes da resposta. Tente reformular a pergunta."
SELECT_DOCUMENT = "Selecione a convenção coletiva que deseja consultar para evitar combinar regras de instrumentos diferentes."
