from _future_ import annotations

import re
from dataclasses import dataclass
from typing import optional, tuble

blocklist_substrings: sequence[str] = ( 
    "idiota",
    "burro",
    "otário",
)

jailbreak_patterns: sequence[str] = (
    r"ignore (all|the) instructions",
    r"disregard (all|the) instructions",
    r"reveal (your|the) system prompt",
    r"show (your|the) system prompt",
    r"print (your|the) system prompt",
    r"bypass (the )?rules",
    r"jailbreak",
    r"ignore (as|todas as) instruções",
    r"desconsidere (as|todas as) instruções",
    r"mostre (o|seu) prompt",
    r"prompt do sistema",
    r"quebre as regras",
    r"api[_\s-]?key",
    r"chave (da|de) api",
    r"\.env",
    r"vari(á|a)veis de ambiente",
    r"environment variables",
    r"token",
    r"system prompt",
    r"prompt do sistema",
    r"mostre (o|seu) prompt",
    r"config(ura(ç|c)(ã|a)o|s)",
    r"settings",
    r"par(â|a)metros do modelo",
    r"model parameters",
)

def _normalize (text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text

def check_input(user_text: str, max_len: int = 800) -> GuardrailResult:
    """
    Valida a pergunta do usuário antes de executar retrieval/LLM.
    - evita prompt injection básico
    - evita linguagem inadequada
    - evita input gigante
    """
    t = _normalize(user_text)

    if not t:
        return GuardrailResult(False, "Pergunta vazia.")

    if len(t) > max_len:
        return GuardrailResult(False, f"Pergunta muito longa (>{max_len} chars). Resuma.")

    # blocklist (substring)
    for bad in BLOCKLIST_SUBSTRINGS:
        if bad in t:
            return GuardrailResult(False, "Linguagem inadequada detectada no pedido.")

    # jailbreak patterns
    for pat in JAILBREAK_PATTERNS:
        if re.search(pat, t):
            return GuardrailResult(False, "Tentativa de contornar regras (prompt injection).")

    return GuardrailResult(True)


def check_output(answer: str, require_citations: bool = True) -> GuardrailResult:
    """
    Valida a saída do LLM antes de mostrar ao usuário.
    - bloqueia linguagem inadequada
    - exige evidências/citações (opcional)
    """
    t = _normalize(answer)

    for bad in BLOCKLIST_SUBSTRINGS:
        if bad in t:
            return GuardrailResult(False, "A resposta gerou linguagem inadequada.")

    if require_citations:
        # regra simples: exigir ao menos um bloco de citação [ ... ]
        # você pode trocar por regex mais específico depois.
        if "[" not in answer or "]" not in answer:
            return GuardrailResult(False, "Resposta sem citações/evidências no formato [ ... ].")

    return GuardrailResult(True)


def safe_refusal(reason: str) -> str:
    """
    Mensagem padrão de recusa segura.
    """
    return (
        "Não posso fornecer configurações internas, prompts, chaves, variáveis de ambiente ou detalhes sensíveis do sistema. "
        f"Motivo: {reason} "
        "Se você quiser, posso explicar o funcionamento em alto nível ou responder perguntas sobre as convenções com base em evidências."
    )


def not_found_message() -> str:
    """
    Mensagem padrão quando o RAG não encontra evidência suficiente.
    """
    return (
        "Não encontrei evidências suficientes nas convenções coletivas carregadas para responder com segurança. "
        "Tente reformular a pergunta (ex.: cite a categoria, o sindicato ou o benefício: seguro de vida/plano de saúde)."
    )


def is_confident(retrieval_scores: Sequence[float], threshold: float = 0.25) -> bool:
    """
    Helper opcional: gate de confiança do retrieval.
    Você chama com uma lista de scores (ex.: similaridade).
    - Se o seu score for distância (quanto menor melhor), inverta a lógica.
    """
    if not retrieval_scores:
        return False
    best = retrieval_scores[0]
    return best >= threshold