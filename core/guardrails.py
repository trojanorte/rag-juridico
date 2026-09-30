"""Small input validation helpers for CLI and tests."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    reason: str = ""

def check_input(user_text: str, max_len: int = 800) -> GuardrailResult:
    text = (user_text or "").strip()
    if not text:
        return GuardrailResult(False, "Pergunta vazia.")
    if len(text) > max_len:
        return GuardrailResult(False, "Pergunta muito longa.")
    return GuardrailResult(True)

def safe_refusal(reason: str) -> str:
    return reason
