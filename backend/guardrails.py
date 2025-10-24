"""
guardrails.py
Docstring (PL): Bezpieczeństwo promptów i PII-masking. Skanuje tekst pod kątem wrażliwych danych
(emaile, telefony, karty, PESEL) i usuwa/anonimizuje. Wymusza limity długości.
"""
from __future__ import annotations
import re
from typing import Dict, Any

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d[\d\s-]{7,}\d"),
    "card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "pesel": re.compile(r"\b\d{11}\b"),
    # nie skanujemy kluczy API tutaj — klucze przechowywane są oddzielnie i nie logowane
}

DENY_PATTERNS = [
    re.compile(r"(?i)\b(drop\s+table|rm\s+-rf|shutdown|format\s+c:)\b"),
]

def redact_pii(text: str) -> str:
    out = text
    for name, pat in PII_PATTERNS.items():
        out = pat.sub(lambda m: f"<{name}:redacted>", out)
    return out

def check_prompt(text: str, max_chars: int = 4000) -> Dict[str, Any]:
    red = redact_pii(text)[:max_chars]
    denied = [p.pattern for p in DENY_PATTERNS if p.search(text)]
    return {"ok": len(denied)==0, "text": red, "denied": denied}