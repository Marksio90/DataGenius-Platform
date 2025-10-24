from __future__ import annotations
from backend.guardrails import check_prompt, redact_pii

def test_guardrails_pii_and_deny():
    text = "Kontakt: jan.kowalski@example.com, tel +48 501-234-567. drop table users;"
    res = check_prompt(text, max_chars=200)
    assert not res["ok"]
    red = redact_pii(text)
    assert "example.com" not in red