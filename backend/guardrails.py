
from __future__ import annotations
import re
from typing import Dict, List

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d[\d\s-]{7,}\d"),
    "iban": re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{1,30}"),
    "pesel_like": re.compile(r"\b\d{11}\b"),
}

def scan_pii(df) -> Dict[str, int]:
    """Prosty skaner PII: zlicza dopasowania na każdej kolumnie tekstowej."""
    report: Dict[str, int] = {}
    for col in df.columns:
        if df[col].dtype == object:
            text = " ".join(map(lambda v: str(v) if v is not None else "", df[col].astype(str).tolist()))
            hits = sum(p.search(text) is not None for p in PII_PATTERNS.values())
            if hits:
                report[col] = int(hits)
    return report

def mask_preview(df, report):
    """Zwraca podgląd danych z prostą maską ••• na potencjalne PII."""
    df = df.copy()
    for col in report:
        df[col] = df[col].astype(str).str.replace(r".", "•", regex=True)
    return df.head(20)
