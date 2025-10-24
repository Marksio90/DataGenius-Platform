
from __future__ import annotations

BRANDS = {
    "TMIV Purple": {
        "--tmiv-accent": "#8A2BE2",
        "--tmiv-accent-2": "#22d3ee",
        "--tmiv-bg": "#0e1117",
        "--tmiv-surface": "#141823",
        "--tmiv-text": "#e5e7eb",
        "--tmiv-border": "#2b3240",
    },
    "Electric Cyan": {
        "--tmiv-accent": "#06b6d4",
        "--tmiv-accent-2": "#22d3ee",
        "--tmiv-bg": "#071218",
        "--tmiv-surface": "#0b1b24",
        "--tmiv-text": "#e6f6ff",
        "--tmiv-border": "#183544",
    },
    "Sunset Orange": {
        "--tmiv-accent": "#f97316",
        "--tmiv-accent-2": "#f59e0b",
        "--tmiv-bg": "#0e1117",
        "--tmiv-surface": "#1a1410",
        "--tmiv-text": "#fef3c7",
        "--tmiv-border": "#3b2a1f",
    },
    "High Contrast": {
        "--tmiv-accent": "#ffffff",
        "--tmiv-accent-2": "#00ffea",
        "--tmiv-bg": "#000000",
        "--tmiv-surface": "#0a0a0a",
        "--tmiv-text": "#ffffff",
        "--tmiv-border": "#222222",
    },
}

def css_override(tokens: dict[str,str]) -> str:
    lines = [":root{"] + [f"  {k}: {v};" for k,v in tokens.items()] + ["}"]
    return "\n".join(lines)
