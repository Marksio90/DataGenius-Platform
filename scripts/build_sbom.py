"""
build_sbom.py
Docstring (PL): Generator SBOM (CycloneDX) – gated. Jeśli biblioteka jest niedostępna,
tworzy prosty fallback JSON z listą pakietów pip.
"""
from __future__ import annotations
import json, subprocess, os
from datetime import datetime

def main():
    out_dir = "artifacts/sbom"; os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "sbom.json")
    try:
        # Próbujemy cyclonedx-bom (jeśli zainstalowany)
        # cyclonedx-py works via CLI "cyclonedx-py"
        res = subprocess.run(["cyclonedx-py", "-o", out, "-F", "json"], check=True, capture_output=True, text=True)
        print("SBOM wygenerowany:", out)
    except Exception as e:
        # Fallback: listuj pip freeze
        pkgs = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
        data = {"generated": datetime.utcnow().isoformat()+"Z", "packages": pkgs, "note":"Fallback SBOM (brak cyclonedx)"
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("SBOM fallback zapisany:", out)

if __name__ == "__main__":
    main()