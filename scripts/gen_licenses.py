"""
gen_licenses.py
Docstring (PL): Generuje prosty zestaw licencji third_party_licenses.txt na podstawie metadanych pip.
"""
from __future__ import annotations
import pkg_resources, os

def main():
    out_dir = "artifacts/sbom"; os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "third_party_licenses.txt")
    with open(out, "w", encoding="utf-8") as f:
        for dist in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
            name = dist.project_name
            ver = dist.version
            lic = dist.get_metadata('METADATA').split("License: ")[-1].splitlines()[0] if dist.has_metadata('METADATA') else "Unknown"
            f.write(f"{name}=={ver} :: {lic}\n")
    print("Zapisano:", out)

if __name__ == "__main__":
    main()