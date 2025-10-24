
import subprocess, os, sys, json

def main():
    out = "artifacts/reports/sbom.json"
    os.makedirs("artifacts/reports", exist_ok=True)
    try:
        # Try CycloneDX bom for Python env
        subprocess.check_call([sys.executable, "-m", "cyclonedx_py", "-j", "-o", out])
        print(out)
    except Exception as e:
        print("SBOM gated: missing cyclonedx-bom. Install via `pip install cyclonedx-bom`.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
