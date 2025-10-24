
import os, json, zipfile, hashlib

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

def collect_files():
    files = []
    for root, _, fs in os.walk("artifacts"):
        for f in fs:
            p = os.path.join(root, f)
            if "exports" in p and p.endswith(".zip"):
                continue
            files.append(p)
    for root, _, fs in os.walk("registry"):
        for f in fs:
            files.append(os.path.join(root, f))
    if os.path.exists("docs/README.md"):
        files.append("docs/README.md")
    return files

def main():
    os.makedirs("artifacts/exports", exist_ok=True)
    files = collect_files()
    manifest = {"files": [{"path": p, "sha256": sha256_of_file(p)} for p in files]}
    with open("artifacts/exports/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    zip_path = "artifacts/exports/tmiv_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + ["artifacts/exports/manifest.json"]:
            z.write(p, p)
    print(zip_path)

if __name__ == "__main__":
    main()
