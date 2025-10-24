
from backend.registry import save_manifest
import os, json

def test_save_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = save_manifest("threshold", {"threshold":0.5,"objective":"youden"}, version="1.0.0")
    assert os.path.exists(p)
    j = json.load(open(p, "r", encoding="utf-8"))
    assert j["name"] == "threshold" and j["payload"]["threshold"] == 0.5
    idx = json.load(open("registry/manifests/index.json","r",encoding="utf-8"))
    assert "threshold" in idx["items"]
