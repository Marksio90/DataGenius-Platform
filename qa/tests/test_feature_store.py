from __future__ import annotations
from backend.feature_store import save_baseline, load_baseline

def test_feature_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_baseline({"eda":{"rows":1}}, name="base")
    data = load_baseline("base")
    assert data["eda"]["rows"] == 1