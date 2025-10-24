
import pandas as pd, os, json, glob
from backend.contracts import save_contract, validate_with_contract

def test_contract_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({'a':[1,2,3], 'b':[0.1,0.2,0.3], 'c':['x','y','z']})
    path = save_contract(df, run_id='t')
    res = validate_with_contract(df, path)
    assert res['ok'] is True
