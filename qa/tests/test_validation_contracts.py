from __future__ import annotations
import pandas as pd
from backend.validation import validate_pandera

def test_validation_contracts():
    df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
    res = validate_pandera(df)
    assert "status" in res