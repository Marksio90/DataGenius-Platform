from __future__ import annotations
import pandas as pd
from backend import file_upload, dtype_sanitizer

def test_file_upload_errors():
    # Nieobsługiwane rozszerzenie
    import io
    df, rep = file_upload.load_from_bytes("file.unsupported", b"abc")
    assert rep["code"] == "TMIV-IO-002"

def test_dtype_sanitizer_contracts():
    df = pd.DataFrame({"a":["1","2","3"], "b":["2020-01-01","2020-01-02","2020-01-03"]})
    df2, rep = dtype_sanitizer.sanitize_dtypes(df)
    # powinna spróbować konwersji 'a' do liczby, 'b' do daty
    assert "a" in rep["changes"] or "b" in rep["changes"]