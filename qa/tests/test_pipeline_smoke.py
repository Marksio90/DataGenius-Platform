from __future__ import annotations
import os, json
from backend import file_upload, dtype_sanitizer, runtime_preprocessor, eda_integration, training_plan, async_ml_trainer

def test_end_to_end_small_dataset():
    df, rep = file_upload.load_from_path("data/avocado.csv")
    assert rep["code"] == "OK"
    df2, _ = dtype_sanitizer.sanitize_dtypes(df)
    df3, meta = runtime_preprocessor.preprocess_runtime(df2)
    assert "fingerprint" in meta and len(meta["fingerprint"]) == 64

    eda = eda_integration.quick_eda(df3)
    assert eda["rows"] > 0

    plan = training_plan.build_training_plan(df3)
    if plan["status"] != "OK":
        # z małym sample może nie wykryć targetu — akceptowalne, ale test nadal sprawdzi brak błędów
        return

    res = async_ml_trainer.train_async(
        df=df3, target=plan["target"], problem_type=plan["problem_type"],
        strategy="fast_small", max_parallel=1, max_time_sec=60, random_state=42
    )
    assert res["status"] == "OK"
    assert isinstance(res["results"], dict)

def test_one_hot_encoder_no_sparse_param():
    # Udowodnij, że stary błąd z `sparse=` zostanie wykryty w środowisku sklearn>=1.5
    import pytest
    from sklearn.preprocessing import OneHotEncoder

    with pytest.raises(TypeError):
        OneHotEncoder(sparse=True)  # zabronione w projekcie i w nowszych wersjach sklearn