# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def test_pipeline_predict_consistency():
    """
    Sprawdza, że po treningu:
    - pipeline istnieje i potrafi przewidywać,
    - predykcje mają prawidłowy rozmiar i są deterministyczne,
    - przewidywanie działa niezależnie od kolejności kolumn,
    - (jeśli dostępne) predict_proba zwraca prawidłowe kształty i zakresy.
    """
    # Synthetic small dataset, mixed types
    n = 120
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "num1": rng.randn(n) * 10 + 5,
            "num2": rng.rand(n),
            "cat1": rng.choice(["A", "B", "C"], size=n),
            "bool1": rng.choice([True, False], size=n),
            "y": rng.choice([0, 1], size=n),
        }
    )

    from backend.ml_integration import MLModelTrainer

    trainer = MLModelTrainer()
    res = trainer.train_model(
        df=df,
        target_column="y",
        problem_type="Klasyfikacja",
        train_size=0.8,
        random_state=13,
    )

    # Ensure pipeline exists and is accessible
    assert "y" in trainer.trained_models, "Brak wpisu dla targetu 'y' w trained_models."
    assert "pipeline" in trainer.trained_models["y"], "Brak klucza 'pipeline' dla targetu 'y'."

    pipe = trainer.trained_models["y"]["pipeline"]
    assert pipe is not None, "Pipeline jest None."
    assert hasattr(pipe, "predict"), "Pipeline nie implementuje metody predict."

    # Prepare X/y
    X = df.drop(columns=["y"])
    y = df["y"]
    # Basic predict
    y_hat = pipe.predict(X)
    assert len(y_hat) == len(y), "Długość predykcji nie zgadza się z długością y."
    assert pd.notnull(pd.Series(y_hat)).all(), "Predykcje zawierają NaN."

    # Determinism with identical input
    y_hat_again = pipe.predict(X.copy())
    np.testing.assert_array_equal(
        y_hat, y_hat_again, err_msg="Predykcje nie są deterministyczne dla tego samego X."
    )

    # Column order invariance (common for ColumnTransformer by name)
    X_shuffled = X.sample(frac=1, axis=1, random_state=7)  # permute column order
    y_hat_shuf = pipe.predict(X_shuffled)
    np.testing.assert_array_equal(
        y_hat, y_hat_shuf, err_msg="Predykcje zmieniają się po zmianie kolejności kolumn."
    )

    # If available, check predict_proba shape/range
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        assert proba.shape[0] == len(X), "predict_proba: nieprawidłowa liczba wierszy."
        # Klasyfikacja binarna lub wieloklasowa – liczba kolumn >= 2
        assert proba.ndim == 2 and proba.shape[1] >= 2, "predict_proba: oczekiwano co najmniej 2 kolumn (klasy)."
        assert np.isfinite(proba).all(), "predict_proba: wartości niefinite."
        assert (proba >= 0).all() and (proba <= 1).all(), "predict_proba: wartości poza zakresem [0,1]."
        # Suma po klasach ~1 (tolerancja numeryczna)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "predict_proba: wiersze nie sumują się do 1."
