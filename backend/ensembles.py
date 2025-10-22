from __future__ import annotations

# backend/ensembles.py
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np


# =========================
# Helpers
# =========================

def _project_simplex(w: np.ndarray) -> np.ndarray:
    """
    Rzutuje wektor na prostąksęplę: w_i >= 0 oraz sum(w)=1.
    Algorytm wg. Duchi et al. (2008).
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.size == 0:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, len(u) + 1)) > (cssv - 1))[0]
    rho = rho_idx[-1] if len(rho_idx) else 0
    theta = (cssv[rho] - 1) / float(rho + 1)
    w_proj = np.maximum(w - theta, 0.0)
    s = w_proj.sum()
    if s <= 1e-12:
        return np.ones_like(w_proj) / len(w_proj)
    return w_proj / s


def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    s = np.nansum(ez, axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return ez / s


def _is_prob_matrix(a: np.ndarray) -> bool:
    return a.ndim == 2 and np.all(np.isfinite(a)) and np.all(a >= -1e-8) and np.all(a <= 1 + 1e-8)


def _clip_proba(P: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.nan_to_num(P, nan=eps, posinf=1 - eps, neginf=eps)
    P = np.clip(P, eps, 1 - eps)
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P /= row_sum
    return P


def _ensure_proba_matrix(p: np.ndarray) -> np.ndarray:
    """
    Ujednolica dowolną predykcję klasyfikacyjną do macierzy [n, C]:
      - jeżeli 1D (n,) traktuj jako p(klasa=1) → zbuduj [1-p, p],
      - jeżeli wartości poza [0,1] lub wiersze nie sumują się ~1 → zrób softmax,
      - finalnie clip+renorm.
    """
    p = np.asarray(p)
    if p.ndim == 1:
        p = np.clip(p.astype(float), 0.0, 1.0)
        p = np.c_[1.0 - p, p]
        return _clip_proba(p)
    if not _is_prob_matrix(p):
        # albo logity, albo cokolwiek – użyj softmax po osi klas
        p = _softmax(p, axis=1)
    return _clip_proba(p)


def _infer_problem(preds: List[np.ndarray], y_true: Optional[np.ndarray]) -> str:
    """
    Zgadnij typ problemu:
      - jeżeli jakakolwiek predykcja ma kształt [n, C] z C>=2 → klasyfikacja,
      - jeżeli 1D i y_true wygląda na etykiety {0,1,...} → klasyfikacja,
      - w przeciwnym razie → regresja.
    """
    if len(preds) == 0:
        return "regression"
    shapes = [np.asarray(p).shape for p in preds]
    if any((len(s) == 2 and s[1] >= 2) for s in shapes):
        return "classification"
    if y_true is not None:
        y = np.asarray(y_true).ravel()
        if y.size > 0 and np.issubdtype(y.dtype, np.integer):
            uniq = np.unique(y[~np.isnan(y)])
            if uniq.size <= 20 and uniq.min() >= 0:
                return "classification"
    return "regression"


def _validate_same_length(preds: List[np.ndarray]) -> int:
    n_list = [np.asarray(p).shape[0] for p in preds]
    if len(set(n_list)) != 1:
        raise ValueError(f"All prediction arrays must have the same n_samples, got: {n_list}")
    return n_list[0]


# =========================
# Blending
# =========================

def weighted_blend(
    preds: List[np.ndarray],
    y_true: Optional[np.ndarray] = None,
    metric: Optional[str] = None,
    *,
    problem: Optional[str] = None,
    n_steps: int = 300,
    lr: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniwersalny blending K predykcji:
    - Regresja: średnia ważona minimalizująca MSE (gradient + projekcja na prostąksęplę).
    - Klasyfikacja: średnia ważona prawdopodobieństw minimalizująca log loss.
    - Gdy y_true/metric brak → równe wagi.

    Parameters
    ----------
    preds: list of np.ndarray
        Dla regresji: [n_samples], dla klasyfikacji: [n_samples] lub [n_samples, n_classes].
    y_true: np.ndarray | None
        Prawdziwe wartości/etykiety (dla dopasowania wag).
    metric: str | None
        Ignorowane przy gradientowym dopasowaniu (używamy MSE/LogLoss). Pozostawione dla zgodności.
    problem: "regression" | "classification" | None
        Gdy None, spróbujemy zgadnąć.
    n_steps: int
        Kroki gradientu.
    lr: float
        Learning rate.
    seed: int
        Inicjalizacja losowa wag (lekka perturbacja).

    Returns
    -------
    blend: np.ndarray
        Zblendowane predykcje (wektor dla regresji, macierz prawdopodobieństw dla klasyfikacji).
    w: np.ndarray
        Wagi (≥0, suma=1).
    """
    if len(preds) == 0:
        raise ValueError("No predictions to blend")
    if any(p is None for p in preds):
        raise ValueError("Some prediction arrays are None")

    n = _validate_same_length(preds)
    problem = problem or _infer_problem(preds, y_true)

    # Brak y_true → równe wagi
    k = len(preds)
    if y_true is None or n == 0:
        w = np.ones(k, dtype=float) / k
        if problem == "classification":
            Ps = [_ensure_proba_matrix(p) for p in preds]
            _validate_same_class_dim(Ps)
            P = np.stack(Ps, axis=2)  # [n, C, k]
            blend = np.tensordot(P, w, axes=([2], [0]))  # [n, C]
            return _clip_proba(blend), w
        else:
            P = np.stack([np.asarray(p, dtype=float).ravel() for p in preds], axis=1)  # [n, k]
            blend = np.average(P, axis=1, weights=w)  # [n]
            return blend, w

    rng = np.random.RandomState(seed)
    w = np.ones(k, dtype=float) / k
    w += rng.normal(0, 0.01, size=k)
    w = _project_simplex(w)

    if problem == "regression":
        # P: [n, k]
        P = np.stack([np.asarray(p, dtype=float).ravel() for p in preds], axis=1)
        y = np.asarray(y_true, dtype=float).ravel()

        for _ in range(max(int(n_steps), 0)):
            # grad MSE = 2/n * P^T (P w - y)
            r = P @ w - y
            grad = 2.0 * (P.T @ r) / max(len(y), 1)
            w = _project_simplex(w - lr * grad)

        blend = P @ w
        return blend, w

    else:
        # Klasyfikacja — przerób wszystko na macierze proba [n, C]
        Ps = [_ensure_proba_matrix(p) for p in preds]
        _validate_same_class_dim(Ps)
        P = np.stack(Ps, axis=2)  # [n, C, k]

        # y_true -> one-hot
        y = np.asarray(y_true).ravel()
        if not np.issubdtype(y.dtype, np.integer):
            # spróbuj zrzutować etykiety (np. 'A','B') na 0..C-1
            _, y = np.unique(y, return_inverse=True)
        C = Ps[0].shape[1]
        if y.max() >= C:
            raise ValueError(f"y_true has label {int(y.max())} but predictions have only C={C} classes")
        Y = np.zeros((len(y), C), dtype=float)
        Y[np.arange(len(y)), y] = 1.0

        # kickstart (mała siatka wag jeśli k<=4 – często stabilniejszy start)
        if k <= 4:
            w = _grid_kickstart_for_logloss(P, Y)

        for _ in range(max(int(n_steps), 0)):
            blend = np.tensordot(P, w, axes=([2], [0]))  # [n, C]
            blend = _clip_proba(blend)
            # grad po wagach
            # dL/dw_j = - mean over n [ sum_c Y[n,c] * P[n,c,j] / blend[n,c] ]
            frac = (Y / blend)  # [n, C]
            grad = -np.tensordot(frac, P, axes=([0, 1], [0, 1])) / max(len(y), 1)  # [k]
            w = _project_simplex(w - lr * grad)

        blend = np.tensordot(P, w, axes=([2], [0]))  # [n, C]
        blend = _clip_proba(blend)
        return blend, w


def _validate_same_class_dim(Ps: List[np.ndarray]) -> int:
    """Upewnij się, że wszystkie macierze proba mają ten sam rozmiar C."""
    Cs = [p.shape[1] for p in Ps]
    if len(set(Cs)) != 1:
        raise ValueError(f"All probability matrices must share the same n_classes, got: {Cs}")
    return Cs[0]


def _grid_kickstart_for_logloss(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Bardzo mała siatka wag (k<=4) dla lepszego startu przy logloss.
    P: [n, C, k], Y: [n, C]
    """
    k = P.shape[2]
    if k == 1:
        return np.array([1.0])

    # siatka rozdzielczości 0.1 (ograniczona – szybka)
    grid = np.linspace(0, 1, 11)
    best_w = np.ones(k) / k
    best_ll = np.inf

    def _ll(w):
        B = np.tensordot(P, w, axes=([2], [0]))
        B = _clip_proba(B)
        # logloss
        return -(Y * np.log(B)).sum() / max(len(Y), 1)

    if k == 2:
        for a in grid:
            w = np.array([a, 1 - a])
            ll = _ll(w)
            if ll < best_ll:
                best_ll, best_w = ll, w
    elif k == 3:
        for a in grid:
            for b in grid:
                if a + b <= 1:
                    w = np.array([a, b, 1 - a - b])
                    ll = _ll(w)
                    if ll < best_ll:
                        best_ll, best_w = ll, w
    elif k == 4:
        # rzadkie próbkowanie – unikamy O(n^3)
        for a in grid[::2]:
            for b in grid[::2]:
                if a + b <= 1:
                    rest = 1 - a - b
                    for c in grid[::2]:
                        if c <= rest:
                            w = np.array([a, b, c, rest - c])
                            ll = _ll(w)
                            if ll < best_ll:
                                best_ll, best_w = ll, w
    return best_w


def blend_from_probabilities(
    probas: List[np.ndarray],
    *,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Szybkie miękkie głosowanie: średnia ważona prawdopodobieństw.
    Gdy weights=None -> równe wagi.
    Zwraca macierz [n_samples, n_classes].
    """
    if len(probas) == 0:
        raise ValueError("No probability matrices passed")
    P = np.stack([_ensure_proba_matrix(p) for p in probas], axis=2)  # [n, C, k]
    _ = _validate_same_class_dim([p for p in [pi for pi in [p for p in probas]] if True])  # sanity alias (no-op)
    k = P.shape[2]
    if weights is None:
        w = np.ones(k) / k
    else:
        w = _project_simplex(np.asarray(weights, dtype=float))
    blend = np.tensordot(P, w, axes=([2], [0]))  # [n, C]
    return _clip_proba(blend)


# =========================
# Sklearn-based ensembles
# =========================

def build_stacking(
    problem: str,
    base_estimators: List[Tuple[str, Any]],
    final_estimator: Any,
    *,
    cv: int | None = 5,
    passthrough: bool = False,
    n_jobs: Optional[int] = None
):
    """
    Stacking dla regresji/klasyfikacji. Parametry zgodne z sklearn.
    """
    from sklearn.ensemble import StackingRegressor, StackingClassifier
    if problem.lower().startswith("reg"):
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=passthrough,
            n_jobs=n_jobs
        )
    else:
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=passthrough,
            n_jobs=n_jobs
        )


def build_voting(
    problem: str,
    estimators: List[Tuple[str, Any]],
    *,
    voting: str = "soft"
):
    """
    VotingClassifier/Regressor:
      - klasyfikacja: voting='hard'|'soft' (soft wymaga predict_proba),
      - regresja: zawsze średnia predykcji.
    """
    if problem.lower().startswith("reg"):
        from sklearn.ensemble import VotingRegressor
        return VotingRegressor(estimators=estimators)
    else:
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier(estimators=estimators, voting=voting)
