from __future__ import annotations

# backend/report_export.py

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json, zipfile, joblib, io
import numpy as np

from backend.safe_utils import truthy_df_safe


# -------------------------------------------------------------------
# Matplotlib (headless)
# -------------------------------------------------------------------
def _safe_mpl_import():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _ensure_plots_dir(out_dir: Path) -> Path:
    d = out_dir / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_Xte(meta: dict):
    """Wczytaj próbkę Xte z meta['Xte_sample_json'] (orient='split') jeśli jest."""
    try:
        import pandas as pd
        Xte_json = meta.get("Xte_sample_json")
        if Xte_json is None:
            return None
        return pd.read_json(io.StringIO(Xte_json), orient="split")
    except Exception:
        return None


def _plot_save(fig, path: Path):
    """Zapisz wykres i bezpiecznie zamknij figurę."""
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass


# ===================== POMOCNICZE (etykiety/klasy) =====================

def _to_numpy(x):
    try:
        return np.asarray(x)
    except Exception:
        return np.array(x, dtype=object)


def _resolve_classes(meta: dict) -> Optional[np.ndarray]:
    """
    Spróbuj wyciągnąć kolejność klas zgodną z kolumnami proba:
    - meta['classes'] lub meta['eval']['classes'] (jak w sklearn estimator.classes_)
    """
    ev = meta.get("eval") or {}
    cls = meta.get("classes") or ev.get("classes")
    if cls is None:
        return None
    arr = np.asarray(list(cls))
    return arr


def _binarize_y_for_proba(y_raw, proba_vec, classes: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Dla binarnej klasyfikacji:
    - y_raw: wektor etykiet (dowolny typ),
    - proba_vec: wektor p(y==pos), kształt [n],
    - classes: opcjonalnie kolejność klas (len=2).
    Zwraca: (y_bin 0/1, p_pos, pos_index)
    """
    y = _to_numpy(y_raw)
    p = np.asarray(proba_vec, dtype=float).ravel()

    if classes is not None and len(classes) == 2:
        # wybierz klasę positive jako classes[1] (sklearnowa konwencja)
        pos_label = classes[1]
        y_bin = (y == pos_label).astype(int)
        return y_bin, p, 1

    # brak jawnych classes -> wybierz label, dla którego średnie p jest większe
    uniq = np.unique(y)
    if uniq.size == 2:
        m0 = float(np.nanmean(p[y == uniq[0]])) if np.any(y == uniq[0]) else 0.0
        m1 = float(np.nanmean(p[y == uniq[1]])) if np.any(y == uniq[1]) else 0.0
        pos_label = uniq[1] if m1 >= m0 else uniq[0]
        y_bin = (y == pos_label).astype(int)
        pos_idx = 1 if pos_label is uniq[1] or (classes is not None and pos_label == uniq[1]) else 0
        # pos_idx jest informacyjny; i tak używamy p jako p_pos
        return y_bin, p, pos_idx

    # fallback: spróbuj rzutować do int 0/1
    try:
        y_bin = _to_numpy(y_raw).astype(int)
        # zabezpieczenie, by wartości były 0/1
        y_bin = (y_bin == np.max(y_bin)).astype(int) if set(np.unique(y_bin)) != {0, 1} else y_bin
        return y_bin, p, 1
    except Exception:
        # ostateczny fallback: traktuj największą częstość jako 'neg', resztę 'pos'
        vc = {lab: int(np.sum(y == lab)) for lab in uniq}
        neg_lab = max(vc, key=vc.get)
        y_bin = (y != neg_lab).astype(int)
        return y_bin, p, 1


def _multiclass_align(y_raw, proba_2d, classes: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dla multiclass:
    - jeżeli classes jest znane (w kolejności kolumn proba) → zmapuj y na indeksy klas,
    - inaczej sfaktoryzuj y i załóż, że kolumny proba odpowiadają indeksom 0..C-1.
    Zwraca: (y_idx, pred_idx) — oba w przestrzeni indeksów 0..C-1.
    """
    y = _to_numpy(y_raw)
    P = np.asarray(proba_2d, dtype=float)
    C = P.shape[1]
    pred_idx = np.argmax(P, axis=1)

    if classes is not None and len(classes) == C:
        # mapowanie label -> indeks wg classes
        map_idx = {lab: i for i, lab in enumerate(classes)}
        y_idx = np.array([map_idx.get(lbl, -1) for lbl in y], dtype=int)
        # usuń niezmapowane jako -1 (jeśli jakieś są, confusion_matrix i tak poradzi sobie po liście labels)
        y_idx[y_idx < 0] = -1
        return y_idx, pred_idx

    # fallback: factorize
    uniq, y_idx = np.unique(y, return_inverse=True)
    # jeśli liczba unikalnych etykiet > C, nadmiar potraktuj jako 'inne' (przykro, ale bez classes nie wiemy jak zmapować)
    y_idx = np.clip(y_idx, 0, C - 1)
    return y_idx, pred_idx


# =========================== KLASYFIKACJA ===========================

def _make_confusion_roc_pr(meta: dict, out_dir: Path) -> List[Path]:
    """
    Tworzy:
      - macierz pomyłek (binary/multiclass),
      - ROC + PR + histogram progów (tylko binary).
    Zwraca listę ścieżek PNG.
    """
    plt = _safe_mpl_import()
    if plt is None:
        return []

    saved: List[Path] = []
    dplots = _ensure_plots_dir(out_dir)
    evalp = meta.get("eval") or {}
    y_true = evalp.get("y_true")
    proba = evalp.get("proba")
    thr = float(meta.get("threshold") or 0.5)

    if y_true is None or proba is None:
        return []

    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, confusion_matrix
    )
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    y_raw = _to_numpy(y_true)
    p = _to_numpy(proba)
    classes = _resolve_classes(meta)

    # --- Confusion matrix (binary / multiclass) ---
    if p.ndim == 1:
        # binarna proba: p_pos = p
        y_bin, p_pos, _ = _binarize_y_for_proba(y_raw, p, classes)
        pred = (p_pos >= thr).astype(int)
        labels_for_cm = [0, 1]
        y_cm = y_bin
        pred_cm = pred

    elif p.ndim == 2 and p.shape[1] == 2:
        # binarne proby w 2 kolumnach: p_pos = kolumna 1
        p_pos = p[:, 1].astype(float)
        y_bin, p_pos, _ = _binarize_y_for_proba(y_raw, p_pos, classes)
        pred = (p_pos >= thr).astype(int)
        labels_for_cm = [0, 1]
        y_cm = y_bin
        pred_cm = pred

    elif p.ndim == 2 and p.shape[1] >= 3:
        # multiclass
        y_idx, pred_idx = _multiclass_align(y_raw, p, classes)
        # etykiety osi: jeżeli mamy classes → wypisz je, inaczej 0..C-1
        if classes is not None and len(classes) == p.shape[1]:
            labels_for_cm = list(range(len(classes)))
            tick_lbls = [str(c) for c in classes]
        else:
            labels_for_cm = list(range(p.shape[1]))
            tick_lbls = [str(i) for i in range(p.shape[1])]
        cm = confusion_matrix(y_idx, pred_idx, labels=labels_for_cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(len(labels_for_cm)))
        ax.set_xticklabels(tick_lbls, rotation=45, ha="right")
        ax.set_yticks(range(len(labels_for_cm)))
        ax.set_yticklabels(tick_lbls)
        fig.colorbar(im, ax=ax)
        f_cm = dplots / "confusion_matrix.png"
        _plot_save(fig, f_cm)
        saved.append(f_cm)

        # brak ROC/PR dla multiclass w tej wersji
        return saved

    else:
        # nieznany format
        return []

    # Confusion matrix (binary)
    cm = confusion_matrix(y_cm, pred_cm, labels=labels_for_cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    fig.colorbar(im, ax=ax)
    f_cm = dplots / "confusion_matrix.png"
    _plot_save(fig, f_cm)
    saved.append(f_cm)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_cm, p_pos)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        f_roc = dplots / "roc_curve.png"
        _plot_save(fig, f_roc)
        saved.append(f_roc)
    except Exception:
        pass

    # PR
    try:
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_cm, p_pos)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        f_pr = dplots / "pr_curve.png"
        _plot_save(fig, f_pr)
        saved.append(f_pr)
    except Exception:
        pass

    # Score distribution & threshold
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(p_pos[y_cm == 0], bins=30, alpha=0.5, label="Class 0")
        ax.hist(p_pos[y_cm == 1], bins=30, alpha=0.5, label="Class 1")
        ax.axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
        ax.set_title("Score distribution by class")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.legend()
        f_hist = dplots / "threshold_hist.png"
        _plot_save(fig, f_hist)
        saved.append(f_hist)
    except Exception:
        pass

    return saved


def _make_calibration_curve(meta: dict, out_dir: Path):
    """Krzywa kalibracji (tylko dla binarnej proby klasy pozytywnej)."""
    plt = _safe_mpl_import()
    if plt is None:
        return None

    evalp = meta.get("eval") or {}
    y_true = evalp.get("y_true")
    proba = evalp.get("proba")
    if y_true is None or proba is None:
        return None

    try:
        from sklearn.calibration import calibration_curve

        y_raw = _to_numpy(y_true)
        p = _to_numpy(proba)
        classes = _resolve_classes(meta)

        if p.ndim == 2:
            if p.shape[1] == 2:
                p = p[:, 1]
            else:
                # multiclass – pomijamy w tej wersji
                return None

        y_bin, p_pos, _ = _binarize_y_for_proba(y_raw, p, classes)
        frac_pos, mean_pred = calibration_curve(y_bin, p_pos, n_bins=10, strategy="uniform")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        path = _ensure_plots_dir(out_dir) / "calibration_curve.png"
        _plot_save(fig, path)
        return path
    except Exception:
        return None


def _make_threshold_sweep(meta: dict, out_dir: Path):
    """Przebieg precision/recall/F1 w funkcji progu (binarna klasyfikacja)."""
    plt = _safe_mpl_import()
    if plt is None:
        return None

    evalp = meta.get("eval") or {}
    y_true = evalp.get("y_true")
    proba = evalp.get("proba")
    if y_true is None or proba is None:
        return None

    from sklearn.metrics import precision_score, recall_score, f1_score

    y_raw = _to_numpy(y_true)
    p = _to_numpy(proba)
    classes = _resolve_classes(meta)

    if p.ndim == 2:
        if p.shape[1] == 2:
            p = p[:, 1]
        else:
            return None

    y_bin, p_pos, _ = _binarize_y_for_proba(y_raw, p, classes)

    thrs = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thrs:
        pred = (p_pos >= t).astype(int)
        precs.append(precision_score(y_bin, pred, zero_division=0))
        recs.append(recall_score(y_bin, pred, zero_division=0))
        f1s.append(f1_score(y_bin, pred, zero_division=0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thrs, precs, label="Precision")
    ax.plot(thrs, recs, label="Recall")
    ax.plot(thrs, f1s, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep")
    ax.legend()
    path = _ensure_plots_dir(out_dir) / "threshold_sweep.png"
    _plot_save(fig, path)
    return path


def _make_lift_gain_ks(meta: dict, out_dir: Path) -> List[Path]:
    """Krzywe Cumulative Gain / Lift / KS (binarna klasyfikacja)."""
    plt = _safe_mpl_import()
    if plt is None:
        return []

    evalp = meta.get("eval") or {}
    y_true = evalp.get("y_true")
    proba = evalp.get("proba")
    if y_true is None or proba is None:
        return []

    y_raw = _to_numpy(y_true)
    p = _to_numpy(proba)
    classes = _resolve_classes(meta)

    if p.ndim == 2:
        if p.shape[1] == 2:
            p = p[:, 1]
        else:
            return []

    y_bin, p_pos, _ = _binarize_y_for_proba(y_raw, p, classes)

    order = np.argsort(-p_pos)
    y_sorted = y_bin[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = max(1, int(y_bin.sum()))
    n = len(y_bin)
    pct = np.arange(1, n + 1) / n
    gain = cum_pos / total_pos
    lift = gain / pct
    pos_cdf = cum_pos / total_pos
    neg_cdf = np.cumsum(1 - y_sorted) / max(1, (n - total_pos))
    ks = np.max(np.abs(pos_cdf - neg_cdf))

    saved: List[Path] = []
    dplots = _ensure_plots_dir(out_dir)

    # Gain
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pct, gain, label="Cumulative gain")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
    ax.set_xlabel("Sample fraction")
    ax.set_ylabel("Gain")
    ax.set_title("Cumulative Gain")
    ax.legend()
    f_gain = dplots / "gain_curve.png"
    _plot_save(fig, f_gain)
    saved.append(f_gain)

    # Lift
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pct, lift, label="Lift")
    ax.axhline(1.0, linestyle="--", label="Baseline")
    ax.set_xlabel("Sample fraction")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Curve")
    ax.legend()
    f_lift = dplots / "lift_curve.png"
    _plot_save(fig, f_lift)
    saved.append(f_lift)

    # KS
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pct, pos_cdf, label="Pos CDF")
    ax.plot(pct, neg_cdf, label="Neg CDF")
    ax.set_xlabel("Sample fraction")
    ax.set_ylabel("CDF")
    ax.set_title(f"KS Curve (KS = {ks:.3f})")
    ax.legend()
    f_ks = dplots / "ks_curve.png"
    _plot_save(fig, f_ks)
    saved.append(f_ks)

    return saved


def _load_features_for_importance(meta: dict):
    """
    Próbuje wyciągnąć cechy po transformacjach z Pipeline (krok 'pre' jeśli istnieje).
    Zwraca: (model, estimator, Xt, feature_names) lub (None, None, None, None).
    """
    try:
        model = meta.get("model")
        Xte = _load_Xte(meta)
        if model is None or Xte is None:
            return None, None, None, None

        from sklearn.pipeline import Pipeline

        pre = None
        est = model
        if isinstance(model, Pipeline):
            pre = model.named_steps.get("pre") or model.named_steps.get("transformer")
            est = model.named_steps.get("model", model.steps[-1][1])
        if pre is None and isinstance(model, Pipeline):
            # Fallback – transformuj całym pipeline bez ostatniego estymatora
            try:
                if len(model.steps) >= 2:
                    from sklearn.pipeline import Pipeline as _P
                    pre = _P(model.steps[:-1])
                else:
                    pre = None
            except Exception:
                pre = None

        if pre is None:
            return None, None, None, None

        Xt = pre.transform(Xte)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(Xt.shape[1])]
        return model, est, Xt, feat_names
    except Exception:
        return None, None, None, None


def _feature_importance(meta: dict, out_dir: Path):
    """
    Rysuje wykres ważności cech:
    - feature_importances_ / coef_
    - w razie braku -> permutation_importance (fallback ze scoringiem zależnym od problemu)
    """
    plt = _safe_mpl_import()
    if plt is None:
        return None

    try:
        model, est, Xt, feat_names = _load_features_for_importance(meta)
        if Xt is None:
            return None

        importances = None

        # (1) Drzewa / Boostingi
        try:
            importances = getattr(est, "feature_importances_", None)
        except Exception:
            importances = None

        # (2) Modele liniowe
        if importances is None:
            try:
                coef = getattr(est, "coef_", None)
                if coef is not None:
                    importances = np.abs(np.asarray(coef).ravel())
            except Exception:
                importances = None

        # (3) Permutation importance (fallback)
        if importances is None:
            try:
                from sklearn.inspection import permutation_importance
                evalp = meta.get("eval") or {}
                y_true = evalp.get("y_true")
                if y_true is None:
                    return None
                y = _to_numpy(y_true)

                problem = (meta.get("problem") or "").lower()
                # bezpieczniejsze dobieranie scoringu
                if "classif" in problem:
                    # gdy multiclass roc_auc się wywraca – użyj accuracy
                    scoring = "roc_auc"
                    try:
                        _ = permutation_importance(est, Xt, y, scoring=scoring, n_repeats=1, random_state=0)
                    except Exception:
                        scoring = "accuracy"
                else:
                    scoring = "r2"

                perm = permutation_importance(est, Xt, y, scoring=scoring, n_repeats=3, random_state=42)
                importances = perm.importances_mean
            except Exception:
                return None

        order = np.argsort(importances)[::-1][:20]
        vals = np.asarray(importances)[order]
        names = np.asarray(feat_names)[order]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.barh(range(len(vals))[::-1], vals[::-1])
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(list(names[::-1]))
        ax.set_title("Feature Importance (top 20)")
        ax.set_xlabel("Importance")

        path = _ensure_plots_dir(out_dir) / "feature_importance.png"
        _plot_save(fig, path)
        return path
    except Exception:
        return None


# ============================ REGRESJA ============================

def _make_regression_plots(meta: dict, out_dir: Path) -> List[Path]:
    """
    Dla regresji: True vs Pred, histogram residuów, Residuals vs Fitted, QQ-plot.
    """
    plt = _safe_mpl_import()
    if plt is None:
        return []

    model = meta.get("model")
    Xte = _load_Xte(meta)
    evalp = meta.get("eval") or {}
    y_true = evalp.get("y_true")
    pred = evalp.get("pred")
    if model is None or Xte is None or y_true is None:
        return []

    y = _to_numpy(y_true).astype(float)
    if pred is None:
        try:
            pred = model.predict(Xte).tolist()
        except Exception:
            return []
    pred = _to_numpy(pred).astype(float)
    resid = y - pred

    saved: List[Path] = []
    dplots = _ensure_plots_dir(out_dir)

    # True vs Pred
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y, pred, s=10)
    m = max(np.max(y), np.max(pred))
    m0 = min(np.min(y), np.min(pred))
    ax.plot([m0, m], [m0, m], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("True vs Predicted")
    f_tv = dplots / "true_vs_pred.png"
    _plot_save(fig, f_tv)
    saved.append(f_tv)

    # Residuals histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(resid, bins=30)
    ax.set_title("Residuals histogram")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    f_rh = dplots / "residuals_hist.png"
    _plot_save(fig, f_rh)
    saved.append(f_rh)

    # Residuals vs Fitted
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pred, resid, s=10)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Fitted (pred)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Fitted")
    f_rvf = dplots / "residuals_vs_fitted.png"
    _plot_save(fig, f_rvf)
    saved.append(f_rvf)

    # QQ plot
    try:
        from scipy import stats
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(resid, dist="norm", plot=ax)
        ax.set_title("QQ Plot of residuals")
        f_qq = dplots / "residuals_qq.png"
        _plot_save(fig, f_qq)
        saved.append(f_qq)
    except Exception:
        pass

    return saved


# ============================ ONNX EXPORT ============================

def export_onnx_model(model, out_dir: Path) -> Optional[Path]:
    """
    Eksport modelu do ONNX (jeśli dostępny skl2onnx).
    Próbuje wywnioskować liczbę cech z modelu (n_features_in_), w przeciwnym razie 100.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_features = getattr(model, "n_features_in_", None)
        if not isinstance(n_features, int) or n_features <= 0:
            n_features = 100
        initial_type = [("input", FloatTensorType([None, n_features]))]

        onx = convert_sklearn(model, initial_types=initial_type, target_opset=16)
        out = out_dir / "model.onnx"
        with open(out, "wb") as f:
            f.write(onx.SerializeToString())
        return out
    except Exception:
        return None


# ============================ RAPORT + TOP-K ============================

def _img_to_b64(path: Path) -> str:
    try:
        import base64
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def build_report(meta: dict, out_dir: Path) -> Path:
    """
    Buduje statyczny raport HTML (i opcjonalnie PDF przez pdfkit/wkhtmltopdf).
    Zwraca ścieżkę do HTML-a.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    title = meta.get("title", "TMIV Ultra Pro — Report")
    problem = meta.get("problem", "-")
    primary = meta.get("primary_metric")
    best_key = meta.get("best_model_key")
    thr = meta.get("threshold")
    lb = meta.get("leaderboard") or []

    figs = []
    for name in [
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "threshold_hist.png",
        "calibration_curve.png",
        "threshold_sweep.png",
        "gain_curve.png",
        "lift_curve.png",
        "ks_curve.png",
        "feature_importance.png",
        "shap_summary.png",  # (opcjonalnie jeśli gdzieś generujesz)
        "true_vs_pred.png",
        "residuals_hist.png",
        "residuals_vs_fitted.png",
        "residuals_qq.png",
    ]:
        p = plots_dir / name
        if p.exists():
            figs.append((name.replace(".png", "").replace("_", " ").title(), _img_to_b64(p)))

    # Leaderboard
    lb_html = ""
    if truthy_df_safe(lb):
        try:
            import pandas as pd
            lb_html = pd.DataFrame(lb).to_html(index=False, border=0)
        except Exception:
            lb_html = "<p>(leaderboard not available)</p>"

    # Meta bez surowego modelu
    try:
        meta_light = {k: v for k, v in meta.items() if k != "model"}
        meta_json = json.dumps(meta_light, indent=2, default=lambda o: str(o))
    except Exception:
        meta_json = "{}"

    html = f"""<!DOCTYPE html>
<html lang="pl"><head>
<meta charset="utf-8" /><title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; background:#f6f7fb; }}
h1 {{ margin-bottom: 0; }}
h2 {{ margin-top: 32px; }}
.card {{ background:#fff; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.08); padding:16px; margin:16px 0; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap:16px; }}
img {{ width:100%; height:auto; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.06);}}
code, pre {{ background:#0b1020; color:#d8e1ff; padding:12px; border-radius:8px; overflow:auto; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ padding:8px 10px; border-bottom:1px solid #eee; text-align:left; }}
</style>
</head><body>
<h1>TMIV — Ultra Pro: Raport</h1>
<div class="card">
<p><b>Problem:</b> {problem} &nbsp;|&nbsp; <b>Best:</b> {best_key} &nbsp;|&nbsp; <b>Primary:</b> {primary} &nbsp;|&nbsp; <b>Threshold:</b> {thr}</p>
</div>

<h2>Leaderboard</h2>
<div class="card">{lb_html}</div>

<h2>Wykresy</h2>
<div class="grid">
{''.join(f'<div class="card"><h3 style="margin:0 0 8px 0">{t}</h3><img src="data:image/png;base64,{b64}" /></div>' for (t,b64) in figs)}
</div>

<h2>Meta</h2>
<div class="card"><pre>{meta_json}</pre></div>
</body></html>"""
    out_path = out_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")

    # PDF (opcjonalnie)
    try:
        import pdfkit  # wymaga zainstalowanego wkhtmltopdf w systemie
        pdf_path = out_dir / "report.pdf"
        pdfkit.from_file(str(out_path), str(pdf_path))
    except Exception:
        pass

    return out_path


def save_topk_indices(meta: dict, out_dir: Path, k: float = 0.1) -> Path:
    """
    Zapisuje indeksy top-k% po skali proba (binarne). Zwraca ścieżkę CSV.
    """
    try:
        import pandas as pd
        evalp = meta.get("eval") or {}
        proba = evalp.get("proba")
        if proba is None:
            raise RuntimeError("No probabilities to compute top-k.")
        p = _to_numpy(proba)
        if p.ndim == 2 and p.shape[1] >= 2:
            p = p[:, 1]
        n = len(p)
        topn = max(1, int(n * float(k)))
        idx = np.argsort(-p)[:topn]
        df = pd.DataFrame({"row_index": idx, "score": p[idx]})
        out = out_dir / f"top_{int(k*100)}pct_indices.csv"
        df.to_csv(out, index=False)
        return out
    except Exception:
        return out_dir / "topk_indices.csv"


# ============================ MASTER EXPORT ============================

def export_bundle(bundle: Dict[str, Any], out_dir: Path, export_onnx: bool = False) -> Path:
    """
    Generuje komplet artefaktów:
      - meta.json (bez surowego modelu),
      - wykresy (wg problemu),
      - feature importance,
      - model.joblib (+ opcjonalnie ONNX),
      - leaderboard.csv,
      - report.html (+ opcjonalnie report.pdf),
      - ZIP całego folderu.
    Zwraca ścieżkę do ZIP-a.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = bundle.get("ultra", {}) or {}

    # meta.json (bez modelu)
    try:
        meta_light = {k: v for k, v in meta.items() if k != "model"}
        (out_dir / "meta.json").write_text(
            json.dumps(meta_light, indent=2, default=lambda o: str(o)), encoding="utf-8"
        )
    except Exception:
        pass

    # Wykresy / importance
    problem = str(meta.get("problem") or "").lower()
    if "classif" in problem:
        _make_confusion_roc_pr(meta, out_dir)
        _make_calibration_curve(meta, out_dir)
        _make_threshold_sweep(meta, out_dir)
        _make_lift_gain_ks(meta, out_dir)
        _feature_importance(meta, out_dir)
    else:
        _make_regression_plots(meta, out_dir)
        _feature_importance(meta, out_dir)

    # Dump model
    model = meta.get("model")
    if model is not None:
        try:
            joblib.dump(model, out_dir / "model.joblib")
        except Exception:
            pass
        if bool(export_onnx):
            export_onnx_model(model, out_dir)

    # Leaderboard CSV
    try:
        import pandas as pd
        lb = meta.get("leaderboard") or []
        if truthy_df_safe(lb):
            pd.DataFrame(lb).to_csv(out_dir / "leaderboard.csv", index=False)
    except Exception:
        pass

    # Raport
    try:
        build_report(meta, out_dir)
    except Exception:
        pass

    # ZIP
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception:
            pass
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            z.write(p, arcname=p.relative_to(out_dir))

    return zip_path
