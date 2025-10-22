from backend.safe_utils import truthy_df_safe
# backend/async_ml_trainer.py - POPRAWIONA WERSJA z run_async_in_streamlit
# -*- coding: utf-8 -*-
import asyncio
import concurrent.futures
import time
from typing import Dict, Any, Callable, Optional, Tuple, List
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import clone
import threading
from functools import wraps
import warnings


class AsyncMLTrainer:
    """Asynchroniczny trainer ML z background processing (ThreadPoolExecutor + asyncio).

    Funkcje:
    - train_models_async(...): trenuje wiele modeli współbieżnie (bezpieczne CV, fallback, metryki).
    - cleanup(): zwalnia wątki.
    - cancel_task(key): anuluje task (jeśli został zarejestrowany).
    """

    def __init__(self, max_workers: int = 2):
        # Mniej wątków = mniejsze ryzyko zatykania CPU przy ciężkich modelach
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.training_tasks: Dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()

    # --- pomocnicze ---
    @staticmethod
    def _norm_problem(problem_type: str) -> str:
        p = (problem_type or "").strip().lower()
        # akceptujemy: "klasyfikacja", "classification", "class", "clf", itd.
        if any(x in p for x in ["klasyf", "class"]):
            return "classification"
        return "regression"

    @staticmethod
    def _pick_scoring(problem_type: str, preferred: Optional[str] = None) -> str:
        """Dobiera scoring z uwzględnieniem preferencji (jeśli działa w sklearn)."""
        p = AsyncMLTrainer._norm_problem(problem_type)
        if truthy_df_safe(preferred):
            alias = preferred.lower()
            mapping = {
                "f1_weighted": "f1_weighted",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "roc_auc_ovr": "roc_auc_ovr",
                "accuracy": "accuracy",
                "precision_weighted": "precision_weighted",
                "recall_weighted": "recall_weighted",
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "neg_root_mean_squared_error": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "neg_mean_absolute_error": "neg_mean_absolute_error",
            }
            return mapping.get(alias, ("accuracy" if p == "classification" else "r2"))
        return "accuracy" if p == "classification" else "r2"

    @staticmethod
    def _is_fast_model_name(name: str) -> bool:
        n = name.replace("_", " ").lower()
        fast_keys = [
            "randomforest", "random forest", "rf",
            "extratrees", "extra trees",
            "histgradient", "hgb", "hist gradient",
            "xgb", "xgboost",
            "lgbm", "lightgbm",
            "catboost",
            "logistic", "ridge", "linear", "sgd"
        ]
        return any(k in n for k in fast_keys)

    @staticmethod
    def _can_stratify(y: np.ndarray) -> bool:
        """Stratyfikuj tylko, jeśli wszystkie klasy mają ≥2 próbki (inaczej train_test_split rzuci błąd)."""
        try:
            _, counts = np.unique(y, return_counts=True)
            return (counts.min() >= 2) and (len(counts) > 1)
        except Exception:
            return False

    # --- API główne ---
    async def train_models_async(
        self,
        algorithms: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        problem_type: str = "Classification",
        preferred_scoring: Optional[str] = None,
        task_key: Optional[str] = None,
        total_timeout_sec: int = 900
    ) -> Dict[str, Dict[str, float]]:
        """Trenuje modele asynchronicznie (bezpieczne CV + fallback train/test).

        Args:
            algorithms: {nazwa_modelu: estimator}
            X_train, y_train: dane treningowe
            cv_folds: liczba foldów do CV
            progress_callback: (progress_float, nazwa_modelu, status) -> None
            problem_type: 'Klasyfikacja' / 'Classification' / 'Regresja' / 'Regression'
            preferred_scoring: preferowana metryka (np. 'f1_weighted' / 'r2' / 'rmse' itd.)
            task_key: opcjonalny klucz do zarządzania/anulowania
            total_timeout_sec: limit czasu na całą rundę (sekundy)

        Returns:
            Dict[nazwa_modelu] => {'cv_score': float, 'cv_std': float, 'training_time': float, 'status': 'completed'|'failed', ...}
        """
        warnings.filterwarnings("ignore")  # wycisz np. ConvergenceWarning
        ptype = self._norm_problem(problem_type)
        scoring = self._pick_scoring(ptype, preferred_scoring)

        # Przy dużych zbiorach – filtrujemy tylko szybkie modele (jeśli da się coś wybrać)
        if len(X_train) > 10000:
            filtered = {n: m for n, m in algorithms.items() if self._is_fast_model_name(n)}
            if truthy_df_safe(filtered):
                algorithms = filtered

        # Funkcja jednostkowego treningu (wykonywana w wątkach executora)
        def train_single_model(name: str, model: Any) -> Dict[str, Any]:
            try:
                start_time = time.time()

                # Klonuj estymator, aby uniknąć współdzielenia stanu między wątkami/foldami
                try:
                    est = clone(model)
                except Exception:
                    est = model  # w ostateczności użyj oryginału

                # 1) Spróbuj CV
                try:
                    cv_scores = cross_val_score(
                        est, X_train, y_train,
                        cv=cv_folds,
                        n_jobs=1,          # ważne w wątkach: nie rozmnażamy kolejnych procesów
                        scoring=scoring if scoring != "roc_auc_ovr" else "roc_auc_ovr"
                    )
                    if hasattr(cv_scores, "__iter__") and len(cv_scores) > 0:
                        mean_score = float(np.mean(cv_scores))
                        std_score = float(np.std(cv_scores))
                    else:
                        raise ValueError("Puste wyniki CV")
                except Exception:
                    # 2) Fallback: prosty holdout
                    stratify = y_train if (ptype == "classification" and self._can_stratify(y_train)) else None
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=stratify
                    )
                    est.fit(X_tr, y_tr)

                    if ptype == "classification":
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                        pred = est.predict(X_val)

                        # Wsparcie dla popularnych preferencji
                        pref = (preferred_scoring or "").lower()
                        if "f1" in pref:
                            mean_score = float(f1_score(y_val, pred, average="weighted"))
                        elif "roc_auc" in pref:
                            mean_score = 0.0
                            try:
                                if hasattr(est, "predict_proba"):
                                    proba = est.predict_proba(X_val)
                                    # binary / multiclass
                                    if proba.ndim == 1 or proba.shape[1] == 2:
                                        p1 = proba if proba.ndim == 1 else proba[:, 1]
                                        mean_score = float(roc_auc_score(y_val, p1))
                                    else:
                                        mean_score = float(roc_auc_score(y_val, proba, multi_class="ovr"))
                                elif hasattr(est, "decision_function"):
                                    dfv = est.decision_function(X_val)
                                    if dfv.ndim == 1:
                                        mean_score = float(roc_auc_score(y_val, dfv))
                                    else:
                                        mean_score = float(roc_auc_score(y_val, dfv, multi_class="ovr"))
                                else:
                                    # fallback do accuracy jeśli brak funkcji scoringowej
                                    mean_score = float(accuracy_score(y_val, pred))
                            except Exception:
                                mean_score = float(accuracy_score(y_val, pred))
                        else:
                            mean_score = float(accuracy_score(y_val, pred))
                    else:
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        pred = est.predict(X_val)
                        if scoring in ("neg_mean_absolute_error",):
                            mean_score = float(-mean_absolute_error(y_val, pred))
                        elif scoring in ("neg_root_mean_squared_error",):
                            mean_score = float(-np.sqrt(mean_squared_error(y_val, pred)))
                        else:
                            mean_score = float(r2_score(y_val, pred))
                    std_score = 0.0

                duration = time.time() - start_time
                return {
                    "name": name,
                    "cv_score": mean_score,
                    "cv_std": std_score,
                    "training_time": duration,
                    "status": "completed",
                    "scoring": scoring,
                }
            except Exception as e:
                return {
                    "name": name,
                    "error": str(e),
                    "status": "failed",
                    "cv_score": 0.0,
                    "cv_std": 0.0,
                    "training_time": 0.0,
                    "scoring": scoring,
                }

        # Utwórz taski w executorze
        loop = asyncio.get_running_loop()
        futures: List[asyncio.Future] = []
        for name, model in algorithms.items():
            if truthy_df_safe(progress_callback):
                try:
                    progress_callback(0.0, name, "queued")
                except Exception:
                    pass
            fut = loop.run_in_executor(self.executor, train_single_model, name, model)
            futures.append(fut)

        # Rejestruj task zbiorczy (opcjonalnie do anulowania)
        aggregate_task = asyncio.gather(*futures, return_exceptions=True)
        if truthy_df_safe(task_key):
            with self._lock:
                self.training_tasks[task_key] = aggregate_task  # Future-like

        results: Dict[str, Dict[str, float]] = {}
        completed = 0
        total = len(futures)

        try:
            # Zbieraj wyniki w miarę kończenia (z limitem czasu łącznym)
            for coro in asyncio.as_completed(futures, timeout=total_timeout_sec):
                res = await coro
                if isinstance(res, Exception):
                    continue
                if truthy_df_safe(res):
                    results[res["name"]] = res
                    completed += 1
                    if truthy_df_safe(progress_callback):
                        try:
                            progress_callback(completed / max(total, 1), res["name"], res["status"])
                        except Exception:
                            pass

        except asyncio.TimeoutError:
            try:
                st.warning("⏱️ Część modeli przekroczyła limit czasu — wykorzystuję dostępne wyniki.")
            except Exception:
                pass

        finally:
            if truthy_df_safe(task_key):
                with self._lock:
                    self.training_tasks.pop(task_key, None)

        return results

    # --- zarządzanie zasobami / anulowanie ---
    def cleanup(self):
        """Wyczyść resources (zamknij executor)."""
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # dla starszych wersji Pythona bez cancel_futures
            self.executor.shutdown(wait=False)

    def cancel_task(self, key: str) -> bool:
        """Próba anulowania zarejestrowanego zadania (anuluje Future agregujący)."""
        with self._lock:
            t = self.training_tasks.get(key)
        if truthy_df_safe(t) and hasattr(t, "cancel"):
            try:
                t.cancel()
                with self._lock:
                    self.training_tasks.pop(key, None)
                return True
            except Exception:
                return False
        return False


# =============================================================================
# DECORATOR: run_async_in_streamlit
# =============================================================================
def run_async_in_streamlit(async_func):
    """
    Dekorator do uruchamiania funkcji async w Streamlit.
    - Gdy nie ma działającego event loopa: używa asyncio.run(coro)
    - Gdy jest event loop: odpala NOWĄ pętlę w osobnym wątku i blokująco czeka na wynik
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            try:
                asyncio.get_running_loop()
                running = True
            except RuntimeError:
                running = False

            if not truthy_df_safe(running):
                return asyncio.run(async_func(*args, **kwargs))

            # mamy już loop — uruchom w osobnym wątku na nowej pętli
            result_box: Dict[str, Any] = {}
            err_box: Dict[str, BaseException] = {}

            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(async_func(*args, **kwargs))
                    result_box["value"] = result
                except BaseException as e:
                    err_box["err"] = e
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()

            if "err" in err_box:
                raise err_box["err"]
            return result_box.get("value")

        except Exception as e:
            try:
                st.error(f"Błąd async execution: {e}")
            except Exception:
                pass
            # Ostateczny fallback – oddaj wyjątek żeby nie maskować problemu
            raise
    return wrapper


# =============================================================================
# Async helper z prawdziwym progresem (callback-driven)
# =============================================================================
async def run_with_progress(
    async_func: Callable[..., Any],
    progress_placeholder,
    status_placeholder,
    *args, **kwargs
):
    """
    Uruchamia funkcję async z progress tracking w Streamlit.
    Oczekuje, że async_func przyjmie argument `progress_callback=(p, name, status)->None`.
    """
    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info("🚀 Rozpoczynam zadanie...")

    def _on_update(p: float, name: str, status: str):
        try:
            progress_bar.progress(max(0.0, min(1.0, float(p))))
            status_placeholder.write(f"**{name}** — {status} ({int(p*100)}%)")
        except Exception:
            pass

    try:
        # Wstrzykuj callback do async_func
        kwargs = dict(kwargs)
        if "progress_callback" not in kwargs:
            kwargs["progress_callback"] = _on_update

        result = await async_func(*args, **kwargs)
        progress_bar.progress(1.0)
        status_placeholder.success("✅ Zadanie ukończone!")
        return result

    except Exception as e:
        progress_bar.empty()
        status_placeholder.error(f"❌ Błąd: {e}")
        raise


# =============================================================================
# Streamlit-friendly async utilities
# =============================================================================
class StreamlitAsyncRunner:
    """Pomocnicza klasa do uruchamiania async w Streamlit."""

    @staticmethod
    def run_async(coroutine):
        """Uruchom coroutine w Streamlit (blokująco, ale bez konfliktu z istniejącą pętlą)."""
        try:
            return asyncio.run(coroutine)
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # Uruchom w nowym wątku / nowej pętli
                box: Dict[str, Any] = {}
                err: Dict[str, BaseException] = {}

                def runner():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        box["value"] = loop.run_until_complete(coroutine)
                    except BaseException as ex:
                        err["e"] = ex
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                t = threading.Thread(target=runner, daemon=True)
                t.start()
                t.join()
                if "e" in err:
                    raise err["e"]
                return box.get("value")
            else:
                raise

    @staticmethod
    async def run_with_streamlit_updates(
        async_func: Callable[..., Any],
        update_callback: Optional[Callable[[str, str], None]] = None,
        *args, **kwargs
    ):
        """
        Uruchamia async funkcję z możliwością updatowania Streamlit UI.
        update_callback(stage, message) — stage ∈ {'start','complete','error'}
        """
        start_time = time.time()
        try:
            if truthy_df_safe(update_callback):
                update_callback("start", "🚀 Rozpoczynam zadanie...")
            result = await async_func(*args, **kwargs)
            duration = time.time() - start_time
            if truthy_df_safe(update_callback):
                update_callback("complete", f"✅ Ukończono w {duration:.1f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            if truthy_df_safe(update_callback):
                update_callback("error", f"❌ Błąd po {duration:.1f}s: {e}")
            raise


# =============================================================================
# PRZYKŁAD UŻYCIA (do szybkiego testu lokalnego)
# =============================================================================
if __name__ == "__main__":

    @run_async_in_streamlit
    async def example_async_function():
        await asyncio.sleep(1)
        return "Hello from async!"

    # W Streamlit używasz normalnie:
    # result = example_async_function()
    # st.write(result)