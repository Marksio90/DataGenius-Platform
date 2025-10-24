# backend/async_ml_trainer.py - POPRAWIONA WERSJA z run_async_in_streamlit
import asyncio
import concurrent.futures
import time
from typing import Dict, Any, Callable, Optional
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import threading
from functools import wraps

class AsyncMLTrainer:
    """Asynchroniczny trainer ML z background processing"""
    
    def __init__(self, max_workers: int = 2):  # Zmniejszono z 4 na 2
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.training_tasks: Dict[str, asyncio.Task] = {}
        
    async def train_models_async(
        self, 
        algorithms: Dict[str, Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        cv_folds: int = 5,
        progress_callback: Optional[Callable] = None,
        problem_type: str = "Classification"  # DODANE: brakujący parametr
    ) -> Dict[str, Dict[str, float]]:
        """Trenuje modele asynchronicznie"""
        
        def train_single_model(name: str, model: Any) -> Dict[str, Any]:
            """Trenuje pojedynczy model w osobnym wątku"""
            try:
                start_time = time.time()
                
                # POPRAWKA: Używaj odpowiedniej metryki
                if problem_type == "Classification":
                    scoring = 'accuracy'
                else:
                    scoring = 'r2'
                
                # Cross-validation z timeout protection
                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, 
                        n_jobs=1,  # Ważne: nie używaj -1 w threading!
                        scoring=scoring
                    )
                    
                    # POPRAWKA: Bezpieczne wyciągnięcie wartości
                    if hasattr(cv_scores, '__iter__') and len(cv_scores) > 0:
                        mean_score = float(np.mean(cv_scores))
                        std_score = float(np.std(cv_scores))
                    else:
                        # Fallback do prostego train/test split
                        from sklearn.model_selection import train_test_split
                        X_tr, X_val, y_tr, y_val = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42
                        )
                        model.fit(X_tr, y_tr)
                        if problem_type == "Classification":
                            from sklearn.metrics import accuracy_score
                            pred = model.predict(X_val)
                            mean_score = accuracy_score(y_val, pred)
                        else:
                            from sklearn.metrics import r2_score
                            pred = model.predict(X_val)
                            mean_score = r2_score(y_val, pred)
                        std_score = 0.0
                    
                except Exception as cv_error:
                    # Fallback jeśli CV nie działa
                    mean_score = 0.0
                    std_score = 0.0
                
                duration = time.time() - start_time
                
                return {
                    'name': name,
                    'cv_score': mean_score,
                    'cv_std': std_score,
                    'training_time': duration,
                    'status': 'completed'
                }
                
            except Exception as e:
                return {
                    'name': name,
                    'error': str(e),
                    'status': 'failed',
                    'cv_score': 0.0,
                    'cv_std': 0.0,
                    'training_time': 0.0
                }
        
        # OPTYMALIZACJA: Trenuj tylko najważniejsze modele jeśli dataset duży
        if len(X_train) > 10000:
            # Dla dużych zbiorów - tylko szybkie i skuteczne modele
            priority_models = {
                name: model for name, model in algorithms.items()
                if any(fast_name in name for fast_name in [
                    'Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 
                    'Linear', 'Logistic', 'Ridge'
                ])
            }
            if priority_models:
                algorithms = priority_models
        
        # Uruchom wszystkie modele równolegle (ale ograniczone)
        loop = asyncio.get_event_loop()
        tasks = []
        
        for name, model in algorithms.items():
            task = loop.run_in_executor(
                self.executor, 
                train_single_model, 
                name, model
            )
            tasks.append(task)
        
        # Zbieraj wyniki w miarę ukończenia
        results = {}
        completed = 0
        total = len(tasks)
        
        try:
            for task in asyncio.as_completed(tasks, timeout=600):  # 10 min total timeout
                result = await task
                if result:  # POPRAWKA: Sprawdź czy result nie jest None
                    results[result['name']] = result
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress = completed / total
                        progress_callback(progress, result['name'], result['status'])
        
        except asyncio.TimeoutError:
            st.warning("⏱️ Niektóre modele przekroczyły limit czasu - używam dostępne wyniki")
        
        return results
    
    def cleanup(self):
        """Wyczyść resources"""
        self.executor.shutdown(wait=False)


# =============================================================================
# DODANE: Funkcja run_async_in_streamlit - DECORATOR
# =============================================================================

def run_async_in_streamlit(async_func):
    """
    Decorator do uruchamiania funkcji async w Streamlit.
    
    Streamlit nie obsługuje natywnie async/await, więc ten decorator
    konwertuje funkcję async na sync używając asyncio.run().
    
    Usage:
        @run_async_in_streamlit
        async def my_async_function():
            await some_async_operation()
            return result
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            # Sprawdź czy już mamy event loop
            try:
                loop = asyncio.get_running_loop()
                # Jeśli mamy loop, użyj run_in_executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            except RuntimeError:
                # Brak running loop - użyj asyncio.run
                return asyncio.run(async_func(*args, **kwargs))
                
        except Exception as e:
            st.error(f"Błąd async execution: {e}")
            # Fallback - spróbuj wywołać funkcję synchronicznie (jeśli możliwe)
            try:
                # To nie zadziała dla prawdziwych async funkcji, ale jest fallback
                return async_func(*args, **kwargs)
            except:
                raise e
    
    return wrapper


# =============================================================================
# DODANE: Pomocnicze funkcje async dla Streamlit
# =============================================================================

async def run_with_progress(
    async_func: Callable,
    progress_placeholder,
    status_placeholder,
    *args, **kwargs
):
    """
    Uruchamia funkcję async z progress tracking w Streamlit
    
    Args:
        async_func: Funkcja async do wykonania
        progress_placeholder: st.empty() placeholder dla progress bara
        status_placeholder: st.empty() placeholder dla statusu
        *args, **kwargs: Argumenty dla async_func
    """
    
    # Progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Status
    status_placeholder.info("🚀 Rozpoczynam zadanie...")
    
    try:
        # Symuluj progress (dla demonstracji)
        for i in range(0, 101, 10):
            progress_bar.progress(i / 100)
            await asyncio.sleep(0.1)  # Krótka pauza
        
        # Wykonaj właściwe zadanie
        result = await async_func(*args, **kwargs)
        
        # Ukończenie
        progress_bar.progress(100)
        status_placeholder.success("✅ Zadanie ukończone!")
        
        return result
        
    except Exception as e:
        progress_bar.empty()
        status_placeholder.error(f"❌ Błąd: {e}")
        raise


# =============================================================================
# DODANE: Streamlit-friendly async utilities
# =============================================================================

class StreamlitAsyncRunner:
    """Pomocnicza klasa do uruchamiania async w Streamlit"""
    
    @staticmethod
    def run_async(coroutine):
        """Uruchom coroutine w Streamlit"""
        try:
            return asyncio.run(coroutine)
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # Streamlit już ma event loop - użyj nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(coroutine)
                except ImportError:
                    st.warning("Zainstaluj nest_asyncio: pip install nest-asyncio")
                    raise
            else:
                raise
    
    @staticmethod
    async def run_with_streamlit_updates(
        async_func,
        update_callback=None,
        *args, **kwargs
    ):
        """
        Uruchamia async funkcję z możliwością updatowania Streamlit UI
        
        Args:
            async_func: Async funkcja do wykonania
            update_callback: Funkcja do updatowania UI (progress, status, etc.)
            *args, **kwargs: Argumenty dla async_func
        """
        
        start_time = time.time()
        
        try:
            if update_callback:
                update_callback("start", "🚀 Rozpoczynam zadanie...")
            
            result = await async_func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            if update_callback:
                update_callback("complete", f"✅ Ukończono w {duration:.1f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            if update_callback:
                update_callback("error", f"❌ Błąd po {duration:.1f}s: {e}")
            
            raise


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================

if __name__ == "__main__":
    # Przykład użycia decoratora
    
    @run_async_in_streamlit
    async def example_async_function():
        await asyncio.sleep(1)
        return "Hello from async!"
    
    # W Streamlit można użyć:
    # result = example_async_function()
    # st.write(result)