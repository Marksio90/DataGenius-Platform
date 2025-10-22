from __future__ import annotations
# -*- coding: utf-8 -*-

from backend.safe_utils import truthy_df_safe

import json
import pickle
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from config.settings import get_settings

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy base (jeśli kiedyś będziesz definiować ORM-owe modele)
Base = declarative_base()


# ======================================================================
# DatabaseManager
# ======================================================================

class DatabaseManager:
    """
    Klasa do zarządzania połączeniami i operacjami bazodanowymi (PostgreSQL / SQLite).
    Ujednolicona na styl SQLAlchemy 1.4/2.0:
    - zawsze używamy sqlalchemy.text i nazwanych parametrów :param
    - transakcje przez engine.begin()
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Inicjalizuje silnik bazy danych."""
        try:
            database_url = self.settings.get_database_url()
            # Dodatkowe connect_args dla SQLite (jeśli ktoś użyje trybu plikowego)
            connect_args = {}
            if str(database_url).startswith("sqlite"):
                connect_args = {"check_same_thread": False}

            self.engine = create_engine(
                database_url,
                echo=self.settings.debug,
                pool_pre_ping=True,
                pool_recycle=300,
                future=True,  # lepsza kompatybilność ze stylem 2.0
                connect_args=connect_args,
            )
            self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
            logger.info("Połączenie z bazą danych zostało ustanowione")
        except Exception as e:
            logger.error(f"Błąd podczas łączenia z bazą danych: {e}")
            self.engine = None
            self.SessionLocal = None

    @contextmanager
    def get_session(self):
        """Context manager dla sesji bazodanowej."""
        if not self.engine or not self.SessionLocal:
            raise RuntimeError("Brak połączenia z bazą danych (engine/session nie zainicjalizowane)")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @contextmanager
    def get_connection(self):
        """Context manager dla surowego połączenia (read-only / szybkie selecty)."""
        if not self.engine:
            raise RuntimeError("Brak połączenia z bazą danych")
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Wykonuje zapytanie SELECT i zwraca wyniki jako DataFrame.

        Uwaga: używamy sqlalchemy.text i nazwanych parametrów :param
        """
        if not self.engine:
            raise RuntimeError("Brak połączenia z bazą danych")

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params or {})
            return df
        except SQLAlchemyError as e:
            logger.error(f"Błąd SQL podczas wykonywania zapytania: {e}")
            raise
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania zapytania: {e}")
            raise

    def execute_statement(self, statement: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Wykonuje statement (INSERT/UPDATE/DELETE/DDL).
        """
        if not self.engine:
            raise RuntimeError("Brak połączenia z bazą danych")

        try:
            # begin() -> autocommit po wyjściu
            with self.engine.begin() as conn:
                conn.execute(text(statement), params or {})
        except SQLAlchemyError as e:
            logger.error(f"Błąd SQL podczas wykonywania statement: {e}")
            raise
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania statement: {e}")
            raise

    def test_connection(self) -> bool:
        """Szybki test połączenia."""
        try:
            if not self.engine:
                return False
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Test połączenia nieudany: {e}")
            return False

    def dispose(self) -> None:
        """Zwalnia połączenia (np. przy zamknięciu aplikacji)."""
        try:
            if self.engine:
                self.engine.dispose()
        except Exception:
            pass


# ======================================================================
# DatasetRepository
# ======================================================================

class DatasetRepository:
    """Operacje na zbiorach danych (metadane + kolumny)."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_dataset(
        self,
        user_id: int,
        dataset_name: str,
        df: pd.DataFrame,
        original_filename: Optional[str] = None,
        is_sample: bool = False
    ) -> int:
        """
        Zapisuje zbiór danych do bazy wraz z metadanymi i metadanymi kolumn.
        Zwraca dataset_id.
        """
        # Metadane
        rows_count = int(len(df))
        columns_count = int(len(df.columns))
        try:
            missing_values = int(pd.isna(df).to_numpy().sum())
        except Exception:
            missing_values = int(df.isnull().sum().sum())
        try:
            memory_usage = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)  # MB
        except Exception:
            memory_usage = 0.0
        numeric_cols = int(len(df.select_dtypes(include=[np.number]).columns))
        categorical_cols = int(len(df.select_dtypes(include=["object", "category"]).columns))

        data_hash = self._calculate_dataframe_hash(df)

        insert_query = """
        INSERT INTO datasets (
            user_id, dataset_name, original_filename, rows_count, columns_count,
            missing_values_count, memory_usage_mb, numeric_columns_count,
            categorical_columns_count, is_sample_dataset, data_hash,
            dataset_type, processing_status
        ) VALUES (
            :user_id, :dataset_name, :original_filename, :rows_count, :columns_count,
            :missing_values, :memory_usage, :numeric_cols,
            :categorical_cols, :is_sample, :data_hash,
            :dataset_type, 'processed'
        ) RETURNING dataset_id
        """

        params = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "original_filename": original_filename,
            "rows_count": rows_count,
            "columns_count": columns_count,
            "missing_values": missing_values,
            "memory_usage": memory_usage,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "is_sample": bool(is_sample),
            "data_hash": data_hash,
            "dataset_type": "sample" if is_sample else "csv",
        }

        with self.db.get_session() as session:
            result = session.execute(text(insert_query), params)
            dataset_id = int(result.scalar_one())

            self._save_column_metadata(session, dataset_id, df)

        return dataset_id

    def _safe_float(self, x: Any) -> Optional[float]:
        try:
            v = float(x)
            if np.isfinite(v):
                return float(v)
            return None
        except Exception:
            return None

    def _save_column_metadata(self, session, dataset_id: int, df: pd.DataFrame) -> None:
        """Zapisuje metadane każdej kolumny do dataset_columns."""
        insert_column_query = """
        INSERT INTO dataset_columns (
            dataset_id, column_name, column_index, data_type, null_count, null_percentage,
            unique_count, unique_percentage, min_value, max_value, mean_value, median_value,
            std_deviation, most_frequent_value, most_frequent_count
        ) VALUES (
            :dataset_id, :column_name, :column_index, :data_type, :null_count, :null_percentage,
            :unique_count, :unique_percentage, :min_value, :max_value, :mean_value, :median_value,
            :std_deviation, :most_frequent_value, :most_frequent_count
        )
        """

        n_rows = len(df)
        payloads: List[Dict[str, Any]] = []

        for idx, column in enumerate(df.columns):
            col_data = df[column]
            null_count = int(col_data.isna().sum())
            null_pct = float((null_count / n_rows) * 100) if n_rows > 0 else 0.0
            unique_count = int(col_data.nunique(dropna=True))
            unique_pct = float((unique_count / n_rows) * 100) if n_rows > 0 else 0.0

            payload: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "column_name": str(column),
                "column_index": int(idx),
                "data_type": str(col_data.dtype),
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_count": unique_count,
                "unique_percentage": unique_pct,
                "min_value": None,
                "max_value": None,
                "mean_value": None,
                "median_value": None,
                "std_deviation": None,
                "most_frequent_value": None,
                "most_frequent_count": None,
            }

            if pd.api.types.is_numeric_dtype(col_data):
                cc = pd.to_numeric(col_data, errors="coerce").dropna()
                if len(cc) > 0:
                    payload.update({
                        "min_value": self._safe_float(cc.min()),
                        "max_value": self._safe_float(cc.max()),
                        "mean_value": self._safe_float(cc.mean()),
                        "median_value": self._safe_float(cc.median()),
                        "std_deviation": self._safe_float(cc.std()) if len(cc) > 1 else 0.0,
                    })
            else:
                vc = col_data.dropna().astype(str).value_counts()
                if len(vc) > 0:
                    payload.update({
                        "most_frequent_value": str(vc.index[0]),
                        "most_frequent_count": int(vc.iloc[0]),
                    })

            payloads.append(payload)

        # executemany dla wydajności
        if payloads:
            session.execute(text(insert_column_query), payloads)

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Stabilny hash DataFrame (łączy hash kolumn, typów, wartości oraz indeksu).
        Bardziej deterministyczny niż df.to_string().
        """
        try:
            from pandas.util import hash_pandas_object  # type: ignore

            h = hashlib.sha256()
            # kolumny + dtypes
            h.update("|".join(map(str, df.columns)).encode("utf-8"))
            h.update("|".join(df.dtypes.astype(str).tolist()).encode("utf-8"))
            # wartości (z indeksem)
            values_hash = hash_pandas_object(df, index=True).values
            h.update(values_hash.tobytes())
            return h.hexdigest()
        except Exception:
            # fallback deterministyczny
            try:
                s = df.to_csv(index=True)  # deterministyczniej niż to_string
                return hashlib.sha256(s.encode("utf-8")).hexdigest()
            except Exception:
                return "unknown"

    def get_datasets_by_user(self, user_id: int) -> pd.DataFrame:
        """Pobiera wszystkie zbiory danych użytkownika (meta)."""
        query = """
        SELECT dataset_id, dataset_name, rows_count, columns_count,
               missing_values_count, memory_usage_mb, created_at,
               is_sample_dataset, processing_status
        FROM datasets
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        """
        return self.db.execute_query(query, {"user_id": user_id})

    def get_dataset_details(self, dataset_id: int) -> Dict[str, Any]:
        """Pobiera metadane zbioru + metadane kolumn."""
        dataset_query = "SELECT * FROM datasets WHERE dataset_id = :dataset_id"
        dataset_info = self.db.execute_query(dataset_query, {"dataset_id": dataset_id})
        if dataset_info.empty:
            raise ValueError(f"Zbiór danych o ID {dataset_id} nie istnieje")

        columns_query = """
        SELECT * FROM dataset_columns
        WHERE dataset_id = :dataset_id
        ORDER BY column_index
        """
        columns_info = self.db.execute_query(columns_query, {"dataset_id": dataset_id})

        return {
            "dataset_info": dataset_info.iloc[0].to_dict(),
            "columns_info": columns_info.to_dict("records"),
        }


# ======================================================================
# ModelRepository
# ======================================================================

class ModelRepository:
    """Operacje na modelach ML i ich metrykach / feature importance."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_model(
        self,
        user_id: int,
        dataset_id: int,
        model_config: Dict[str, Any],
        model_results: Dict[str, Any],
        model_object: Any = None
    ) -> int:
        """
        Zapisuje wytrenowany model, metryki i (opcjonalnie) feature importance.
        Zwraca model_id.
        """
        # Serializacja modelu (opcjonalnie)
        model_binary = None
        if model_object is not None:
            try:
                model_binary = pickle.dumps(model_object, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"Nie udało się zserializować modelu: {e}")

        training_duration = float(model_config.get("training_duration", 0.0))
        training_start = datetime.now() - timedelta(seconds=training_duration)
        training_end = datetime.now()

        insert_model_query = """
        INSERT INTO ml_models (
            user_id, dataset_id, model_name, target_column, problem_type,
            algorithm_name, train_test_split, cv_folds, random_state,
            training_status, feature_count, training_samples_count,
            test_samples_count, model_binary, training_start_time,
            training_end_time, training_duration_seconds
        ) VALUES (
            :user_id, :dataset_id, :model_name, :target_column, :problem_type,
            :algorithm_name, :train_test_split, :cv_folds, :random_state,
            'completed', :feature_count, :training_samples,
            :test_samples, :model_binary, :training_start,
            :training_end, :training_duration
        ) RETURNING model_id
        """

        # problem_type normalizacja
        pt_raw = str(model_config.get("problem_type", "")).lower()
        if pt_raw.startswith(("regresja", "regression", "reg")):
            problem_type_db = "regression"
        else:
            problem_type_db = "classification"

        model_params = {
            "user_id": user_id,
            "dataset_id": dataset_id,
            "model_name": f"{model_config.get('algorithm_name','model')}_{model_config.get('target_column','target')}",
            "target_column": model_config.get("target_column"),
            "problem_type": problem_type_db,
            "algorithm_name": model_config.get("algorithm_name"),
            "train_test_split": float(model_config.get("train_test_split", 0.8)),
            "cv_folds": int(model_config.get("cv_folds", 5)),
            "random_state": int(model_config.get("random_state", 42)),
            "feature_count": int(len(model_results.get("feature_names", []))),
            "training_samples": int(model_config.get("training_samples", 0)),
            "test_samples": int(model_config.get("test_samples", 0)),
            "model_binary": model_binary,
            "training_start": training_start,
            "training_end": training_end,
            "training_duration": training_duration,
        }

        with self.db.get_session() as session:
            result = session.execute(text(insert_model_query), model_params)
            model_id = int(result.scalar_one())

            self._save_model_metrics(session, model_id, model_results, problem_type_db)
            if "feature_importance" in model_results and isinstance(model_results["feature_importance"], pd.DataFrame):
                self._save_feature_importance(session, model_id, model_results["feature_importance"])

        return model_id

    def _save_model_metrics(self, session, model_id: int, results: Dict[str, Any], problem_type_db: str) -> None:
        """Zapis metryk modelu do model_metrics."""
        metrics_data: Dict[str, Any] = {
            "model_id": model_id,
            "r2_score": None,
            "mae": None,
            "mse": None,
            "rmse": None,
            "accuracy": None,
            "precision_score": None,
            "recall_score": None,
            "f1_score": None,
            "cv_score_mean": None,
            "cv_score_std": None,
        }

        if problem_type_db == "regression":
            metrics_data.update({
                "r2_score": results.get("r2"),
                "mae": results.get("mae"),
                "mse": results.get("mse"),
                "rmse": results.get("rmse"),
            })
        else:
            metrics_data.update({
                "accuracy": results.get("accuracy"),
                "precision_score": results.get("precision"),
                "recall_score": results.get("recall"),
                "f1_score": results.get("f1"),
            })

        # CV metrics (jeśli są)
        model_scores = results.get("model_scores", {}) or {}
        best_model = results.get("best_model", "")
        if truthy_df_safe(best_model) and best_model in model_scores:
            metrics_data.update({
                "cv_score_mean": model_scores[best_model].get("cv_score"),
                "cv_score_std": model_scores[best_model].get("cv_std"),
            })

        insert_metrics_query = """
        INSERT INTO model_metrics (
            model_id, r2_score, mae, mse, rmse, accuracy, precision_score,
            recall_score, f1_score, cv_score_mean, cv_score_std
        ) VALUES (
            :model_id, :r2_score, :mae, :mse, :rmse, :accuracy, :precision_score,
            :recall_score, :f1_score, :cv_score_mean, :cv_score_std
        )
        """
        session.execute(text(insert_metrics_query), metrics_data)

    def _save_feature_importance(self, session, model_id: int, feature_importance_df: pd.DataFrame) -> None:
        """Zapisuje ważność cech z DataFrame['feature','importance'].""" 
        if feature_importance_df.empty or "feature" not in feature_importance_df or "importance" not in feature_importance_df:
            return

        # Sumuj bezwzględne wartości, aby uniknąć dzielenia przez ~0 dla dodatnich i ujemnych SHAP
        try:
            total_importance = float(np.abs(feature_importance_df["importance"]).sum())
        except Exception:
            s = feature_importance_df["importance"].sum()
            total_importance = float(s) if s is not None else 0.0

        insert_importance_query = """
        INSERT INTO feature_importance (
            model_id, feature_name, importance_value, importance_rank, importance_percentage
        ) VALUES (
            :model_id, :feature_name, :importance_value, :importance_rank, :importance_percentage
        )
        """

        payloads: List[Dict[str, Any]] = []
        for idx, row in feature_importance_df.reset_index(drop=True).iterrows():
            try:
                importance_val = float(row["importance"])
            except Exception:
                try:
                    importance_val = float(str(row["importance"]))
                except Exception:
                    importance_val = 0.0

            pct = (abs(importance_val) / total_importance) * 100.0 if total_importance > 0 else 0.0
            payloads.append({
                "model_id": model_id,
                "feature_name": str(row["feature"]),
                "importance_value": importance_val,
                "importance_rank": int(idx + 1),
                "importance_percentage": float(pct),
            })

        if payloads:
            session.execute(text(insert_importance_query), payloads)

    def get_user_models(self, user_id: int) -> pd.DataFrame:
        """Lista modeli użytkownika (skrót)."""
        query = """
        SELECT 
            m.model_id,
            m.model_name,
            d.dataset_name,
            m.target_column,
            m.problem_type,
            m.algorithm_name,
            m.training_status,
            mt.accuracy,
            mt.r2_score,
            m.created_at
        FROM ml_models m
        LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.user_id = :user_id
        ORDER BY m.created_at DESC
        """
        return self.db.execute_query(query, {"user_id": user_id})

    def get_model_details(self, model_id: int) -> Dict[str, Any]:
        """Szczegóły modelu + feature importance."""
        model_query = """
        SELECT m.*, d.dataset_name, mt.*
        FROM ml_models m
        LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.model_id = :model_id
        """
        model_info = self.db.execute_query(model_query, {"model_id": model_id})
        if model_info.empty:
            raise ValueError(f"Model o ID {model_id} nie istnieje")

        importance_query = """
        SELECT feature_name, importance_value, importance_rank, importance_percentage
        FROM feature_importance
        WHERE model_id = :model_id
        ORDER BY importance_rank
        """
        feature_importance = self.db.execute_query(importance_query, {"model_id": model_id})

        return {
            "model_info": model_info.iloc[0].to_dict(),
            "feature_importance": feature_importance.to_dict("records"),
        }

    def load_model_object(self, model_id: int) -> Any:
        """Ładuje zserializowany obiekt modelu (pickle) z kolumny model_binary."""
        query = "SELECT model_binary FROM ml_models WHERE model_id = :model_id"
        result = self.db.execute_query(query, {"model_id": model_id})
        if result.empty or result.iloc[0]["model_binary"] is None:
            raise ValueError(f"Nie znaleziono zserializowanego modelu dla ID {model_id}")

        try:
            model_binary = result.iloc[0]["model_binary"]
            return pickle.loads(model_binary)
        except Exception as e:
            raise ValueError(f"Błąd podczas deserializacji modelu: {e}")


# ======================================================================
# AnalyticsRepository
# ======================================================================

class AnalyticsRepository:
    """Zapytania statystyczne (użytkownik/global)."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Statystyki aktywności użytkownika."""
        datasets_count_query = "SELECT COUNT(*) AS count FROM datasets WHERE user_id = :user_id"
        models_count_query = "SELECT COUNT(*) AS count FROM ml_models WHERE user_id = :user_id"

        datasets_count = self.db.execute_query(datasets_count_query, {"user_id": user_id})
        models_count = self.db.execute_query(models_count_query, {"user_id": user_id})

        best_models_query = """
        SELECT 
            m.model_name,
            m.algorithm_name,
            COALESCE(mt.accuracy, mt.r2_score) AS score,
            m.problem_type
        FROM ml_models m
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.user_id = :user_id AND m.training_status = 'completed'
        ORDER BY COALESCE(mt.accuracy, mt.r2_score) DESC
        LIMIT 5
        """
        best_models = self.db.execute_query(best_models_query, {"user_id": user_id})

        algorithms_query = """
        SELECT 
            algorithm_name,
            COUNT(*) AS usage_count
        FROM ml_models
        WHERE user_id = :user_id
        GROUP BY algorithm_name
        ORDER BY usage_count DESC
        LIMIT 5
        """
        popular_algorithms = self.db.execute_query(algorithms_query, {"user_id": user_id})

        return {
            "datasets_count": int(datasets_count.iloc[0]["count"]),
            "models_count": int(models_count.iloc[0]["count"]),
            "best_models": best_models.to_dict("records"),
            "popular_algorithms": popular_algorithms.to_dict("records"),
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Statystyki globalne (bez parametrów)."""
        total_users_query = "SELECT COUNT(*) AS count FROM users WHERE is_active = TRUE"
        total_datasets_query = "SELECT COUNT(*) AS count FROM datasets"
        total_models_query = "SELECT COUNT(*) AS count FROM ml_models WHERE training_status = 'completed'"

        total_users = self.db.execute_query(total_users_query)
        total_datasets = self.db.execute_query(total_datasets_query)
        total_models = self.db.execute_query(total_models_query)

        popular_algorithms_query = """
        SELECT 
            m.algorithm_name,
            COUNT(*) AS usage_count,
            AVG(COALESCE(mt.accuracy, mt.r2_score)) AS avg_score
        FROM ml_models m
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.training_status = 'completed'
        GROUP BY m.algorithm_name
        ORDER BY usage_count DESC
        LIMIT 10
        """
        popular_algorithms = self.db.execute_query(popular_algorithms_query)

        problem_types_query = """
        SELECT 
            problem_type,
            COUNT(*) AS count
        FROM ml_models
        WHERE training_status = 'completed'
        GROUP BY problem_type
        """
        problem_types = self.db.execute_query(problem_types_query)

        return {
            "total_users": int(total_users.iloc[0]["count"]),
            "total_datasets": int(total_datasets.iloc[0]["count"]),
            "total_models": int(total_models.iloc[0]["count"]),
            "popular_algorithms": popular_algorithms.to_dict("records"),
            "problem_types_distribution": problem_types.to_dict("records"),
        }


# ======================================================================
# DatabaseUtils
# ======================================================================

class DatabaseUtils:
    """Narzędzia administracyjne bazy (DDL/cleanup/backup)."""

    @staticmethod
    def create_tables(db_manager: DatabaseManager) -> None:
        """Tworzy wszystkie tabele z pliku db/schema.sql (prosty runner)."""
        if not db_manager.engine:
            raise RuntimeError("Brak połączenia z bazą danych")

        try:
            with open("db/schema.sql", "r", encoding="utf-8") as f:
                schema_sql = f.read()

            statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]

            with db_manager.engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))

            logger.info("Tabele zostały utworzone pomyślnie")
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia tabel: {e}")
            raise

    @staticmethod
    def cleanup_old_data(db_manager: DatabaseManager, days_old: int = 30) -> None:
        """Czyści stare dane (bezpiecznie parametryzowane, Postgres make_interval)."""
        if not db_manager.engine:
            raise RuntimeError("Brak połączenia z bazą danych")

        queries = [
            text("""
                DELETE FROM user_sessions
                WHERE expires_at < CURRENT_TIMESTAMP - make_interval(days => :days)
            """),
            text("""
                DELETE FROM activity_logs
                WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => :days)
            """),
            text("""
                DELETE FROM ml_models
                WHERE training_status IN ('pending', 'training', 'failed')
                  AND created_at < CURRENT_TIMESTAMP - make_interval(days => :days)
            """),
        ]

        try:
            with db_manager.engine.begin() as conn:
                for q in queries:
                    result = conn.execute(q, {"days": int(days_old)})
                    try:
                        logger.info(f"Usunięto {result.rowcount} rekordów")
                    except Exception:
                        pass
            logger.info(f"Oczyszczono dane starsze niż {days_old} dni")
        except Exception as e:
            logger.error(f"Błąd podczas czyszczenia danych: {e}")
            raise

    @staticmethod
    def backup_database(db_manager: DatabaseManager, backup_path: str) -> None:
        """
        Tworzy backup bazy danych przez pg_dump.
        Bardziej niezawodny parsing URL-a przez sqlalchemy.engine.make_url.
        """
        import os
        import subprocess
        from sqlalchemy.engine import make_url

        if not db_manager.engine:
            raise RuntimeError("Brak połączenia z bazą danych")

        try:
            url = make_url(db_manager.settings.get_database_url())

            user = url.username or ""
            password = url.password or ""
            host = url.host or "localhost"
            port = str(url.port or 5432)
            database = url.database or ""

            env = os.environ.copy()
            if truthy_df_safe(password):
                env["PGPASSWORD"] = password

            cmd = [
                "pg_dump",
                f"--host={host}",
                f"--port={port}",
                f"--username={user}",
                f"--dbname={database}",
                "--verbose",
                "--clean",
                "--no-owner",
                "--no-privileges",
                f"--file={backup_path}",
            ]

            subprocess.run(cmd, env=env, check=True)
            logger.info(f"Backup utworzony: {backup_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Błąd podczas tworzenia backupu: {e}")
            raise
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas backupu: {e}")
            raise


# ======================================================================
# Singletons (lazy)
# ======================================================================

_db_manager: Optional[DatabaseManager] = None
_dataset_repo: Optional[DatasetRepository] = None
_model_repo: Optional[ModelRepository] = None
_analytics_repo: Optional[AnalyticsRepository] = None


def get_database_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_dataset_repository() -> DatasetRepository:
    global _dataset_repo
    if _dataset_repo is None:
        _dataset_repo = DatasetRepository(get_database_manager())
    return _dataset_repo


def get_model_repository() -> ModelRepository:
    global _model_repo
    if _model_repo is None:
        _model_repo = ModelRepository(get_database_manager())
    return _model_repo


def get_analytics_repository() -> AnalyticsRepository:
    global _analytics_repo
    if _analytics_repo is None:
        _analytics_repo = AnalyticsRepository(get_database_manager())
    return _analytics_repo
