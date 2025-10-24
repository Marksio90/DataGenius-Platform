import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
import hashlib
import pickle
from contextlib import contextmanager

from config.settings import get_settings

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

class DatabaseManager:
    """
    Klasa do zarządzania połączeniami i operacjami bazodanowymi
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Inicjalizuje silnik bazy danych"""
        try:
            database_url = self.settings.get_database_url()
            self.engine = create_engine(
                database_url,
                echo=self.settings.debug,
                pool_pre_ping=True,
                pool_recycle=300
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Połączenie z bazą danych zostało ustanowione")
        except Exception as e:
            logger.error(f"Błąd podczas łączenia z bazą danych: {e}")
            self.engine = None
    
    @contextmanager
    def get_session(self):
        """Context manager dla sesji bazodanowej"""
        if not self.engine:
            raise Exception("Brak połączenia z bazą danych")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """
        Wykonuje zapytanie SQL i zwraca wyniki jako DataFrame
        
        Args:
            query: Zapytanie SQL
            params: Parametry zapytania
            
        Returns:
            DataFrame z wynikami
        """
        if not self.engine:
            raise Exception("Brak połączenia z bazą danych")
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params=params)
            return result
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania zapytania: {e}")
            raise
    
    def execute_statement(self, statement: str, params: Dict = None) -> None:
        """
        Wykonuje statement SQL (INSERT, UPDATE, DELETE)
        
        Args:
            statement: Statement SQL
            params: Parametry statement
        """
        if not self.engine:
            raise Exception("Brak połączenia z bazą danych")
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(statement), params or {})
                conn.commit()
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania statement: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Testuje połączenie z bazą danych"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Test połączenia nieudany: {e}")
            return False

class DatasetRepository:
    """
    Klasa do operacji na zbiorach danych w bazie
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_dataset(self, user_id: int, dataset_name: str, df: pd.DataFrame, 
                    original_filename: str = None, is_sample: bool = False) -> int:
        """
        Zapisuje zbiór danych do bazy
        
        Args:
            user_id: ID użytkownika
            dataset_name: Nazwa zbioru
            df: DataFrame z danymi
            original_filename: Oryginalna nazwa pliku
            is_sample: Czy to przykładowy zbiór
            
        Returns:
            ID zapisanego zbioru danych
        """
        
        # Obliczanie metadanych
        rows_count = len(df)
        columns_count = len(df.columns)
        missing_values = df.isnull().sum().sum()
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Hash danych
        data_hash = self._calculate_dataframe_hash(df)
        
        # Wstawienie do tabeli datasets
        insert_query = """
        INSERT INTO datasets (
            user_id, dataset_name, original_filename, rows_count, columns_count,
            missing_values_count, memory_usage_mb, numeric_columns_count,
            categorical_columns_count, is_sample_dataset, data_hash,
            dataset_type, processing_status
        ) VALUES (
            %(user_id)s, %(dataset_name)s, %(original_filename)s, %(rows_count)s, %(columns_count)s,
            %(missing_values)s, %(memory_usage)s, %(numeric_cols)s,
            %(categorical_cols)s, %(is_sample)s, %(data_hash)s,
            %(dataset_type)s, 'processed'
        ) RETURNING dataset_id
        """
        
        params = {
            'user_id': user_id,
            'dataset_name': dataset_name,
            'original_filename': original_filename,
            'rows_count': rows_count,
            'columns_count': columns_count,
            'missing_values': missing_values,
            'memory_usage': memory_usage,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'is_sample': is_sample,
            'data_hash': data_hash,
            'dataset_type': 'sample' if is_sample else 'csv'
        }
        
        with self.db.get_session() as session:
            result = session.execute(text(insert_query), params)
            dataset_id = result.fetchone()[0]
            
            # Zapisanie metadanych kolumn
            self._save_column_metadata(session, dataset_id, df)
            
        return dataset_id
    
    def _save_column_metadata(self, session, dataset_id: int, df: pd.DataFrame):
        """Zapisuje metadane kolumn"""
        
        for idx, column in enumerate(df.columns):
            col_data = df[column]
            
            # Podstawowe statystyki
            null_count = col_data.isnull().sum()
            null_pct = (null_count / len(col_data)) * 100 if len(col_data) > 0 else 0
            unique_count = col_data.nunique()
            unique_pct = (unique_count / len(col_data)) * 100 if len(col_data) > 0 else 0
            
            # Przygotowanie danych do wstawienia
            column_data = {
                'dataset_id': dataset_id,
                'column_name': column,
                'column_index': idx,
                'data_type': str(col_data.dtype),
                'null_count': int(null_count),
                'null_percentage': float(null_pct),
                'unique_count': int(unique_count),
                'unique_percentage': float(unique_pct)
            }
            
            # Statystyki dla kolumn numerycznych
            if pd.api.types.is_numeric_dtype(col_data):
                clean_data = col_data.dropna()
                if len(clean_data) > 0:
                    column_data.update({
                        'min_value': float(clean_data.min()),
                        'max_value': float(clean_data.max()),
                        'mean_value': float(clean_data.mean()),
                        'median_value': float(clean_data.median()),
                        'std_deviation': float(clean_data.std()) if len(clean_data) > 1 else 0
                    })
            
            # Statystyki dla kolumn kategorycznych
            else:
                value_counts = col_data.value_counts()
                if len(value_counts) > 0:
                    column_data.update({
                        'most_frequent_value': str(value_counts.index[0]),
                        'most_frequent_count': int(value_counts.iloc[0])
                    })
            
            # Wstawienie kolumny
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
            
            session.execute(text(insert_column_query), column_data)
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Oblicza hash DataFrame dla kontroli integralności"""
        try:
            # Konwersja DataFrame do stringa i obliczenie hash
            df_string = df.to_string()
            return hashlib.sha256(df_string.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def get_datasets_by_user(self, user_id: int) -> pd.DataFrame:
        """Pobiera wszystkie zbiory danych użytkownika"""
        query = """
        SELECT dataset_id, dataset_name, rows_count, columns_count, 
               missing_values_count, memory_usage_mb, created_at,
               is_sample_dataset, processing_status
        FROM datasets 
        WHERE user_id = %(user_id)s
        ORDER BY created_at DESC
        """
        return self.db.execute_query(query, {'user_id': user_id})
    
    def get_dataset_details(self, dataset_id: int) -> Dict[str, Any]:
        """Pobiera szczegóły zbioru danych"""
        # Podstawowe informacje
        dataset_query = """
        SELECT * FROM datasets WHERE dataset_id = %(dataset_id)s
        """
        dataset_info = self.db.execute_query(dataset_query, {'dataset_id': dataset_id})
        
        if dataset_info.empty:
            raise ValueError(f"Zbiór danych o ID {dataset_id} nie istnieje")
        
        # Informacje o kolumnach
        columns_query = """
        SELECT * FROM dataset_columns 
        WHERE dataset_id = %(dataset_id)s
        ORDER BY column_index
        """
        columns_info = self.db.execute_query(columns_query, {'dataset_id': dataset_id})
        
        return {
            'dataset_info': dataset_info.iloc[0].to_dict(),
            'columns_info': columns_info.to_dict('records')
        }

class ModelRepository:
    """
    Klasa do operacji na modelach ML w bazie
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_model(self, user_id: int, dataset_id: int, model_config: Dict[str, Any],
                  model_results: Dict[str, Any], model_object: Any = None) -> int:
        """
        Zapisuje wytrenowany model do bazy
        
        Args:
            user_id: ID użytkownika
            dataset_id: ID zbioru danych
            model_config: Konfiguracja modelu
            model_results: Wyniki trenowania
            model_object: Obiekt modelu do serializacji
            
        Returns:
            ID zapisanego modelu
        """
        
        # Serializacja modelu (opcjonalnie)
        model_binary = None
        if model_object:
            try:
                model_binary = pickle.dumps(model_object)
            except Exception as e:
                logger.warning(f"Nie udało się zserializować modelu: {e}")
        
        # Wstawienie modelu
        insert_model_query = """
        INSERT INTO ml_models (
            user_id, dataset_id, model_name, target_column, problem_type,
            algorithm_name, train_test_split, cv_folds, random_state,
            training_status, feature_count, training_samples_count,
            test_samples_count, model_binary, training_start_time,
            training_end_time, training_duration_seconds
        ) VALUES (
            %(user_id)s, %(dataset_id)s, %(model_name)s, %(target_column)s, %(problem_type)s,
            %(algorithm_name)s, %(train_test_split)s, %(cv_folds)s, %(random_state)s,
            'completed', %(feature_count)s, %(training_samples)s,
            %(test_samples)s, %(model_binary)s, %(training_start)s,
            %(training_end)s, %(training_duration)s
        ) RETURNING model_id
        """
        
        training_start = datetime.now() - timedelta(seconds=model_config.get('training_duration', 0))
        training_end = datetime.now()
        
        model_params = {
            'user_id': user_id,
            'dataset_id': dataset_id,
            'model_name': f"{model_config['algorithm_name']}_{model_config['target_column']}",
            'target_column': model_config['target_column'],
            'problem_type': 'regression' if model_config['problem_type'] == 'Regresja' else 'classification',
            'algorithm_name': model_config['algorithm_name'],
            'train_test_split': model_config.get('train_test_split', 0.8),
            'cv_folds': model_config.get('cv_folds', 5),
            'random_state': model_config.get('random_state', 42),
            'feature_count': len(model_results.get('feature_names', [])),
            'training_samples': model_config.get('training_samples', 0),
            'test_samples': model_config.get('test_samples', 0),
            'model_binary': model_binary,
            'training_start': training_start,
            'training_end': training_end,
            'training_duration': model_config.get('training_duration', 0)
        }
        
        with self.db.get_session() as session:
            result = session.execute(text(insert_model_query), model_params)
            model_id = result.fetchone()[0]
            
            # Zapisanie metryk
            self._save_model_metrics(session, model_id, model_results, model_config['problem_type'])
            
            # Zapisanie feature importance
            if 'feature_importance' in model_results:
                self._save_feature_importance(session, model_id, model_results['feature_importance'])
        
        return model_id
    
    def _save_model_metrics(self, session, model_id: int, results: Dict[str, Any], problem_type: str):
        """Zapisuje metryki modelu"""
        
        metrics_data = {'model_id': model_id}
        
        if problem_type == 'Regresja':
            metrics_data.update({
                'r2_score': results.get('r2'),
                'mae': results.get('mae'),
                'mse': results.get('mse'),
                'rmse': results.get('rmse')
            })
        else:
            metrics_data.update({
                'accuracy': results.get('accuracy'),
                'precision_score': results.get('precision'),
                'recall_score': results.get('recall'),
                'f1_score': results.get('f1')
            })
        
        # CV scores
        model_scores = results.get('model_scores', {})
        best_model = results.get('best_model', '')
        if best_model and best_model in model_scores:
            metrics_data.update({
                'cv_score_mean': model_scores[best_model].get('cv_score'),
                'cv_score_std': model_scores[best_model].get('cv_std')
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
    
    def _save_feature_importance(self, session, model_id: int, feature_importance_df: pd.DataFrame):
        """Zapisuje ważność cech"""
        
        total_importance = feature_importance_df['importance'].sum()
        
        for idx, row in feature_importance_df.iterrows():
            importance_data = {
                'model_id': model_id,
                'feature_name': row['feature'],
                'importance_value': row['importance'],
                'importance_rank': idx + 1,
                'importance_percentage': (row['importance'] / total_importance) * 100 if total_importance > 0 else 0
            }
            
            insert_importance_query = """
            INSERT INTO feature_importance (
                model_id, feature_name, importance_value, importance_rank, importance_percentage
            ) VALUES (
                :model_id, :feature_name, :importance_value, :importance_rank, :importance_percentage
            )
            """
            
            session.execute(text(insert_importance_query), importance_data)
    
    def get_user_models(self, user_id: int) -> pd.DataFrame:
        """Pobiera wszystkie modele użytkownika"""
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
        WHERE m.user_id = %(user_id)s
        ORDER BY m.created_at DESC
        """
        return self.db.execute_query(query, {'user_id': user_id})
    
    def get_model_details(self, model_id: int) -> Dict[str, Any]:
        """Pobiera szczegóły modelu"""
        # Podstawowe informacje o modelu
        model_query = """
        SELECT m.*, d.dataset_name, mt.*
        FROM ml_models m
        LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.model_id = %(model_id)s
        """
        model_info = self.db.execute_query(model_query, {'model_id': model_id})
        
        if model_info.empty:
            raise ValueError(f"Model o ID {model_id} nie istnieje")
        
        # Feature importance
        importance_query = """
        SELECT feature_name, importance_value, importance_rank, importance_percentage
        FROM feature_importance
        WHERE model_id = %(model_id)s
        ORDER BY importance_rank
        """
        feature_importance = self.db.execute_query(importance_query, {'model_id': model_id})
        
        return {
            'model_info': model_info.iloc[0].to_dict(),
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def load_model_object(self, model_id: int) -> Any:
        """Ładuje zserializowany obiekt modelu"""
        query = """
        SELECT model_binary FROM ml_models WHERE model_id = %(model_id)s
        """
        result = self.db.execute_query(query, {'model_id': model_id})
        
        if result.empty or result.iloc[0]['model_binary'] is None:
            raise ValueError(f"Nie znaleziono zserializowanego modelu dla ID {model_id}")
        
        try:
            model_binary = result.iloc[0]['model_binary']
            return pickle.loads(model_binary)
        except Exception as e:
            raise ValueError(f"Błąd podczas deserializacji modelu: {e}")

class AnalyticsRepository:
    """
    Klasa do operacji analitycznych na danych
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Pobiera statystyki użytkownika"""
        
        # Liczba zbiorów danych
        datasets_count_query = """
        SELECT COUNT(*) as count FROM datasets WHERE user_id = %(user_id)s
        """
        datasets_count = self.db.execute_query(datasets_count_query, {'user_id': user_id})
        
        # Liczba modeli
        models_count_query = """
        SELECT COUNT(*) as count FROM ml_models WHERE user_id = %(user_id)s
        """
        models_count = self.db.execute_query(models_count_query, {'user_id': user_id})
        
        # Najlepsze modele
        best_models_query = """
        SELECT 
            m.model_name,
            m.algorithm_name,
            COALESCE(mt.accuracy, mt.r2_score) as score,
            m.problem_type
        FROM ml_models m
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.user_id = %(user_id)s AND m.training_status = 'completed'
        ORDER BY COALESCE(mt.accuracy, mt.r2_score) DESC
        LIMIT 5
        """
        best_models = self.db.execute_query(best_models_query, {'user_id': user_id})
        
        # Najczęściej używane algorytmy
        algorithms_query = """
        SELECT 
            algorithm_name,
            COUNT(*) as usage_count
        FROM ml_models
        WHERE user_id = %(user_id)s
        GROUP BY algorithm_name
        ORDER BY usage_count DESC
        LIMIT 5
        """
        popular_algorithms = self.db.execute_query(algorithms_query, {'user_id': user_id})
        
        return {
            'datasets_count': int(datasets_count.iloc[0]['count']),
            'models_count': int(models_count.iloc[0]['count']),
            'best_models': best_models.to_dict('records'),
            'popular_algorithms': popular_algorithms.to_dict('records')
        }
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Pobiera globalne statystyki aplikacji"""
        
        # Całkowite liczby
        total_users_query = "SELECT COUNT(*) as count FROM users WHERE is_active = TRUE"
        total_datasets_query = "SELECT COUNT(*) as count FROM datasets"
        total_models_query = "SELECT COUNT(*) as count FROM ml_models WHERE training_status = 'completed'"
        
        total_users = self.db.execute_query(total_users_query)
        total_datasets = self.db.execute_query(total_datasets_query)
        total_models = self.db.execute_query(total_models_query)
        
        # Najpopularniejsze algorytmy globalnie
        popular_algorithms_query = """
        SELECT 
            algorithm_name,
            COUNT(*) as usage_count,
            AVG(COALESCE(mt.accuracy, mt.r2_score)) as avg_score
        FROM ml_models m
        LEFT JOIN model_metrics mt ON m.model_id = mt.model_id
        WHERE m.training_status = 'completed'
        GROUP BY algorithm_name
        ORDER BY usage_count DESC
        LIMIT 10
        """
        popular_algorithms = self.db.execute_query(popular_algorithms_query)
        
        # Statystyki problemów ML
        problem_types_query = """
        SELECT 
            problem_type,
            COUNT(*) as count
        FROM ml_models
        WHERE training_status = 'completed'
        GROUP BY problem_type
        """
        problem_types = self.db.execute_query(problem_types_query)
        
        return {
            'total_users': int(total_users.iloc[0]['count']),
            'total_datasets': int(total_datasets.iloc[0]['count']),
            'total_models': int(total_models.iloc[0]['count']),
            'popular_algorithms': popular_algorithms.to_dict('records'),
            'problem_types_distribution': problem_types.to_dict('records')
        }

class DatabaseUtils:
    """
    Klasa z funkcjami pomocniczymi dla bazy danych
    """
    
    @staticmethod
    def create_tables(db_manager: DatabaseManager):
        """Tworzy wszystkie tabele w bazie danych"""
        try:
            # Czytanie pliku schema.sql
            with open('db/schema.sql', 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Wykonanie skryptu
            with db_manager.engine.connect() as conn:
                # Podzielenie na pojedyncze statements
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
                conn.commit()
            
            logger.info("Tabele zostały utworzone pomyślnie")
            
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia tabel: {e}")
            raise
    
    @staticmethod
    def cleanup_old_data(db_manager: DatabaseManager, days_old: int = 30):
        """Czyści stare dane z bazy"""
        
        cleanup_queries = [
            # Stare sesje
            "DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '%s days'",
            
            # Stare logi aktywności
            "DELETE FROM activity_logs WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'",
            
            # Nieukończone modele starsze niż X dni
            """DELETE FROM ml_models 
               WHERE training_status IN ('pending', 'training', 'failed') 
               AND created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'"""
        ]
        
        try:
            with db_manager.engine.connect() as conn:
                for query in cleanup_queries:
                    result = conn.execute(text(query % days_old))
                    logger.info(f"Usunięto {result.rowcount} rekordów")
                conn.commit()
            
            logger.info(f"Oczyszczono dane starsze niż {days_old} dni")
            
        except Exception as e:
            logger.error(f"Błąd podczas czyszczenia danych: {e}")
            raise
    
    @staticmethod
    def backup_database(db_manager: DatabaseManager, backup_path: str):
        """Tworzy backup bazy danych"""
        import subprocess
        import os
        
        try:
            # Pobieranie informacji o połączeniu
            url_parts = db_manager.settings.get_database_url().replace('postgresql://', '').split('@')
            user_pass = url_parts[0].split(':')
            host_db = url_parts[1].split('/')
            
            user = user_pass[0]
            password = user_pass[1] if len(user_pass) > 1 else ''
            host = host_db[0].split(':')[0]
            port = host_db[0].split(':')[1] if ':' in host_db[0] else '5432'
            database = host_db[1]
            
            # Ustawienie zmiennej środowiskowej dla hasła
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            # Komenda pg_dump
            cmd = [
                'pg_dump',
                f'--host={host}',
                f'--port={port}',
                f'--username={user}',
                f'--dbname={database}',
                '--verbose',
                '--clean',
                '--no-owner',
                '--no-privileges',
                f'--file={backup_path}'
            ]
            
            # Wykonanie backupu
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"Backup utworzony: {backup_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Błąd podczas tworzenia backupu: {e}")
            raise
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas backupu: {e}")
            raise

# Inicjalizacja globalnych obiektów
_db_manager = None
_dataset_repo = None
_model_repo = None
_analytics_repo = None

def get_database_manager() -> DatabaseManager:
    """Zwraca globalny obiekt DatabaseManager (singleton)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_dataset_repository() -> DatasetRepository:
    """Zwraca globalny obiekt DatasetRepository"""
    global _dataset_repo
    if _dataset_repo is None:
        _dataset_repo = DatasetRepository(get_database_manager())
    return _dataset_repo

def get_model_repository() -> ModelRepository:
    """Zwraca globalny obiekt ModelRepository"""
    global _model_repo
    if _model_repo is None:
        _model_repo = ModelRepository(get_database_manager())
    return _model_repo

def get_analytics_repository() -> AnalyticsRepository:
    """Zwraca globalny obiekt AnalyticsRepository"""
    global _analytics_repo
    if _analytics_repo is None:
        _analytics_repo = AnalyticsRepository(get_database_manager())
    return _analytics_repo