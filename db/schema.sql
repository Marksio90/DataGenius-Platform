-- =============================================================================
-- SCHEMAT BAZY DANYCH DLA APLIKACJI "THE MOST IMPORTANT VARIABLES"
-- =============================================================================
-- Wersja: 1.0.1
-- Autor: Zespół Hackathon ML
-- Data: 2024

-- =============================================================================
-- DROP VIEW/TABLE IF EXISTS (KOLEJNOŚĆ: NAJPIERW WIDOKI, POTEM TABEL)
-- =============================================================================
DROP VIEW IF EXISTS v_dataset_stats CASCADE;
DROP VIEW IF EXISTS v_top_features CASCADE;
DROP VIEW IF EXISTS v_model_summary CASCADE;

DROP TABLE IF EXISTS model_predictions CASCADE;
DROP TABLE IF EXISTS feature_importance CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS dataset_columns CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS activity_logs CASCADE;

-- =============================================================================
-- TABELA UŻYTKOWNIKÓW
-- =============================================================================
CREATE TABLE users (
    user_id            SERIAL PRIMARY KEY,
    username           VARCHAR(50) UNIQUE NOT NULL,
    email              VARCHAR(100) UNIQUE NOT NULL,
    password_hash      VARCHAR(255) NOT NULL,
    first_name         VARCHAR(50),
    last_name          VARCHAR(50),
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active          BOOLEAN   DEFAULT TRUE,
    last_login         TIMESTAMP
);

-- =============================================================================
-- TABELA ZBIORÓW DANYCH
-- =============================================================================
CREATE TABLE datasets (
    dataset_id                 SERIAL PRIMARY KEY,
    user_id                    INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    dataset_name               VARCHAR(100) NOT NULL,
    original_filename          VARCHAR(255),
    file_size_bytes            BIGINT,
    rows_count                 INTEGER,
    columns_count              INTEGER,
    dataset_type               VARCHAR(20) CHECK (dataset_type IN ('csv', 'json', 'sample')),
    upload_timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description                TEXT,
    is_sample_dataset          BOOLEAN   DEFAULT FALSE,
    data_hash                  VARCHAR(64), -- SHA-256 hash dla integrity check

    -- Metadane
    missing_values_count       INTEGER      DEFAULT 0,
    numeric_columns_count      INTEGER      DEFAULT 0,
    categorical_columns_count  INTEGER      DEFAULT 0,
    memory_usage_mb            DECIMAL(10,2),

    -- Status przetwarzania
    processing_status          VARCHAR(20)  DEFAULT 'uploaded'
        CHECK (processing_status IN ('uploaded', 'processing', 'processed', 'error')),
    error_message              TEXT,

    created_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Opcjonalna unikalność nazwy zbioru per użytkownik
-- CREATE UNIQUE INDEX IF NOT EXISTS uq_datasets_user_name ON datasets(user_id, dataset_name);

-- =============================================================================
-- TABELA KOLUMN ZBIORÓW DANYCH
-- =============================================================================
CREATE TABLE dataset_columns (
    column_id               SERIAL PRIMARY KEY,
    dataset_id              INTEGER REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    column_name             VARCHAR(100) NOT NULL,
    column_index            INTEGER NOT NULL,
    data_type               VARCHAR(50) NOT NULL,

    -- Statystyki kolumny
    null_count              INTEGER      DEFAULT 0,
    null_percentage         DECIMAL(5,2) DEFAULT 0 CHECK (null_percentage BETWEEN 0 AND 100),
    unique_count            INTEGER,
    unique_percentage       DECIMAL(5,2)      CHECK (unique_percentage BETWEEN 0 AND 100),

    -- Dla kolumn numerycznych
    min_value               DECIMAL(20,10),
    max_value               DECIMAL(20,10),
    mean_value              DECIMAL(20,10),
    median_value            DECIMAL(20,10),
    std_deviation           DECIMAL(20,10),

    -- Dla kolumn kategorycznych
    most_frequent_value     TEXT,
    most_frequent_count     INTEGER,

    -- AI-generated description
    ai_description          TEXT,
    column_role             VARCHAR(20) CHECK (column_role IN ('feature', 'target', 'id', 'metadata')),

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(dataset_id, column_name)
);

-- =============================================================================
-- TABELA MODELI ML
-- =============================================================================
CREATE TABLE ml_models (
    model_id                    SERIAL PRIMARY KEY,
    dataset_id                  INTEGER REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    user_id                     INTEGER REFERENCES users(user_id) ON DELETE CASCADE,

    -- Konfiguracja modelu
    model_name                  VARCHAR(100) NOT NULL,
    target_column               VARCHAR(100) NOT NULL,
    problem_type                VARCHAR(20)  NOT NULL CHECK (problem_type IN ('regression', 'classification')),
    algorithm_name              VARCHAR(50)  NOT NULL,

    -- Parametry trenowania
    train_test_split            DECIMAL(3,2) DEFAULT 0.80 CHECK (train_test_split > 0 AND train_test_split < 1),
    cv_folds                    INTEGER      DEFAULT 5    CHECK (cv_folds >= 2),
    random_state                INTEGER      DEFAULT 42,

    -- Status trenowania
    training_status             VARCHAR(20)  DEFAULT 'pending'
        CHECK (training_status IN ('pending', 'training', 'completed', 'failed')),
    training_start_time         TIMESTAMP,
    training_end_time           TIMESTAMP,
    training_duration_seconds   INTEGER CHECK (training_duration_seconds IS NULL OR training_duration_seconds >= 0),

    -- Metadane modelu
    feature_count               INTEGER,
    training_samples_count      INTEGER,
    test_samples_count          INTEGER,

    -- Serializowany model (opcjonalnie)
    model_binary                BYTEA, -- Zapisany model w formacie pickle
    model_version               VARCHAR(20) DEFAULT '1.0',

    error_message               TEXT,
    notes                       TEXT,

    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TABELA METRYK MODELI
-- =============================================================================
CREATE TABLE model_metrics (
    metric_id               SERIAL PRIMARY KEY,
    model_id                INTEGER REFERENCES ml_models(model_id) ON DELETE CASCADE,

    -- Metryki regresji
    r2_score                DECIMAL(10,6),
    mae                     DECIMAL(15,6),
    mse                     DECIMAL(15,6),
    rmse                    DECIMAL(15,6),

    -- Metryki klasyfikacji
    accuracy                DECIMAL(10,6),
    precision_score         DECIMAL(10,6),
    recall_score            DECIMAL(10,6),
    f1_score                DECIMAL(10,6),

    -- Cross-validation scores
    cv_score_mean           DECIMAL(10,6),
    cv_score_std            DECIMAL(10,6),

    -- Dodatkowe metryki
    roc_auc                 DECIMAL(10,6), -- Dla klasyfikacji binarnej
    confusion_matrix        JSONB,         -- JSON z macierzą pomyłek
    classification_report   JSONB,         -- JSON z raportem klasyfikacji

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TABELA WAŻNOŚCI CECH (FEATURE IMPORTANCE)
-- =============================================================================
CREATE TABLE feature_importance (
    importance_id           SERIAL PRIMARY KEY,
    model_id                INTEGER REFERENCES ml_models(model_id) ON DELETE CASCADE,
    feature_name            VARCHAR(100) NOT NULL,
    importance_value        DECIMAL(15,10) NOT NULL,
    importance_rank         INTEGER NOT NULL CHECK (importance_rank >= 1),
    importance_percentage   DECIMAL(5,2) CHECK (importance_percentage IS NULL OR (importance_percentage >= 0 AND importance_percentage <= 100)),

    -- Dodatkowe informacje o cesze
    feature_type            VARCHAR(20), -- 'numeric', 'categorical', 'binary'
    original_column_name    VARCHAR(100), -- Na wypadek feature engineering

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_id, feature_name)
);

-- =============================================================================
-- TABELA PREDYKCJI MODELU
-- =============================================================================
CREATE TABLE model_predictions (
    prediction_id           SERIAL PRIMARY KEY,
    model_id                INTEGER REFERENCES ml_models(model_id) ON DELETE CASCADE,

    -- Dane wejściowe (JSON)
    input_data              JSONB NOT NULL, -- JSON z wartościami features
    prediction_value        DECIMAL(20,10), -- Dla regresji
    prediction_class        VARCHAR(100),   -- Dla klasyfikacji
    prediction_probability  DECIMAL(10,6) CHECK (prediction_probability IS NULL OR (prediction_probability >= 0 AND prediction_probability <= 1)),

    -- Metadane predykcji
    prediction_timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score        DECIMAL(10,6),
    processing_time_ms      INTEGER CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0),

    -- Opcjonalnie - rzeczywista wartość (do ewaluacji)
    actual_value            DECIMAL(20,10),
    actual_class            VARCHAR(100),

    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TABELA SESJI UŻYTKOWNIKÓW
-- =============================================================================
CREATE TABLE user_sessions (
    session_id      SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    session_token   VARCHAR(255) UNIQUE NOT NULL,
    session_data    JSONB, -- JSON z danymi sesji
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at      TIMESTAMP NOT NULL,
    is_active       BOOLEAN  DEFAULT TRUE,
    ip_address      INET,
    user_agent      TEXT
);

-- =============================================================================
-- TABELA LOGÓW AKTYWNOŚCI
-- =============================================================================
CREATE TABLE activity_logs (
    log_id          SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    activity_type   VARCHAR(50) NOT NULL, -- 'upload', 'train_model', 'predict', etc.
    description     TEXT,
    metadata        JSONB, -- JSON z dodatkowymi informacjami
    ip_address      INET,
    user_agent      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEKSY DLA WYDAJNOŚCI
-- =============================================================================

-- Indeksy dla datasets
CREATE INDEX idx_datasets_user_id          ON datasets(user_id);
CREATE INDEX idx_datasets_upload_timestamp ON datasets(upload_timestamp);
CREATE INDEX idx_datasets_type             ON datasets(dataset_type);
CREATE INDEX idx_datasets_status           ON datasets(processing_status);

-- Indeksy dla dataset_columns
CREATE INDEX idx_dataset_columns_dataset_id ON dataset_columns(dataset_id);
CREATE INDEX idx_dataset_columns_type       ON dataset_columns(data_type);
CREATE INDEX idx_dataset_columns_role       ON dataset_columns(column_role);

-- Indeksy dla ml_models
CREATE INDEX idx_ml_models_dataset_id   ON ml_models(dataset_id);
CREATE INDEX idx_ml_models_user_id      ON ml_models(user_id);
CREATE INDEX idx_ml_models_status       ON ml_models(training_status);
CREATE INDEX idx_ml_models_algorithm    ON ml_models(algorithm_name);
CREATE INDEX idx_ml_models_problem_type ON ml_models(problem_type);
CREATE INDEX idx_ml_models_created_at   ON ml_models(created_at);

-- Indeksy dla feature_importance
CREATE INDEX idx_feature_importance_model_id ON feature_importance(model_id);
CREATE INDEX idx_feature_importance_rank     ON feature_importance(importance_rank);
CREATE INDEX idx_feature_importance_value    ON feature_importance(importance_value DESC);

-- Indeksy dla model_predictions
CREATE INDEX idx_model_predictions_model_id   ON model_predictions(model_id);
CREATE INDEX idx_model_predictions_timestamp  ON model_predictions(prediction_timestamp);

-- Indeksy dla user_sessions
CREATE INDEX idx_user_sessions_user_id   ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token     ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Indeksy dla activity_logs
CREATE INDEX idx_activity_logs_user_id  ON activity_logs(user_id);
CREATE INDEX idx_activity_logs_type     ON activity_logs(activity_type);
CREATE INDEX idx_activity_logs_created_at ON activity_logs(created_at);

-- =============================================================================
-- TRIGGERY DO AUTOMATYCZNEGO USTAWIANIA updated_at
-- =============================================================================

-- Funkcja do ustawiania updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggery
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at
    BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- WIDOKI (VIEWS) DLA UŁATWIENIA ZAPYTAŃ
-- =============================================================================

-- Widok z kompletnymi informacjami o modelach
CREATE VIEW v_model_summary AS
SELECT 
    m.model_id,
    m.model_name,
    d.dataset_name,
    u.username,
    m.target_column,
    m.problem_type,
    m.algorithm_name,
    m.training_status,
    m.feature_count,
    m.training_duration_seconds,
    mt.r2_score,
    mt.accuracy,
    mt.cv_score_mean,
    m.created_at
FROM ml_models m
LEFT JOIN datasets d ON m.dataset_id = d.dataset_id
LEFT JOIN users u    ON m.user_id = u.user_id
LEFT JOIN model_metrics mt ON m.model_id = mt.model_id;

-- Widok z najważniejszymi cechami dla każdego modelu
CREATE VIEW v_top_features AS
SELECT 
    fi.model_id,
    m.model_name,
    fi.feature_name,
    fi.importance_value,
    fi.importance_rank,
    fi.importance_percentage
FROM feature_importance fi
JOIN ml_models m ON fi.model_id = m.model_id
WHERE fi.importance_rank <= 10
ORDER BY fi.model_id, fi.importance_rank;

-- Widok ze statystykami zbiorów danych
CREATE VIEW v_dataset_stats AS
SELECT 
    d.dataset_id,
    d.dataset_name,
    d.rows_count,
    d.columns_count,
    d.missing_values_count,
    d.memory_usage_mb,
    COUNT(DISTINCT m.model_id) AS models_count,
    d.created_at
FROM datasets d
LEFT JOIN ml_models m ON d.dataset_id = m.dataset_id
GROUP BY d.dataset_id, d.dataset_name, d.rows_count, d.columns_count, 
         d.missing_values_count, d.memory_usage_mb, d.created_at;

-- =============================================================================
-- FUNKCJE POMOCNICZE
-- =============================================================================

-- Funkcja do obliczania accuracy dla modelu
CREATE OR REPLACE FUNCTION calculate_model_accuracy(p_model_id INTEGER)
RETURNS DECIMAL(10,6) AS $$
DECLARE
    correct_predictions INTEGER;
    total_predictions   INTEGER;
    acc                 DECIMAL(10,6);
BEGIN
    -- Policz prawidłowe predykcje
    SELECT COUNT(*) INTO correct_predictions
    FROM model_predictions 
    WHERE model_id = p_model_id 
      AND prediction_class = actual_class;

    -- Policz wszystkie predykcje z rzeczywistą klasą
    SELECT COUNT(*) INTO total_predictions
    FROM model_predictions 
    WHERE model_id = p_model_id 
      AND actual_class IS NOT NULL;

    -- Oblicz accuracy
    IF total_predictions > 0 THEN
        acc := correct_predictions::DECIMAL / total_predictions::DECIMAL;
    ELSE
        acc := NULL;
    END IF;

    RETURN acc;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do czyszczenia starych sesji
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < CURRENT_TIMESTAMP OR is_active = FALSE;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- PRZYKŁADOWE DANE TESTOWE (OPCJONALNIE)
-- =============================================================================

-- Wstawienie przykładowych użytkowników
INSERT INTO users (username, email, password_hash, first_name, last_name) VALUES
('demo_user', 'demo@example.com', '$2b$12$example_hash', 'Demo', 'User'),
('admin',     'admin@example.com', '$2b$12$example_hash', 'Admin', 'User');

-- Wstawienie przykładowych zbiorów danych
INSERT INTO datasets (user_id, dataset_name, original_filename, rows_count, columns_count, dataset_type, is_sample_dataset, processing_status)
VALUES
(1, 'Boston Housing',     'boston_housing.csv', 506, 14, 'sample', TRUE, 'processed'),
(1, 'Iris Classification','iris.csv',           150, 5,  'sample', TRUE, 'processed'),
(1, 'Wine Quality',       'wine_quality.csv',  1599, 12, 'sample', TRUE, 'processed');

-- =============================================================================
-- KOMENTARZE I DOKUMENTACJA
-- =============================================================================

COMMENT ON TABLE users IS 'Tabela użytkowników aplikacji';
COMMENT ON TABLE datasets IS 'Tabela przechowująca informacje o zbiorach danych';
COMMENT ON TABLE dataset_columns IS 'Tabela z metadanymi kolumn w zbiorach danych';
COMMENT ON TABLE ml_models IS 'Tabela z wytrenowanymi modelami ML';
COMMENT ON TABLE model_metrics IS 'Tabela z metrykami wydajności modeli';
COMMENT ON TABLE feature_importance IS 'Tabela z ważnością cech dla każdego modelu';
COMMENT ON TABLE model_predictions IS 'Tabela z predykcjami modeli';

COMMENT ON COLUMN datasets.data_hash IS 'SHA-256 hash zawartości pliku dla kontroli integralności';
COMMENT ON COLUMN ml_models.model_binary IS 'Serializowany model ML w formacie binary (pickle)';
COMMENT ON COLUMN feature_importance.importance_percentage IS 'Procent ważności cechy względem całkowitej ważności';

-- =============================================================================
-- KOŃCOWE INFORMACJE (PODGLĄD TABEL)
-- =============================================================================
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY tablename;
