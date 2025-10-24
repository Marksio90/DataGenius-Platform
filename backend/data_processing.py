import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Klasa do zaawansowanego przetwarzania i analizy danych"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.pca = None
        self.is_processed = False
        
    def load_sample_datasets(self):
        """Ładuje przykładowe datasety"""
        
        datasets = {
            "🍎 Iris Dataset": self._create_iris_data(),
            "🏠 Boston Housing": self._create_housing_data(), 
            "🍷 Wine Quality": self._create_wine_data(),
            "💳 Titanic": self._create_titanic_data(),
            "🚗 Auto MPG": self._create_auto_data(),
            "📊 Sales Data": self._create_sales_data(),
            "🏥 Medical Data": self._create_medical_data(),
            "💰 Financial Data": self._create_financial_data()
        }
        
        return datasets
    
    def _create_iris_data(self):
        """Tworzy dataset podobny do Iris"""
        np.random.seed(42)
        
        data = []
        species = ['setosa', 'versicolor', 'virginica']
        
        for i, spec in enumerate(species):
            n_samples = 50
            sepal_length = np.random.normal(5.0 + i, 0.5, n_samples)
            sepal_width = np.random.normal(3.0 + i*0.3, 0.3, n_samples)  
            petal_length = np.random.normal(2.0 + i*1.5, 0.5, n_samples)
            petal_width = np.random.normal(0.5 + i*0.8, 0.3, n_samples)
            
            for j in range(n_samples):
                data.append({
                    'sepal_length': sepal_length[j],
                    'sepal_width': sepal_width[j],
                    'petal_length': petal_length[j], 
                    'petal_width': petal_width[j],
                    'species': spec
                })
        
        return pd.DataFrame(data)
    
    def _create_housing_data(self):
        """Tworzy dataset mieszkaniowy"""
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'rooms': np.random.randint(1, 8, n_samples),
            'area': np.random.normal(100, 30, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'location': np.random.choice(['Center', 'Suburbs', 'Outskirts'], n_samples),
            'garage': np.random.choice([0, 1], n_samples),
            'garden': np.random.choice([0, 1], n_samples)
        }
        
        # Cena zależna od innych cech
        price_base = (data['rooms'] * 50000 + 
                     data['area'] * 1000 + 
                     (50 - data['age']) * 1000)
        
        location_multiplier = {'Center': 1.5, 'Suburbs': 1.2, 'Outskirts': 1.0}
        price_multipliers = [location_multiplier[loc] for loc in data['location']]
        
        data['price'] = price_base * price_multipliers + np.random.normal(0, 20000, n_samples)
        data['price'] = np.maximum(data['price'], 50000)  # Minimum price
        
        return pd.DataFrame(data)
    
    def _create_wine_data(self):
        """Tworzy dataset wina"""
        np.random.seed(42)
        n_samples = 400
        
        quality = np.random.randint(3, 9, n_samples)
        
        data = {
            'alcohol': np.random.normal(10 + quality * 0.3, 1, n_samples),
            'acidity': np.random.normal(8 - quality * 0.2, 1, n_samples),
            'sugar': np.random.normal(5 + quality * 0.1, 2, n_samples),
            'pH': np.random.normal(3.2 + quality * 0.05, 0.2, n_samples),
            'sulfates': np.random.normal(0.5 + quality * 0.05, 0.1, n_samples),
            'density': np.random.normal(0.996, 0.002, n_samples),
            'color': np.random.choice(['red', 'white'], n_samples),
            'region': np.random.choice(['Toskania', 'Bordeaux', 'Rioja', 'Napa'], n_samples),
            'quality': quality
        }
        
        return pd.DataFrame(data)
    
    def _create_titanic_data(self):
        """Tworzy dataset Titanic"""
        np.random.seed(42)
        n_samples = 500
        
        # Generowanie danych z realistycznymi korelacjami
        pclass = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5])
        sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
        age = np.random.normal(30, 15, n_samples)
        age = np.clip(age, 1, 80)  # Ograniczenie wieku
        
        # Prawdopodobieństwo przeżycia zależne od klasy, płci i wieku
        survival_prob = np.zeros(n_samples)
        for i in range(n_samples):
            base_prob = 0.4
            if sex[i] == 'female':
                base_prob += 0.4
            if pclass[i] == 1:
                base_prob += 0.2
            elif pclass[i] == 2:
                base_prob += 0.1
            if age[i] < 16:
                base_prob += 0.2
            
            survival_prob[i] = min(base_prob, 0.95)
        
        survived = np.random.binomial(1, survival_prob, n_samples)
        
        data = {
            'pclass': pclass,
            'sex': sex,
            'age': age,
            'sibsp': np.random.poisson(0.5, n_samples),
            'parch': np.random.poisson(0.4, n_samples),
            'fare': np.random.lognormal(3, 1, n_samples),
            'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7]),
            'survived': survived
        }
        
        return pd.DataFrame(data)
    
    def _create_auto_data(self):
        """Tworzy dataset samochodowy"""
        np.random.seed(42)
        n_samples = 350
        
        data = {
            'cylinders': np.random.choice([4, 6, 8], n_samples, p=[0.6, 0.3, 0.1]),
            'displacement': np.random.normal(200, 50, n_samples),
            'horsepower': np.random.normal(150, 40, n_samples),
            'weight': np.random.normal(3000, 500, n_samples),
            'acceleration': np.random.normal(15, 3, n_samples),
            'model_year': np.random.randint(70, 85, n_samples),
            'origin': np.random.choice(['USA', 'Europe', 'Japan'], n_samples, p=[0.6, 0.2, 0.2])
        }
        
        # MPG zależne od innych parametrów
        mpg = (50 - data['weight']/100 - data['displacement']/10 + 
               data['model_year']/2 + np.random.normal(0, 3, n_samples))
        data['mpg'] = np.maximum(mpg, 5)
        
        return pd.DataFrame(data)
    
    def _create_sales_data(self):
        """Tworzy dataset sprzedażowy"""
        np.random.seed(42)
        n_samples = 600
        
        # Generowanie dat
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        
        # Sezonowość
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        trend = 1000 + np.arange(n_samples) * 0.5
        
        data = {
            'date': dates,
            'product': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'sales_rep': np.random.choice([f'Rep_{i}' for i in range(1, 11)], n_samples),
            'advertising_spend': np.random.gamma(2, 500, n_samples),
            'temperature': np.random.normal(20, 10, n_samples),
            'is_weekend': [(d.weekday() >= 5) for d in dates]
        }
        
        # Sprzedaż zależna od wielu czynników
        base_sales = trend * seasonal_factor
        advertising_effect = data['advertising_spend'] * 0.1
        weekend_effect = np.where(data['is_weekend'], 200, 0)
        
        data['sales'] = (base_sales + advertising_effect + weekend_effect + 
                        np.random.normal(0, 100, n_samples))
        data['sales'] = np.maximum(data['sales'], 0)
        
        return pd.DataFrame(data)
    
    def _create_medical_data(self):
        """Tworzy dataset medyczny"""
        np.random.seed(42)
        n_samples = 400
        
        age = np.random.gamma(4, 10, n_samples)
        age = np.clip(age, 18, 90)
        
        data = {
            'age': age,
            'gender': np.random.choice(['M', 'F'], n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure_sys': np.random.normal(120 + age * 0.3, 15, n_samples),
            'blood_pressure_dia': np.random.normal(80 + age * 0.2, 10, n_samples),
            'cholesterol': np.random.normal(200 + age * 0.5, 30, n_samples),
            'glucose': np.random.normal(90 + age * 0.2, 15, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'exercise_hours': np.random.gamma(2, 1, n_samples),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        # Prawdopodobieństwo choroby
        risk_score = (age * 0.01 + data['bmi'] * 0.05 + data['smoking'] * 0.3 + 
                     data['family_history'] * 0.4 + np.random.normal(0, 0.2, n_samples))
        
        data['disease_risk'] = 1 / (1 + np.exp(-risk_score))  # Sigmoid
        data['has_disease'] = np.random.binomial(1, data['disease_risk'], n_samples)
        
        return pd.DataFrame(data)
    
    def _create_financial_data(self):
        """Tworzy dataset finansowy"""
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'income': np.random.lognormal(10, 0.8, n_samples),
            'age': np.random.normal(35, 12, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                        n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'experience_years': np.random.gamma(2, 5, n_samples),
            'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Other'],
                                           n_samples, p=[0.25, 0.2, 0.15, 0.15, 0.25]),
            'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples, p=[0.3, 0.4, 0.3]),
            'debt': np.random.lognormal(8, 1.2, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples)
        }
        
        # Ograniczenia realistyczne
        data['age'] = np.clip(data['age'], 22, 70)
        data['credit_score'] = np.clip(data['credit_score'], 300, 850)
        data['experience_years'] = np.minimum(data['experience_years'], data['age'] - 22)
        
        return pd.DataFrame(data)
    
    def perform_eda(self, df, target_column=None):
        """Wykonuje eksploracyjną analizę danych"""
        
        eda_results = {}
        
        # Podstawowe informacje
        eda_results['basic_info'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Statystyki opisowe
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            eda_results['numeric_stats'] = df[numeric_cols].describe()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            eda_results['categorical_stats'] = {}
            for col in categorical_cols:
                eda_results['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
        
        # Analiza korelacji dla target
        if target_column and target_column in df.columns:
            if target_column in numeric_cols:
                # Korelacja Pearsona dla target numerycznego
                correlations = df[numeric_cols].corr()[target_column].abs().sort_values(ascending=False)
                eda_results['target_correlations'] = correlations.drop(target_column).head(10).to_dict()
            
            # Mutual Information dla wszystkich cech
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 10:
                # Klasyfikacja
                mi_scores = mutual_info_classif(pd.get_dummies(X), y, random_state=42)
            else:
                # Regresja
                mi_scores = mutual_info_regression(pd.get_dummies(X), y, random_state=42)
            
            mi_dict = dict(zip(X.columns, mi_scores))
            eda_results['mutual_information'] = dict(sorted(mi_dict.items(), 
                                                           key=lambda x: x[1], reverse=True)[:10])
        
        return eda_results
    
    def detect_outliers(self, df, method='iqr'):
        """Wykrywa outliers w danych"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > 3]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'values': outliers[col].tolist()[:10]  # Pierwszych 10
            }
        
        return outliers_info
    
    def suggest_data_improvements(self, df):
        """Sugeruje ulepszenia danych"""
        
        suggestions = []
        
        # Brakujące wartości
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.3]
        
        if len(high_missing) > 0:
            suggestions.append({
                'type': 'missing_data',
                'severity': 'high',
                'message': f"Kolumny z >30% brakujących wartości: {list(high_missing.index)}",
                'action': "Rozważ usunięcie lub specjalne imputowanie"
            })
        
        # Niska wariancja
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() < 1e-6:
                suggestions.append({
                    'type': 'low_variance',
                    'severity': 'medium',
                    'message': f"Kolumna '{col}' ma bardzo niską wariancję",
                    'action': "Rozważ usunięcie tej kolumny"
                })
        
        # Wysokie korelacje
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Znajdź pary z wysoką korelacją
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                suggestions.append({
                    'type': 'high_correlation',
                    'severity': 'medium',
                    'message': f"Wysokie korelacje między: {high_corr_pairs[:3]}",
                    'action': "Rozważ usunięcie jednej z skorelowanych cech"
                })
        
        # Outliers
        outliers_info = self.detect_outliers(df)
        high_outlier_cols = [col for col, info in outliers_info.items() 
                            if info['percentage'] > 5]
        
        if high_outlier_cols:
            suggestions.append({
                'type': 'outliers',
                'severity': 'low',
                'message': f"Kolumny z >5% outliers: {high_outlier_cols}",
                'action': "Sprawdź czy outliers to błędy czy prawdziwe wartości"
            })
        
        return suggestions
    
    def create_advanced_visualizations(self, df, target_column=None):
        """Tworzy zaawansowane wizualizacje"""
        
        visualizations = {}
        
        # 1. Macierz korelacji z dendrogramem
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix, 
                color_continuous_scale='RdBu_r',
                title="Macierz korelacji",
                aspect='auto'
            )
            visualizations['correlation_heatmap'] = fig
        
        # 2. Pair plot dla top cech
        if target_column and len(numeric_cols) > 1:
            # Wybierz top 4 cechy najbardziej skorelowane z target
            if target_column in numeric_cols:
                target_corr = df[numeric_cols].corr()[target_column].abs().sort_values(ascending=False)
                top_features = target_corr.drop(target_column).head(3).index.tolist()
                top_features.append(target_column)
                
                if len(top_features) >= 2:
                    fig = px.scatter_matrix(
                        df[top_features],
                        title="Scatter matrix najważniejszych cech"
                    )
                    visualizations['scatter_matrix'] = fig
        
        # 3. Rozkłady zmiennych
        if len(numeric_cols) > 0:
            # Wybierz kolumnę z największą wariancją dla przykładu
            var_col = df[numeric_cols].var().idxmax()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[var_col], name='Histogram', nbinsx=30),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[var_col], name='Box Plot'),
                row=1, col=2
            )
            
            # Violin plot
            fig.add_trace(
                go.Violin(y=df[var_col], name='Violin Plot'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"Analiza rozkładu: {var_col}",
                showlegend=False,
                height=600
            )
            
            visualizations['distribution_analysis'] = fig
        
        # 4. Analiza kategoryczna
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and target_column:
            cat_col = categorical_cols[0]  # Pierwsza kolumna kategoryczna
            
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 10:
                # Target kategoryczny - stacked bar
                cross_tab = pd.crosstab(df[cat_col], df[target_column])
                
                fig = go.Figure()
                for col in cross_tab.columns:
                    fig.add_trace(go.Bar(
                        x=cross_tab.index,
                        y=cross_tab[col],
                        name=str(col)
                    ))
                
                fig.update_layout(
                    title=f"Związek {cat_col} z {target_column}",
                    barmode='stack',
                    xaxis_title=cat_col,
                    yaxis_title='Liczba obserwacji'
                )
                
            else:
                # Target numeryczny - box plot
                fig = px.box(
                    df, x=cat_col, y=target_column,
                    title=f"Rozkład {target_column} według {cat_col}"
                )
            
            visualizations['categorical_analysis'] = fig
        
        return visualizations
    
    def perform_feature_engineering(self, df):
        """Wykonuje podstawowy feature engineering"""
        
        df_engineered = df.copy()
        new_features_info = []
        
        # 1. Tworzenie cech pochodnych dla dat
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df_engineered[f'{col}_year'] = df[col].dt.year
            df_engineered[f'{col}_month'] = df[col].dt.month
            df_engineered[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df_engineered[f'{col}_quarter'] = df[col].dt.quarter
            
            new_features_info.append(f"Utworzono cechy czasowe dla {col}")
        
        # 2. Binning dla cech numerycznych
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() > 10:  # Tylko dla cech z dużą liczbą unikalnych wartości
                try:
                    df_engineered[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=False)
                    new_features_info.append(f"Utworzono wersję zbinowaną dla {col}")
                except:
                    pass
        
        # 3. Interakcje między cechami numerycznymi (top 3)
        if len(numeric_cols) >= 2:
            # Wybierz top 3 cechy o największej wariancji
            top_numeric = df[numeric_cols].var().nlargest(3).index.tolist()
            
            if len(top_numeric) >= 2:
                # Utwórz interakcje między parami
                df_engineered[f'{top_numeric[0]}_x_{top_numeric[1]}'] = df[top_numeric[0]] * df[top_numeric[1]]
                new_features_info.append(f"Utworzono interakcję {top_numeric[0]} × {top_numeric[1]}")
                
                if len(top_numeric) >= 3:
                    df_engineered[f'{top_numeric[0]}_x_{top_numeric[2]}'] = df[top_numeric[0]] * df[top_numeric[2]]
                    new_features_info.append(f"Utworzono interakcję {top_numeric[0]} × {top_numeric[2]}")
        
        # 4. Agregacje dla cech kategorycznych
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for cat_col in categorical_cols:
            if df[cat_col].nunique() < 20:  # Tylko dla kategorii z niewielką liczbą poziomów
                for num_col in numeric_cols[:2]:  # Tylko dla 2 pierwszych cech numerycznych
                    try:
                        # Średnia w grupach
                        group_mean = df.groupby(cat_col)[num_col].mean()
                        df_engineered[f'{num_col}_mean_by_{cat_col}'] = df[cat_col].map(group_mean)
                        new_features_info.append(f"Utworzono średnią {num_col} według {cat_col}")
                    except:
                        pass
        
        return df_engineered, new_features_info
    
    def perform_dimensionality_reduction(self, df, target_column=None, method='pca'):
        """Wykonuje redukcję wymiarowości"""
        
        # Przygotuj dane numeryczne
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) < 2:
            return None, "Za mało cech numerycznych do redukcji wymiarowości"
        
        X = df[numeric_cols].fillna(0)  # Wypełnij brakujące wartości
        
        # Standaryzacja
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'pca':
            reducer = PCA(n_components=min(2, len(numeric_cols)))
            X_reduced = reducer.fit_transform(X_scaled)
            
            # Informacje o komponenetach
            explained_var = reducer.explained_variance_ratio_
            
            result_df = pd.DataFrame({
                'PC1': X_reduced[:, 0],
                'PC2': X_reduced[:, 1] if X_reduced.shape[1] > 1 else X_reduced[:, 0]
            })
            
            # Dodaj target jeśli istnieje
            if target_column:
                result_df[target_column] = df[target_column].values
            
            info = {
                'explained_variance': explained_var.tolist(),
                'total_explained': explained_var.sum(),
                'feature_importance': dict(zip(numeric_cols, 
                                             np.abs(reducer.components_[0]) if len(reducer.components_) > 0 else []))
            }
            
        elif method == 'tsne':
            # t-SNE (wolniejsze, ale lepsze dla wizualizacji)
            if len(X_scaled) > 1000:  # Ogranicz dla dużych datasetów
                indices = np.random.choice(len(X_scaled), 1000, replace=False)
                X_sample = X_scaled[indices]
            else:
                X_sample = X_scaled
                indices = np.arange(len(X_scaled))
            
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)-1))
            X_reduced = reducer.fit_transform(X_sample)
            
            result_df = pd.DataFrame({
                'TSNE1': X_reduced[:, 0],
                'TSNE2': X_reduced[:, 1]
            })
            
            # Dodaj target dla próbki
            if target_column:
                result_df[target_column] = df[target_column].iloc[indices].values
            
            info = {
                'method': 'tsne',
                'sample_size': len(X_sample),
                'original_features': len(numeric_cols)
            }
        
        return result_df, info

# Singleton pattern
_data_processor_instance = None

def get_data_processor():
    """Zwraca singleton instancję DataProcessor"""
    global _data_processor_instance
    if _data_processor_instance is None:
        _data_processor_instance = DataProcessor()
    return _data_processor_instance

# Pomocnicze funkcje dla Streamlit
def render_eda_dashboard(df, target_column=None):
    """Renderuje dashboard EDA"""
    
    data_processor = get_data_processor()
    
    # Podstawowe metryki
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Wiersze", f"{len(df):,}")
    
    with col2:
        st.metric("📝 Kolumny", f"{len(df.columns)}")
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("❌ Brakujące", f"{missing_pct:.1f}%")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Rozmiar", f"{memory_mb:.1f} MB")
    
    # Szczegółowa analiza
    st.markdown("---")
    
    # EDA
    with st.spinner("Wykonywanie analizy eksploracyjnej..."):
        eda_results = data_processor.perform_eda(df, target_column)
    
    # Korelacje z target
    if 'target_correlations' in eda_results:
        st.subheader("🎯 Korelacje z target")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Korelacja Pearsona:")
            for feature, corr in list(eda_results['target_correlations'].items())[:5]:
                st.text(f"• {feature}: {corr:.3f}")
        
        with col2:
            if 'mutual_information' in eda_results:
                st.markdown("#### 🔗 Mutual Information:")
                for feature, mi in list(eda_results['mutual_information'].items())[:5]:
                    st.text(f"• {feature}: {mi:.3f}")
    
    # Sugestie ulepszeń
    st.markdown("---")
    suggestions = data_processor.suggest_data_improvements(df)
    
    if suggestions:
        st.subheader("💡 Sugestie ulepszeń danych")
        
        for suggestion in suggestions:
            if suggestion['severity'] == 'high':
                st.error(f"🔴 {suggestion['message']}\n**Akcja**: {suggestion['action']}")
            elif suggestion['severity'] == 'medium':
                st.warning(f"🟡 {suggestion['message']}\n**Akcja**: {suggestion['action']}")
            else:
                st.info(f"🔵 {suggestion['message']}\n**Akcja**: {suggestion['action']}")

def render_advanced_visualizations(df, target_column=None):
    """Renderuje zaawansowane wizualizacje"""
    
    data_processor = get_data_processor()
    
    with st.spinner("Tworzenie wizualizacji..."):
        visualizations = data_processor.create_advanced_visualizations(df, target_column)
    
    for viz_name, fig in visualizations.items():
        st.plotly_chart(fig, use_container_width=True)

def render_feature_engineering_section(df):
    """Renderuje sekcję feature engineering"""
    
    st.subheader("🔧 Feature Engineering")
    
    data_processor = get_data_processor()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Wykonaj Feature Engineering"):
            with st.spinner("Tworzenie nowych cech..."):
                df_engineered, new_features_info = data_processor.perform_feature_engineering(df)
                
                st.session_state.df_engineered = df_engineered
                st.session_state.new_features_info = new_features_info
    
    with col2:
        if st.button("🔍 Redukcja wymiarowości (PCA)"):
            target_col = st.session_state.get('target_column')
            
            with st.spinner("Wykonywanie PCA..."):
                result_df, info = data_processor.perform_dimensionality_reduction(
                    df, target_col, method='pca'
                )
                
                if result_df is not None:
                    st.session_state.pca_result = result_df
                    st.session_state.pca_info = info
    
    # Wyświetl wyniki feature engineering
    if 'new_features_info' in st.session_state:
        st.markdown("#### ✨ Nowe cechy utworzone:")
        for info in st.session_state.new_features_info:
            st.success(f"✅ {info}")
        
        # Porównanie przed/po
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Oryginalne cechy", len(df.columns))
        
        with col2:
            if 'df_engineered' in st.session_state:
                st.metric("🚀 Po feature engineering", len(st.session_state.df_engineered.columns))
    
    # Wyświetl wyniki PCA
    if 'pca_result' in st.session_state:
        st.markdown("#### 📊 Wyniki PCA:")
        
        pca_info = st.session_state.pca_info
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Wariancja wyjaśniona", f"{pca_info['total_explained']:.1%}")
        
        with col2:
            st.metric("📉 Redukcja wymiarów", f"{len(pca_info.get('feature_importance', {}))} → 2")
        
        # Wykres PCA
        pca_df = st.session_state.pca_result
        target_col = st.session_state.get('target_column')
        
        if target_col and target_col in pca_df.columns:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color=target_col,
                title="Wizualizacja PCA",
                labels={'PC1': f'PC1 ({pca_info["explained_variance"][0]:.1%})',
                       'PC2': f'PC2 ({pca_info["explained_variance"][1]:.1%})'}
            )
        else:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                title="Wizualizacja PCA"
            )
        
        st.plotly_chart(fig, use_container_width=True)

def render_data_quality_report(df):
    """Renderuje raport jakości danych"""
    
    st.subheader("📋 Raport jakości danych")
    
    data_processor = get_data_processor()
    
    # Sekcje raportu
    tab1, tab2, tab3 = st.tabs(["📊 Podstawowe", "🔍 Outliers", "💡 Sugestie"])
    
    with tab1:
        # Podstawowe statystyki
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Statystyki numeryczne:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                stats_df = df[numeric_cols].describe()
                st.dataframe(stats_df.round(3))
            else:
                st.info("Brak kolumn numerycznych")
        
        with col2:
            st.markdown("#### 🔤 Statystyki kategoryczne:")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Pokaż tylko pierwsze 3
                    unique_count = df[col].nunique()
                    most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                    
                    st.text(f"• {col}:")
                    st.text(f"  Unikalne: {unique_count}")
                    st.text(f"  Najczęstsze: {most_common}")
                    st.text("")
            else:
                st.info("Brak kolumn kategorycznych")
    
    with tab2:
        # Analiza outliers
        st.markdown("#### 🎯 Wykrywanie outliers")
        
        outlier_method = st.selectbox(
            "Metoda wykrywania:",
            ["IQR (Interquartile Range)", "Z-Score"],
            key="outlier_method"
        )
        
        method = 'iqr' if 'IQR' in outlier_method else 'zscore'
        
        with st.spinner("Analizowanie outliers..."):
            outliers_info = data_processor.detect_outliers(df, method=method)
        
        # Wyświetl wyniki
        for col, info in outliers_info.items():
            if info['count'] > 0:
                st.warning(f"**{col}**: {info['count']} outliers ({info['percentage']:.1f}%)")
                
                # Pokaż przykładowe wartości
                if info['values']:
                    st.text(f"Przykłady: {info['values'][:5]}")
            else:
                st.success(f"**{col}**: Brak outliers")
    
    with tab3:
        # Sugestie ulepszeń
        suggestions = data_processor.suggest_data_improvements(df)
        
        if suggestions:
            for suggestion in suggestions:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}
                icon = severity_icon.get(suggestion['severity'], "ℹ️")
                
                with st.expander(f"{icon} {suggestion['type'].replace('_', ' ').title()}"):
                    st.write(f"**Problem**: {suggestion['message']}")
                    st.write(f"**Rekomendacja**: {suggestion['action']}")
        else:
            st.success("✅ Dane wyglądają dobrze! Nie znaleziono problemów.")

def render_dataset_comparison(df1, df2, name1="Dataset 1", name2="Dataset 2"):
    """Porównuje dwa datasety"""
    
    st.subheader(f"⚖️ Porównanie: {name1} vs {name2}")
    
    # Podstawowe porównanie
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"#### 📊 {name1}")
        st.metric("Wiersze", len(df1))
        st.metric("Kolumny", len(df1.columns))
        st.metric("Rozmiar (MB)", f"{df1.memory_usage(deep=True).sum() / 1024**2:.2f}")
    
    with col2:
        st.markdown(f"#### 📊 {name2}")
        st.metric("Wiersze", len(df2))
        st.metric("Kolumny", len(df2.columns))
        st.metric("Rozmiar (MB)", f"{df2.memory_usage(deep=True).sum() / 1024**2:.2f}")
    
    with col3:
        st.markdown("#### 📈 Różnice")
        row_diff = len(df2) - len(df1)
        col_diff = len(df2.columns) - len(df1.columns)
        
        st.metric("Δ Wiersze", row_diff)
        st.metric("Δ Kolumny", col_diff)
    
    # Porównanie kolumn
    st.markdown("---")
    st.markdown("#### 🔍 Porównanie kolumn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        common_cols = set(df1.columns) & set(df2.columns)
        st.success(f"✅ Wspólne kolumny: {len(common_cols)}")
        if common_cols:
            for col in list(common_cols)[:5]:
                st.text(f"• {col}")
    
    with col2:
        unique_cols1 = set(df1.columns) - set(df2.columns)
        unique_cols2 = set(df2.columns) - set(df1.columns)
        
        if unique_cols1:
            st.warning(f"⚠️ Tylko w {name1}: {len(unique_cols1)}")
            for col in list(unique_cols1)[:3]:
                st.text(f"• {col}")
        
        if unique_cols2:
            st.info(f"ℹ️ Tylko w {name2}: {len(unique_cols2)}")
            for col in list(unique_cols2)[:3]:
                st.text(f"• {col}")

def render_interactive_data_explorer(df):
    """Renderuje interaktywny eksplorator danych"""
    
    st.subheader("🔍 Interaktywny eksplorator danych")
    
    # Filtry
    with st.expander("🎛️ Filtry danych", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtry numeryczne
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                st.markdown("**📊 Filtry numeryczne:**")
                
                selected_numeric = st.selectbox("Kolumna numeryczna:", numeric_cols, key="filter_numeric")
                
                if selected_numeric:
                    min_val = float(df[selected_numeric].min())
                    max_val = float(df[selected_numeric].max())
                    
                    range_vals = st.slider(
                        f"Zakres {selected_numeric}:",
                        min_val, max_val, (min_val, max_val),
                        key="numeric_range"
                    )
                    
                    # Zastosuj filtr
                    df = df[(df[selected_numeric] >= range_vals[0]) & (df[selected_numeric] <= range_vals[1])]
        
        with col2:
            # Filtry kategoryczne
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                st.markdown("**🔤 Filtry kategoryczne:**")
                
                selected_categorical = st.selectbox("Kolumna kategoryczna:", categorical_cols, key="filter_categorical")
                
                if selected_categorical:
                    unique_vals = df[selected_categorical].unique()
                    
                    selected_vals = st.multiselect(
                        f"Wartości {selected_categorical}:",
                        unique_vals,
                        default=unique_vals.tolist(),
                        key="categorical_values"
                    )
                    
                    # Zastosuj filtr
                    if selected_vals:
                        df = df[df[selected_categorical].isin(selected_vals)]
    
    # Podgląd danych po filtracji
    st.markdown(f"📋 **Dane po filtracji**: {len(df)} wierszy, {len(df.columns)} kolumn")
    
    # Opcje wyświetlania
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_head = st.number_input("Pokaż pierwszych N wierszy:", 1, 100, 10)
    
    with col2:
        show_sample = st.checkbox("Losowa próbka", value=False)
    
    with col3:
        show_info = st.checkbox("Pokaż info o kolumnach", value=False)
    
    # Wyświetl dane
    if show_sample:
        sample_size = min(show_head, len(df))
        display_df = df.sample(n=sample_size)
        st.markdown(f"📊 **Losowa próbka ({sample_size} wierszy):**")
    else:
        display_df = df.head(show_head)
        st.markdown(f"📊 **Pierwsze {show_head} wiersze:**")
    
    st.dataframe(display_df)
    
    # Info o kolumnach
    if show_info:
        st.markdown("---")
        st.markdown("#### 📝 Informacje o kolumnach:")
        
        info_data = []
        for col in df.columns:
            info_data.append({
                'Kolumna': col,
                'Typ': str(df[col].dtype),
                'Brakujące': df[col].isnull().sum(),
                'Unikalne': df[col].nunique(),
                'Przykład': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
            })
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df)