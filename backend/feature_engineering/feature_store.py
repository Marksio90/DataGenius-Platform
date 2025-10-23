"""
Feature Store - zarządzanie i przechowywanie features.

Funkcjonalności:
- Zapisywanie i ładowanie feature sets
- Wersjonowanie features
- Feature metadata
- Feature lineage
- Cache features
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from backend.error_handler import handle_errors
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeatureStore:
    """
    Local feature store.
    
    Struktura:
    feature_store/
      ├── features/
      │   ├── feature_set_v1.parquet
      │   └── feature_set_v2.parquet
      └── metadata/
          ├── feature_set_v1.json
          └── feature_set_v2.json
    """

    def __init__(self, store_path: Optional[Path] = None):
        """
        Inicjalizacja feature store.

        Args:
            store_path: Ścieżka do store
        """
        self.store_path = store_path or (settings.artifacts_dir / "feature_store")
        self.store_path.mkdir(exist_ok=True, parents=True)

        self.features_path = self.store_path / "features"
        self.features_path.mkdir(exist_ok=True)

        self.metadata_path = self.store_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)

        logger.info(f"Feature Store zainicjalizowany: {self.store_path}")

    @handle_errors(show_in_ui=False)
    def save_feature_set(
        self,
        df: pd.DataFrame,
        feature_set_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Zapisuje feature set.

        Args:
            df: DataFrame z features
            feature_set_name: Nazwa feature set
            version: Wersja (None = auto)
            metadata: Metadata

        Returns:
            str: Version string
        """
        # Auto version
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f"{feature_set_name}_v{version}"

        # Save data (Parquet for efficiency)
        data_path = self.features_path / f"{filename}.parquet"
        df.to_parquet(data_path, index=True)

        # Save metadata
        feature_metadata = {
            'name': feature_set_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(df.columns),
            'features': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'data_path': str(data_path),
            'metadata': metadata or {}
        }

        metadata_path = self.metadata_path / f"{filename}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)

        logger.info(f"Feature set zapisany: {feature_set_name} v{version}")

        return version

    @handle_errors(show_in_ui=False)
    def load_feature_set(
        self,
        feature_set_name: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ładuje feature set.

        Args:
            feature_set_name: Nazwa feature set
            version: Wersja (None = najnowsza)

        Returns:
            pd.DataFrame: Features
        """
        # Find versions
        if version is None:
            # Get latest version
            pattern = f"{feature_set_name}_v*.parquet"
            files = list(self.features_path.glob(pattern))

            if not files:
                raise FileNotFoundError(f"Brak feature set: {feature_set_name}")

            # Sort by modification time
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            data_path = files[0]
        else:
            filename = f"{feature_set_name}_v{version}.parquet"
            data_path = self.features_path / filename

            if not data_path.exists():
                raise FileNotFoundError(f"Feature set nie istnieje: {filename}")

        # Load
        df = pd.read_parquet(data_path)

        logger.info(f"Feature set załadowany: {data_path.name}")

        return df

    def list_feature_sets(self) -> List[Dict]:
        """
        Listuje dostępne feature sets.

        Returns:
            List[Dict]: Lista feature sets
        """
        feature_sets = []

        for metadata_file in self.metadata_path.glob("*.json"):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                feature_sets.append(metadata)

        # Sort by created_at
        feature_sets.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return feature_sets

    def get_feature_set_info(
        self,
        feature_set_name: str,
        version: Optional[str] = None
    ) -> Dict:
        """
        Zwraca informacje o feature set.

        Args:
            feature_set_name: Nazwa
            version: Wersja

        Returns:
            Dict: Metadata
        """
        if version is None:
            # Get latest
            pattern = f"{feature_set_name}_v*.json"
            files = list(self.metadata_path.glob(pattern))

            if not files:
                raise FileNotFoundError(f"Brak feature set: {feature_set_name}")

            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            metadata_file = files[0]
        else:
            filename = f"{feature_set_name}_v{version}.json"
            metadata_file = self.metadata_path / filename

            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata nie istnieje: {filename}")

        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return metadata

    def delete_feature_set(
        self,
        feature_set_name: str,
        version: str
    ):
        """
        Usuwa feature set.

        Args:
            feature_set_name: Nazwa
            version: Wersja
        """
        filename = f"{feature_set_name}_v{version}"

        # Delete data
        data_path = self.features_path / f"{filename}.parquet"
        if data_path.exists():
            data_path.unlink()

        # Delete metadata
        metadata_path = self.metadata_path / f"{filename}.json"
        if metadata_path.exists():
            metadata_path.unlink()

        logger.info(f"Feature set usunięty: {filename}")

    def merge_feature_sets(
        self,
        feature_sets: List[Tuple[str, Optional[str]]],
        output_name: str
    ) -> str:
        """
        Merguje wiele feature sets.

        Args:
            feature_sets: Lista (name, version)
            output_name: Nazwa output feature set

        Returns:
            str: Wersja nowego feature set
        """
        logger.info(f"Mergowanie {len(feature_sets)} feature sets...")

        dfs = []

        for name, version in feature_sets:
            df = self.load_feature_set(name, version)
            dfs.append(df)

        # Merge on index
        df_merged = pd.concat(dfs, axis=1)

        # Remove duplicate columns
        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        # Save
        version = self.save_feature_set(
            df_merged,
            output_name,
            metadata={
                'source_feature_sets': [
                    {'name': name, 'version': version}
                    for name, version in feature_sets
                ]
            }
        )

        logger.info(f"Feature sets zmergowane: {output_name} v{version}")

        return version