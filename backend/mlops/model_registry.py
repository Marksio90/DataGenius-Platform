"""
Model Registry - zarządzanie wersjami modeli.

Funkcjonalności:
- Rejestrowanie modeli
- Wersjonowanie
- Staging/Production promotion
- Model metadata
- Model lineage
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from backend.error_handler import handle_errors
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelRegistry:
    """
    Local model registry (bez MLflow).
    
    Struktura:
    registry/
      ├── models/
      │   ├── model_name_v1/
      │   │   ├── model.joblib
      │   │   └── metadata.json
      │   └── model_name_v2/
      └── registry.json
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Inicjalizacja registry.

        Args:
            registry_path: Ścieżka do registry
        """
        self.registry_path = registry_path or (settings.artifacts_dir / "registry")
        self.registry_path.mkdir(exist_ok=True, parents=True)

        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)

        self.registry_file = self.registry_path / "registry.json"

        # Load registry
        self.registry = self._load_registry()

        logger.info(f"Model Registry zainicjalizowany: {self.registry_path}")

    def _load_registry(self) -> Dict:
        """Ładuje registry z pliku."""
        if self.registry_file.exists():
            import json
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {'models': {}}

    def _save_registry(self):
        """Zapisuje registry do pliku."""
        import json
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    @handle_errors(show_in_ui=False)
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        stage: str = 'none',
        metadata: Optional[Dict] = None,
        tags: Optional[Dict] = None
    ) -> str:
        """
        Rejestruje model.

        Args:
            model: Model do zarejestrowania
            model_name: Nazwa modelu
            version: Wersja (None = auto-increment)
            stage: Stage ('none', 'staging', 'production', 'archived')
            metadata: Metadata
            tags: Tagi

        Returns:
            str: Version string
        """
        # Auto-increment version
        if version is None:
            if model_name in self.registry['models']:
                versions = [v['version'] for v in self.registry['models'][model_name]]
                last_version = max([int(v.replace('v', '')) for v in versions])
                version = f"v{last_version + 1}"
            else:
                version = "v1"

        # Create model dir
        model_dir = self.models_path / f"{model_name}_{version}"
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        model_metadata = {
            'name': model_name,
            'version': version,
            'stage': stage,
            'registered_at': datetime.now().isoformat(),
            'model_class': model.__class__.__name__,
            'model_path': str(model_path),
            'metadata': metadata or {},
            'tags': tags or {}
        }

        metadata_path = model_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update registry
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = []

        self.registry['models'][model_name].append(model_metadata)

        self._save_registry()

        logger.info(f"Model zarejestrowany: {model_name} {version} (stage={stage})")

        return version

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """
        Ładuje model z registry.

        Args:
            model_name: Nazwa modelu
            version: Wersja (None = najnowsza)
            stage: Stage (None = ignore)

        Returns:
            Any: Model
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} nie istnieje w registry")

        versions = self.registry['models'][model_name]

        # Filter by stage
        if stage:
            versions = [v for v in versions if v['stage'] == stage]

        if not versions:
            raise ValueError(f"Brak modeli dla {model_name} (stage={stage})")

        # Select version
        if version:
            model_metadata = next((v for v in versions if v['version'] == version), None)
            if model_metadata is None:
                raise ValueError(f"Wersja {version} nie istnieje dla {model_name}")
        else:
            # Latest version
            model_metadata = versions[-1]

        # Load model
        model_path = Path(model_metadata['model_path'])

        if not model_path.exists():
            raise FileNotFoundError(f"Model file nie istnieje: {model_path}")

        model = joblib.load(model_path)

        logger.info(f"Model załadowany: {model_name} {model_metadata['version']}")

        return model

    def update_stage(
        self,
        model_name: str,
        version: str,
        new_stage: str
    ):
        """
        Aktualizuje stage modelu.

        Args:
            model_name: Nazwa modelu
            version: Wersja
            new_stage: Nowy stage
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} nie istnieje")

        versions = self.registry['models'][model_name]
        model_metadata = next((v for v in versions if v['version'] == version), None)

        if model_metadata is None:
            raise ValueError(f"Wersja {version} nie istnieje")

        model_metadata['stage'] = new_stage
        model_metadata['stage_updated_at'] = datetime.now().isoformat()

        self._save_registry()

        logger.info(f"Stage zaktualizowany: {model_name} {version} -> {new_stage}")

    def list_models(self, stage: Optional[str] = None) -> List[Dict]:
        """
        Listuje modele.

        Args:
            stage: Filtr stage (None = wszystkie)

        Returns:
            List[Dict]: Lista modeli
        """
        all_models = []

        for model_name, versions in self.registry['models'].items():
            for version_info in versions:
                if stage is None or version_info['stage'] == stage:
                    all_models.append(version_info)

        return all_models

    def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict:
        """
        Zwraca informacje o modelu.

        Args:
            model_name: Nazwa modelu
            version: Wersja (None = najnowsza)

        Returns:
            Dict: Informacje
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} nie istnieje")

        versions = self.registry['models'][model_name]

        if version:
            model_metadata = next((v for v in versions if v['version'] == version), None)
            if model_metadata is None:
                raise ValueError(f"Wersja {version} nie istnieje")
        else:
            model_metadata = versions[-1]

        return model_metadata

    def delete_model(
        self,
        model_name: str,
        version: str
    ):
        """
        Usuwa model.

        Args:
            model_name: Nazwa modelu
            version: Wersja
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} nie istnieje")

        versions = self.registry['models'][model_name]
        model_metadata = next((v for v in versions if v['version'] == version), None)

        if model_metadata is None:
            raise ValueError(f"Wersja {version} nie istnieje")

        # Delete files
        model_dir = self.models_path / f"{model_name}_{version}"
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)

        # Remove from registry
        self.registry['models'][model_name] = [
            v for v in versions if v['version'] != version
        ]

        # Remove model_name if no versions left
        if not self.registry['models'][model_name]:
            del self.registry['models'][model_name]

        self._save_registry()

        logger.info(f"Model usunięty: {model_name} {version}")