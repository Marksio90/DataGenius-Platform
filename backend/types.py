
from __future__ import annotations
from typing import Literal, TypedDict, Optional, Dict, List
from pydantic import BaseModel, Field

Stage = Literal["upload","sanity","contracts","eda","plan","train","results","export"]

class UiNotice(BaseModel):
    """Powiadomienie dla UI."""
    code: str = Field(..., description="Kod TMIV-XXX")
    level: Literal["INFO","WARN","ERROR"]
    message_pl: str
    details: Optional[Dict] = None

class ProgressEvent(BaseModel):
    """Zdarzenie postępu dla live timeline."""
    run_id: str
    stage: Stage
    pct: float = Field(ge=0.0, le=100.0)
    note_pl: Optional[str] = None

class ArtifactRef(BaseModel):
    """Wpis do indeksu artefaktów."""
    kind: Literal["model","metrics","plot","pdf","sbom","manifest","baseline","threshold"]
    path: str
    sha256: str

class ThresholdManifest(BaseModel):
    """Manifest wybranego progu decyzyjnego."""
    run_id: str
    version: str
    threshold: float
    objective: Literal["youden","cost"]
    cost_matrix: Optional[Dict[str, float]] = None
    metrics_at_threshold: Dict[str, float]

class DataQualityBaseline(BaseModel):
    """Baseline jakości danych."""
    run_id: str
    schema_hash: str
    missingness: Dict[str, float]
    cardinality: Dict[str, int]
    dtypes: Dict[str, str]
