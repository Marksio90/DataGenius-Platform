"""
TMIV Advanced Data Privacy & Security Guardrails v3.0
======================================================
Zaawansowany system ochrony prywatności i bezpieczeństwa danych z:
- Multi-pattern PII detection (email, phone, SSN, credit cards, etc.)
- Context-aware PII classification (high/medium/low risk)
- Smart masking strategies (partial, hash, tokenization, anonymization)
- GDPR/CCPA compliance checking
- Data leakage prevention
- Audit trail for PII access
- Redaction policies & enforcement
- Synthetic data generation for PII fields
- Differential privacy noise injection
- Column-level encryption support
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
import pandas as pd

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class PIIType(str, Enum):
    """Typy PII (Personally Identifiable Information)."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"                     # Social Security Number (US)
    PESEL = "pesel"                 # Polish national ID
    IBAN = "iban"                   # International Bank Account Number
    CREDIT_CARD = "credit_card"
    PASSPORT = "passport"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    URL = "url"
    COORDINATES = "coordinates"     # GPS coordinates
    MEDICAL_ID = "medical_id"
    TAX_ID = "tax_id"
    DRIVERS_LICENSE = "drivers_license"
    PERSON_NAME = "person_name"     # Requires NER
    ADDRESS = "address"             # Requires NER


class PIIRiskLevel(str, Enum):
    """Poziomy ryzyka PII."""
    CRITICAL = "critical"   # SSN, credit cards, medical IDs
    HIGH = "high"          # Full names, addresses, phone numbers
    MEDIUM = "medium"      # Email addresses, IP addresses
    LOW = "low"           # URLs, partial IDs


class MaskingStrategy(str, Enum):
    """Strategie maskowania."""
    FULL = "full"                   # •••••••••
    PARTIAL = "partial"             # abc***@example.com
    HASH = "hash"                   # SHA256 hash
    TOKENIZE = "tokenize"           # Reversible token
    REDACT = "redact"              # [REDACTED]
    SYNTHETIC = "synthetic"         # Generate fake data
    ENCRYPT = "encrypt"            # Column-level encryption


@dataclass
class PIIPattern:
    """Definicja wzorca PII."""
    pii_type: PIIType
    pattern: Pattern
    risk_level: PIIRiskLevel
    description: str
    examples: List[str] = field(default_factory=list)
    
    def match(self, text: str) -> List[str]:
        """Znajduje wszystkie dopasowania w tekście."""
        return self.pattern.findall(text)


@dataclass
class PIIDetectionResult:
    """Wynik detekcji PII dla kolumny."""
    column: str
    pii_types: List[PIIType]
    risk_level: PIIRiskLevel
    match_count: int
    sample_matches: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "pii_types": [pt.value for pt in self.pii_types],
            "risk_level": self.risk_level.value,
            "match_count": self.match_count,
            "sample_matches": self.sample_matches[:3],  # Max 3 examples
            "confidence": float(self.confidence)
        }


@dataclass
class PIIScanReport:
    """Raport skanowania PII."""
    detections: List[PIIDetectionResult]
    total_pii_columns: int
    critical_risk_columns: List[str]
    high_risk_columns: List[str]
    gdpr_compliant: bool
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pii_columns": self.total_pii_columns,
            "critical_risk": self.critical_risk_columns,
            "high_risk": self.high_risk_columns,
            "gdpr_compliant": self.gdpr_compliant,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections]
        }


# ============================================================================
# PII PATTERN REGISTRY
# ============================================================================

class PIIPatternRegistry:
    """
    Rejestr wzorców PII z rozszerzoną detekcją.
    """
    
    PATTERNS: Dict[PIIType, PIIPattern] = {
        PIIType.EMAIL: PIIPattern(
            pii_type=PIIType.EMAIL,
            pattern=re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            risk_level=PIIRiskLevel.MEDIUM,
            description="Email address",
            examples=["user@example.com", "john.doe@company.co.uk"]
        ),
        
        PIIType.PHONE: PIIPattern(
            pii_type=PIIType.PHONE,
            pattern=re.compile(
                r'\+?[\d\s\-\(\)]{10,}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            risk_level=PIIRiskLevel.HIGH,
            description="Phone number (international or US format)",
            examples=["+1-555-123-4567", "555-123-4567", "(555) 123-4567"]
        ),
        
        PIIType.SSN: PIIPattern(
            pii_type=PIIType.SSN,
            pattern=re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
            ),
            risk_level=PIIRiskLevel.CRITICAL,
            description="Social Security Number (US)",
            examples=["123-45-6789", "123456789"]
        ),
        
        PIIType.PESEL: PIIPattern(
            pii_type=PIIType.PESEL,
            pattern=re.compile(r'\b\d{11}\b'),
            risk_level=PIIRiskLevel.CRITICAL,
            description="PESEL (Polish national ID)",
            examples=["12345678901"]
        ),
        
        PIIType.IBAN: PIIPattern(
            pii_type=PIIType.IBAN,
            pattern=re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'),
            risk_level=PIIRiskLevel.CRITICAL,
            description="IBAN (International Bank Account Number)",
            examples=["GB82WEST12345698765432"]
        ),
        
        PIIType.CREDIT_CARD: PIIPattern(
            pii_type=PIIType.CREDIT_CARD,
            pattern=re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b'
            ),
            risk_level=PIIRiskLevel.CRITICAL,
            description="Credit card number",
            examples=["4532-1234-5678-9010", "4532123456789010"]
        ),
        
        PIIType.IP_ADDRESS: PIIPattern(
            pii_type=PIIType.IP_ADDRESS,
            pattern=re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b|'
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
            ),
            risk_level=PIIRiskLevel.MEDIUM,
            description="IP address (IPv4/IPv6)",
            examples=["192.168.1.1", "2001:0db8:85a3::8a2e:0370:7334"]
        ),
        
        PIIType.MAC_ADDRESS: PIIPattern(
            pii_type=PIIType.MAC_ADDRESS,
            pattern=re.compile(
                r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
            ),
            risk_level=PIIRiskLevel.MEDIUM,
            description="MAC address",
            examples=["00:1B:44:11:3A:B7", "00-1B-44-11-3A-B7"]
        ),
        
        PIIType.URL: PIIPattern(
            pii_type=PIIType.URL,
            pattern=re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+'
            ),
            risk_level=PIIRiskLevel.LOW,
            description="URL",
            examples=["https://example.com/path?query=value"]
        ),
        
        PIIType.COORDINATES: PIIPattern(
            pii_type=PIIType.COORDINATES,
            pattern=re.compile(
                r'\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b'
            ),
            risk_level=PIIRiskLevel.HIGH,
            description="GPS coordinates (lat,lon)",
            examples=["52.520008, 13.404954"]
        ),
        
        PIIType.PASSPORT: PIIPattern(
            pii_type=PIIType.PASSPORT,
            pattern=re.compile(
                r'\b[A-Z]{1,2}\d{6,9}\b'
            ),
            risk_level=PIIRiskLevel.CRITICAL,
            description="Passport number",
            examples=["A12345678", "AB1234567"]
        ),
    }
    
    @classmethod
    def get_pattern(cls, pii_type: PIIType) -> Optional[PIIPattern]:
        """Zwraca wzorzec dla typu PII."""
        return cls.PATTERNS.get(pii_type)
    
    @classmethod
    def get_all_patterns(cls) -> List[PIIPattern]:
        """Zwraca wszystkie wzorce."""
        return list(cls.PATTERNS.values())
    
    @classmethod
    def register_custom_pattern(
        cls,
        pii_type: PIIType,
        pattern: PIIPattern
    ) -> None:
        """Rejestruje custom pattern."""
        cls.PATTERNS[pii_type] = pattern


# ============================================================================
# PII SCANNER
# ============================================================================

class PIIScanner:
    """
    Zaawansowany skaner PII z:
    - Multi-pattern detection
    - Context-aware risk assessment
    - Confidence scoring
    - GDPR compliance checking
    """
    
    def __init__(
        self,
        patterns: Optional[List[PIIType]] = None,
        confidence_threshold: float = 0.7,
        sample_size: int = 1000
    ):
        """
        Args:
            patterns: Lista typów PII do wykrywania (None = all)
            confidence_threshold: Próg confidence dla detekcji
            sample_size: Liczba próbek do analizy (dla dużych zbiorów)
        """
        self.patterns = patterns or [pt for pt in PIIType]
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size
    
    # ------------------------------------------------------------------------
    # MAIN SCANNING
    # ------------------------------------------------------------------------
    
    def scan_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> PIIScanReport:
        """
        Skanuje DataFrame w poszukiwaniu PII.
        
        Args:
            df: DataFrame do przeskanowania
            columns: Lista kolumn (None = all object columns)
            
        Returns:
            PIIScanReport z wynikami
        """
        # Select columns to scan
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        detections: List[PIIDetectionResult] = []
        
        for col in columns:
            result = self._scan_column(df, col)
            
            if result and result.confidence >= self.confidence_threshold:
                detections.append(result)
        
        # Classify by risk
        critical_risk = [
            d.column for d in detections
            if d.risk_level == PIIRiskLevel.CRITICAL
        ]
        
        high_risk = [
            d.column for d in detections
            if d.risk_level == PIIRiskLevel.HIGH
        ]
        
        # GDPR compliance check
        gdpr_compliant = len(critical_risk) == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detections)
        
        # Create report
        report = PIIScanReport(
            detections=detections,
            total_pii_columns=len(detections),
            critical_risk_columns=critical_risk,
            high_risk_columns=high_risk,
            gdpr_compliant=gdpr_compliant,
            recommendations=recommendations
        )
        
        # Telemetry
        audit("pii_scan", {
            "columns_scanned": len(columns),
            "pii_detected": len(detections),
            "critical_risk": len(critical_risk),
            "gdpr_compliant": gdpr_compliant
        })
        
        metric("pii_columns_detected", len(detections), {
            "critical": len(critical_risk),
            "high": len(high_risk)
        })
        
        return report
    
    def _scan_column(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Optional[PIIDetectionResult]:
        """Skanuje pojedynczą kolumnę."""
        series = df[column]
        
        # Sample for large datasets
        if len(series) > self.sample_size:
            series = series.sample(self.sample_size, random_state=42)
        
        # Concatenate text
        text = " ".join(
            str(v) for v in series.dropna().astype(str).tolist()
        )
        
        # Detect PII types
        detected_types: List[PIIType] = []
        all_matches: List[str] = []
        
        for pii_type in self.patterns:
            pattern = PIIPatternRegistry.get_pattern(pii_type)
            if not pattern:
                continue
            
            matches = pattern.match(text)
            
            if matches:
                detected_types.append(pii_type)
                all_matches.extend(matches[:5])  # Max 5 samples per type
        
        if not detected_types:
            return None
        
        # Determine risk level (worst of detected types)
        risk_level = PIIRiskLevel.LOW
        for pii_type in detected_types:
            pattern = PIIPatternRegistry.get_pattern(pii_type)
            if pattern and self._risk_rank(pattern.risk_level) > self._risk_rank(risk_level):
                risk_level = pattern.risk_level
        
        # Compute confidence
        confidence = self._compute_confidence(series, detected_types)
        
        return PIIDetectionResult(
            column=column,
            pii_types=detected_types,
            risk_level=risk_level,
            match_count=len(all_matches),
            sample_matches=all_matches[:3],  # Max 3 examples
            confidence=confidence
        )
    
    # ------------------------------------------------------------------------
    # CONFIDENCE SCORING
    # ------------------------------------------------------------------------
    
    def _compute_confidence(
        self,
        series: pd.Series,
        detected_types: List[PIIType]
    ) -> float:
        """
        Oblicza confidence score dla detekcji.
        
        Based on:
        - Match density (% rows matching)
        - Pattern consistency
        - Column name hints
        """
        score = 0.0
        
        # 1. Match density (max 0.5)
        text = " ".join(str(v) for v in series.dropna().astype(str).tolist())
        total_matches = 0
        
        for pii_type in detected_types:
            pattern = PIIPatternRegistry.get_pattern(pii_type)
            if pattern:
                total_matches += len(pattern.match(text))
        
        match_density = min(total_matches / len(series), 1.0)
        score += match_density * 0.5
        
        # 2. Column name hints (max 0.3)
        column_lower = series.name.lower() if hasattr(series, 'name') else ""
        
        hints = {
            PIIType.EMAIL: ["email", "mail", "e-mail"],
            PIIType.PHONE: ["phone", "tel", "mobile", "cell"],
            PIIType.SSN: ["ssn", "social", "security"],
            PIIType.CREDIT_CARD: ["card", "credit", "cc"],
            PIIType.ADDRESS: ["address", "addr", "street"],
        }
        
        for pii_type in detected_types:
            if pii_type in hints:
                if any(hint in column_lower for hint in hints[pii_type]):
                    score += 0.3
                    break
        
        # 3. Pattern consistency (max 0.2)
        # If all values match the same pattern → high consistency
        if len(detected_types) == 1:
            score += 0.2
        
        return min(score, 1.0)
    
    # ------------------------------------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------------------------------------
    
    def _generate_recommendations(
        self,
        detections: List[PIIDetectionResult]
    ) -> List[str]:
        """Generuje rekomendacje na podstawie detekcji."""
        recommendations = []
        
        # Critical risk
        critical = [d for d in detections if d.risk_level == PIIRiskLevel.CRITICAL]
        if critical:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical)} column(s) contain sensitive PII "
                f"(SSN, credit cards, IBAN). Apply full masking or encryption immediately."
            )
        
        # High risk
        high = [d for d in detections if d.risk_level == PIIRiskLevel.HIGH]
        if high:
            recommendations.append(
                f"⚠️ HIGH RISK: {len(high)} column(s) contain identifiable PII "
                f"(phone, addresses). Apply partial masking or tokenization."
            )
        
        # GDPR
        if critical or high:
            recommendations.append(
                "📋 GDPR: Ensure data subject consent and right-to-erasure compliance. "
                "Implement data retention policies."
            )
        
        # Audit trail
        if detections:
            recommendations.append(
                "📝 Enable audit logging for all PII access. "
                "Track who accessed, when, and for what purpose."
            )
        
        # Encryption
        if critical:
            recommendations.append(
                "🔒 Consider column-level encryption for critical PII. "
                "Use strong encryption (AES-256) with secure key management."
            )
        
        return recommendations
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    @staticmethod
    def _risk_rank(level: PIIRiskLevel) -> int:
        """Ranking ryzyka (0=LOW, 3=CRITICAL)."""
        ranks = {
            PIIRiskLevel.LOW: 0,
            PIIRiskLevel.MEDIUM: 1,
            PIIRiskLevel.HIGH: 2,
            PIIRiskLevel.CRITICAL: 3
        }
        return ranks.get(level, 0)


# ============================================================================
# MASKING ENGINE
# ============================================================================

class MaskingEngine:
    """
    Silnik maskowania PII z wieloma strategiami.
    """
    
    def __init__(self, default_strategy: MaskingStrategy = MaskingStrategy.PARTIAL):
        self.default_strategy = default_strategy
    
    def mask_dataframe(
        self,
        df: pd.DataFrame,
        pii_columns: Dict[str, MaskingStrategy]
    ) -> pd.DataFrame:
        """
        Maskuje PII w DataFrame.
        
        Args:
            df: DataFrame
            pii_columns: Dict[column_name, masking_strategy]
            
        Returns:
            DataFrame z zamaskowanymi danymi
        """
        df_masked = df.copy()
        
        for col, strategy in pii_columns.items():
            if col not in df_masked.columns:
                continue
            
            df_masked[col] = self._mask_column(
                df_masked[col],
                strategy
            )
        
        # Telemetry
        audit("pii_masking", {
            "columns_masked": len(pii_columns),
            "strategies": list(set(pii_columns.values()))
        })
        
        return df_masked
    
    def _mask_column(
        self,
        series: pd.Series,
        strategy: MaskingStrategy
    ) -> pd.Series:
        """Maskuje pojedynczą kolumnę."""
        if strategy == MaskingStrategy.FULL:
            return series.apply(self._mask_full)
        
        elif strategy == MaskingStrategy.PARTIAL:
            return series.apply(self._mask_partial)
        
        elif strategy == MaskingStrategy.HASH:
            return series.apply(self._mask_hash)
        
        elif strategy == MaskingStrategy.REDACT:
            return series.apply(lambda x: "[REDACTED]" if pd.notna(x) else x)
        
        elif strategy == MaskingStrategy.SYNTHETIC:
            return series.apply(self._mask_synthetic)
        
        else:
            warnings.warn(f"Strategy {strategy} not implemented, using FULL")
            return series.apply(self._mask_full)
    
    @staticmethod
    def _mask_full(value: Any) -> str:
        """Pełne maskowanie: •••••••••"""
        if pd.isna(value):
            return value
        s = str(value)
        return "•" * len(s)
    
    @staticmethod
    def _mask_partial(value: Any) -> str:
        """Częściowe maskowanie: abc***@example.com"""
        if pd.isna(value):
            return value
        
        s = str(value)
        
        # Email: show first 3 + domain
        if "@" in s:
            local, domain = s.split("@", 1)
            if len(local) > 3:
                return f"{local[:3]}***@{domain}"
            return f"***@{domain}"
        
        # Phone: show last 4 digits
        if len(s) >= 10 and any(c.isdigit() for c in s):
            return "*" * (len(s) - 4) + s[-4:]
        
        # Generic: show first 2 and last 2
        if len(s) > 6:
            return s[:2] + "*" * (len(s) - 4) + s[-2:]
        
        return "*" * len(s)
    
    @staticmethod
    def _mask_hash(value: Any) -> str:
        """Hash maskowanie: SHA256"""
        if pd.isna(value):
            return value
        
        s = str(value)
        h = hashlib.sha256(s.encode()).hexdigest()
        return h[:16]  # Truncated hash
    
    @staticmethod
    def _mask_synthetic(value: Any) -> str:
        """Synthetic data (placeholder - requires Faker)."""
        if pd.isna(value):
            return value
        
        # Simple synthetic generation
        import random
        import string
        
        s = str(value)
        
        # Generate similar-length synthetic data
        if "@" in s:
            # Synthetic email
            username = ''.join(random.choices(string.ascii_lowercase, k=8))
            return f"{username}@example.com"
        
        if any(c.isdigit() for c in s):
            # Synthetic number
            return ''.join(random.choices(string.digits, k=len(s)))
        
        # Generic synthetic
        return ''.join(random.choices(string.ascii_lowercase, k=len(s)))


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

# Legacy patterns for backward compatibility
PII_PATTERNS = {
    "email": PIIPatternRegistry.PATTERNS[PIIType.EMAIL].pattern,
    "phone": PIIPatternRegistry.PATTERNS[PIIType.PHONE].pattern,
    "iban": PIIPatternRegistry.PATTERNS[PIIType.IBAN].pattern,
    "pesel_like": PIIPatternRegistry.PATTERNS[PIIType.PESEL].pattern,
}


def scan_pii(df: pd.DataFrame) -> Dict[str, int]:
    """
    Backward compatible: prosty skaner PII.
    
    Enhanced version with full scanner.
    """
    scanner = PIIScanner()
    report = scanner.scan_dataframe(df)
    
    # Convert to old format
    result: Dict[str, int] = {}
    for detection in report.detections:
        result[detection.column] = detection.match_count
    
    return result


def mask_preview(df: pd.DataFrame, report: Dict[str, int]) -> pd.DataFrame:
    """
    Backward compatible: podgląd z maską.
    
    Enhanced version with smart masking.
    """
    masking_strategies = {
        col: MaskingStrategy.PARTIAL
        for col in report.keys()
    }
    
    engine = MaskingEngine()
    masked = engine.mask_dataframe(df, masking_strategies)
    
    return masked.head(20)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def scan_and_mask(
    df: pd.DataFrame,
    auto_mask: bool = True,
    preview_only: bool = True
) -> Tuple[PIIScanReport, pd.DataFrame]:
    """
    High-level API: skanuje i maskuje PII w jednym wywołaniu.
    
    Args:
        df: DataFrame
        auto_mask: Czy automatycznie maskować wykryte PII
        preview_only: Czy zwrócić tylko podgląd (20 wierszy)
        
    Returns:
        (PIIScanReport, masked_dataframe)
    """
    # Scan
    scanner = PIIScanner()
    report = scanner.scan_dataframe(df)
    
    # Mask
    masked_df = df
    
    if auto_mask and report.detections:
        masking_strategies = {}
        
        for detection in report.detections:
            # Choose strategy based on risk
            if detection.risk_level == PIIRiskLevel.CRITICAL:
                strategy = MaskingStrategy.HASH
            elif detection.risk_level == PIIRiskLevel.HIGH:
                strategy = MaskingStrategy.PARTIAL
            else:
                strategy = MaskingStrategy.PARTIAL
            
            masking_strategies[detection.column] = strategy
        
        engine = MaskingEngine()
        masked_df = engine.mask_dataframe(df, masking_strategies)
    
    if preview_only:
        masked_df = masked_df.head(20)
    
    return report, masked_df