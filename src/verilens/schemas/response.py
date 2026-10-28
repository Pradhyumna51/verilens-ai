"""
Pydantic schemas defining the stable JSON API contract for VeriLens AI.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class DetectorStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"

class Verdict(str, Enum):
    LIKELY_REAL = "LIKELY_REAL"
    UNCERTAIN = "UNCERTAIN"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EvidenceSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class EvidenceItem(BaseModel):
    id: str = Field(..., description="Unique evidence identifier, e.g. ev_01")
    detector_id: str = Field(..., description="Source detector identifier")
    evidence_type: str = Field(..., description="Category of evidence (e.g. FREQUENCY_ANOMALY, PERCEPTUAL_FINGERPRINT)")
    description: str = Field(..., description="Human-readable description of the observed evidence")
    severity: EvidenceSeverity = Field(default=EvidenceSeverity.LOW, description="Severity rating")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw supporting telemetry/data")

class DetectorFinding(BaseModel):
    detector_id: str = Field(..., description="Unique detector identifier")
    detector_name: str = Field(..., description="Human-friendly detector name")
    status: DetectorStatus = Field(..., description="Execution status: NOT_RUN, UNCERTAIN, ERROR, SUCCESS")
    manipulation_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Normalized score 0.0 (authentic) to 1.0 (manipulated/synthetic)"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in this detector's finding")
    summary: str = Field(..., description="Concise finding summary")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Detailed domain metrics")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")
    error_message: Optional[str] = Field(None, description="Diagnostic error if status is ERROR")

class OverallAssessment(BaseModel):
    verdict: Verdict = Field(..., description="Final overall verdict")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate confidence score")
    risk_level: RiskLevel = Field(..., description="Assessed misinformation risk level")
    summary: str = Field(..., description="Executive narrative summary")
    primary_concerns: List[str] = Field(default_factory=list, description="List of primary detected issues")
    basis_sources: List[str] = Field(default_factory=list, description="Specific factual and algorithmic sources grounding this assessment")

class Explanation(BaseModel):
    headline: str = Field(..., description="One-line explanation headline")
    rationale: str = Field(..., description="Detailed paragraph explaining the rationale")
    supporting_points: List[str] = Field(default_factory=list, description="Bullet points supporting the verdict")
    counter_evidence: List[str] = Field(default_factory=list, description="Points that mitigate concern or provide context")

class Limitations(BaseModel):
    disclaimers: List[str] = Field(default_factory=list, description="System disclaimers")
    untested_modalities: List[str] = Field(default_factory=list, description="Modalities omitted or not evaluated")
    model_assumptions: List[str] = Field(default_factory=list, description="Assumptions made by the detection models")
    human_verification_recommended: bool = Field(True, description="Whether human review is recommended")

class AnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique analysis request ID")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    processing_time_ms: float = Field(..., description="Total pipeline latency in ms")
    modalities_received: List[str] = Field(..., description="Modalities provided: ['text', 'image', 'video']")
    overall_assessment: OverallAssessment
    findings: Dict[str, DetectorFinding]
    evidence: List[EvidenceItem]
    explanation: Explanation
    limitations: Limitations

class AnalysisRequest(BaseModel):
    text: Optional[str] = Field(None, description="Caption, headline, or claim text to analyze")
    enabled_detectors: Optional[List[str]] = Field(None, description="Subset of detector IDs to run")

class DetectorMeta(BaseModel):
    id: str
    name: str
    description: str
    required_modalities: List[str]

class LimitsMeta(BaseModel):
    max_media_size_bytes: int
    max_media_size_mb: float
    max_text_chars: int
    allowed_media_types: List[str]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    limits: LimitsMeta
    detectors: List[DetectorMeta]
