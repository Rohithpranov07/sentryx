"""
app/schemas.py
Pydantic models for all API request/response contracts.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    green  = "green"
    yellow = "yellow"
    orange = "orange"
    red    = "red"


class Action(str, Enum):
    publish  = "publish"
    label    = "label"
    restrict = "restrict"
    block    = "block"


# ── /v1/analyze response ──────────────────────────────────────────────────────

class FingerprintInfo(BaseModel):
    fingerprint_id : str
    sha256         : str
    phash          : str
    dhash          : str


class LedgerInfo(BaseModel):
    registered     : bool
    fingerprint_id : str
    timestamp      : float
    ledger_type    : str
    note           : str


class AnalyzeResponse(BaseModel):
    status              : str                    # "approved" | "labeled" | "restricted" | "blocked"
    risk_level          : RiskLevel
    verdict             : str
    action              : Action
    description         : str
    color_code          : str
    confidence          : float = Field(..., ge=0.0, le=1.0)
    forensic_signals    : List[str]
    fingerprint         : FingerprintInfo
    ledger              : Optional[LedgerInfo]   # None if green/yellow (not stored)
    processing_time_ms  : float
    filename            : str
    file_size_bytes     : int
    image_dimensions    : str
    device_used         : str
    model               : str
    poc_note            : str


# ── /v1/fingerprint/{hash} response ──────────────────────────────────────────

class FingerprintLookupResponse(BaseModel):
    found          : bool
    sha256         : str
    status         : Optional[str]       # "known_threat" | "not_found"
    risk_level     : Optional[str]
    verdict        : Optional[str]
    first_seen     : Optional[float]
    fingerprint_id : Optional[str]
    ledger_type    : str


# ── /v1/health response ───────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name      : str
    version   : str
    device    : str
    status    : str


class LedgerStats(BaseModel):
    total_fingerprints : int
    by_risk_level      : dict


class HealthResponse(BaseModel):
    api              : str
    version          : str
    forensic_engine  : str
    ledger           : str
    model            : ModelInfo
    ledger_stats     : LedgerStats
    uptime_seconds   : float
