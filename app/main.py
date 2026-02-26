"""
app/main.py

SENTRY-X FastAPI Application
Real-Time Media Integrity Firewall

Endpoints:
  POST /v1/analyze              → Submit media for deepfake detection
  GET  /v1/fingerprint/{sha256} → Query provenance ledger
  GET  /v1/health               → System health + model status
  GET  /                        → API info
"""

import time
import io
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from app.schemas import (
    AnalyzeResponse, FingerprintLookupResponse, HealthResponse,
    FingerprintInfo, LedgerInfo, ModelInfo, LedgerStats
)
from models.detector import detector, classify_risk
from utils.fingerprint import generate_fingerprints
from utils.ledger import init_ledger, lookup_sha256, lookup_similar, register_threat, get_stats
from utils.device import DEVICE

from app.video_routes import router as video_router
from app.v2_routes import v2_router

# ── Startup ───────────────────────────────────────────────────────────────────
init_ledger()
START_TIME = time.time()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE", 52428800))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SENTRY-X API",
    description=(
        "Real-Time Media Integrity Firewall. "
        "Detects deepfakes, verifies authenticity, and permanently registers threats."
    ),
    version=os.getenv("APP_VERSION", "0.1.0"),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/heic", "image/heif"}

# Optional soft-registration of HEIF plugins
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(video_router, prefix="/v1/analyze", tags=["Video Detection"])
app.include_router(v2_router, prefix="/v2", tags=["V2 Pipeline"])

@app.get("/", tags=["Info"])
async def root():
    return {
        "system": "SENTRY-X",
        "tagline": "Real-Time Media Integrity Firewall",
        "version": os.getenv("APP_VERSION", "0.1.0"),
        "endpoints": {
            "analyze":       "POST /v1/analyze",
            "analyze_video": "POST /v1/analyze/video",
            "analyze_v2":    "POST /v2/analyze",
            "fingerprint":   "GET  /v1/fingerprint/{sha256}",
            "health":        "GET  /v1/health",
            "docs":          "GET  /docs",
        },
        "status": "operational",
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["Detection"])
async def analyze(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WebP, BMP)"),
    platform_id: str = Form(default="unknown", description="Originating platform identifier"),
):
    """
    ## Analyze media for deepfake manipulation

    Runs the full SENTRY-X pipeline:
    1. Validate and fingerprint the upload
    2. Check provenance ledger for known threats (fast-path)
    3. Run AI forensic detection (V4 Proven Ensemble)
    4. Classify into risk tier (🟢🟡🟠🔴)
    5. Register RED/ORANGE verdicts to permanent ledger
    6. Return structured verdict with forensic signals

    **Supported formats:** JPEG, PNG, WebP, BMP  
    **Max file size:** 50MB  
    **Typical latency:** 100–3000ms depending on device
    """
    t_start = time.time()

    # ── 1. Validate upload ────────────────────────────────────────────────────
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Supported: {SUPPORTED_TYPES}"
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_UPLOAD_BYTES // 1048576}MB"
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── 2. Parse image ────────────────────────────────────────────────────────
    try:
        # If pillow_heif is registered, Image.open works seamlessly.
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")
        w, h = image.size
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    # ── 3. Generate fingerprints ──────────────────────────────────────────────
    fingerprints = generate_fingerprints(file_bytes, image)

    # ── 4. Ledger fast-path: exact match ─────────────────────────────────────
    known = lookup_sha256(fingerprints["sha256"])
    if known and known["risk_level"] in ("red", "orange"):
        t_end = time.time()
        processing_ms = (t_end - t_start) * 1000

        risk = {
            "risk_level": known["risk_level"],
            "verdict": known["verdict"],
            "action": "block" if known["risk_level"] == "red" else "restrict",
            "color_code": "#ef4444" if known["risk_level"] == "red" else "#f97316",
            "description": "⚡ Fast-path: Known threat fingerprint matched in provenance ledger.",
        }

        return AnalyzeResponse(
            status="blocked_known_threat",
            risk_level=risk["risk_level"],
            verdict=risk["verdict"],
            action=risk["action"],
            description=risk["description"],
            color_code=risk["color_code"],
            confidence=known["confidence"],
            forensic_signals=["Known threat — fingerprint matched in provenance ledger", "No re-analysis required"],
            fingerprint=FingerprintInfo(**fingerprints),
            ledger=LedgerInfo(
                registered=True,
                fingerprint_id=fingerprints["fingerprint_id"],
                timestamp=known["timestamp"],
                ledger_type="local_provenance_node",
                note="Previously registered threat. Instant block — no AI inference needed.",
            ),
            processing_time_ms=round(processing_ms, 2),
            filename=file.filename,
            file_size_bytes=len(file_bytes),
            image_dimensions=f"{w}x{h}",
            device_used=str(DEVICE),
            model="ledger_fast_path",
            poc_note="PoC ledger: SQLite. Production: Polygon L2 smart contract.",
        )

    # ── 5. Ledger fast-path: perceptual similarity ────────────────────────────
    similar = lookup_similar(fingerprints["phash"])
    similarity_note = None
    if similar:
        similarity_note = (
            f"Near-duplicate of known threat detected "
            f"(Hamming distance: {similar['hamming_distance']}, threshold: 10). "
            f"Original threat ID: {similar['fingerprint_id']}"
        )

    # ── 6. AI forensic detection ──────────────────────────────────────────────
    confidence, forensic_signals = detector.predict(image)

    # Boost confidence if perceptual similarity to known threat found
    if similarity_note:
        confidence = min(1.0, confidence + 0.25)
        forensic_signals.insert(0, similarity_note)

    # ── 7. Risk classification ────────────────────────────────────────────────
    risk = classify_risk(confidence)

    # Map risk action to API status
    status_map = {
        "publish":  "approved",
        "label":    "labeled",
        "restrict": "restricted",
        "block":    "blocked",
    }

    # ── 8. Register threat to ledger if RED or ORANGE ─────────────────────────
    ledger_entry = None
    if risk["risk_level"] in ("red", "orange"):
        ledger_entry_raw = register_threat(
            fingerprints=fingerprints,
            verdict={**risk, "confidence": confidence, "forensic_signals": forensic_signals},
            filename=file.filename,
            platform_id=platform_id,
        )
        ledger_entry = LedgerInfo(**ledger_entry_raw)

    # ── 9. Build response ─────────────────────────────────────────────────────
    t_end = time.time()
    processing_ms = (t_end - t_start) * 1000

    return AnalyzeResponse(
        status=status_map[risk["action"]],
        risk_level=risk["risk_level"],
        verdict=risk["verdict"],
        action=risk["action"],
        description=risk["description"],
        color_code=risk["color_code"],
        confidence=round(confidence, 4),
        forensic_signals=forensic_signals,
        fingerprint=FingerprintInfo(**fingerprints),
        ledger=ledger_entry,
        processing_time_ms=round(processing_ms, 2),
        filename=file.filename,
        file_size_bytes=len(file_bytes),
        image_dimensions=f"{w}x{h}",
        device_used=str(DEVICE),
        model="V4 Proven Pipeline Ensemble",
        poc_note=(
            "V1 legacy endpoint updated to report V4 model availability."
        ),
    )


@app.get("/v1/fingerprint/{sha256}", response_model=FingerprintLookupResponse, tags=["Provenance"])
async def lookup_fingerprint(sha256: str):
    """
    ## Query the provenance ledger by SHA256 fingerprint

    Returns known threat status if the fingerprint exists in the ledger.
    Used by platforms to check a file before running full AI analysis.
    """
    if len(sha256) != 64:
        raise HTTPException(status_code=400, detail="SHA256 must be 64 hex characters.")

    entry = lookup_sha256(sha256)

    if not entry:
        return FingerprintLookupResponse(
            found=False,
            sha256=sha256,
            status="not_found",
            risk_level=None,
            verdict=None,
            first_seen=None,
            fingerprint_id=None,
            ledger_type="local_provenance_node",
        )

    return FingerprintLookupResponse(
        found=True,
        sha256=sha256,
        status="known_threat",
        risk_level=entry["risk_level"],
        verdict=entry["verdict"],
        first_seen=entry["timestamp"],
        fingerprint_id=entry["fingerprint_id"],
        ledger_type="local_provenance_node",
    )


@app.get("/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    ## System health check

    Returns model status, ledger stats, and uptime.
    """
    ledger_stats = get_stats()

    return HealthResponse(
        api="healthy",
        version=os.getenv("APP_VERSION", "0.1.0"),
        forensic_engine="healthy",
        ledger="healthy",
        model=ModelInfo(
            name="V4 Proven Ensemble",
            version="ViT / XceptionNet / GAN FFT",
            device=str(DEVICE),
            status="loaded",
        ),
        ledger_stats=LedgerStats(
            total_fingerprints=ledger_stats["total_fingerprints"],
            by_risk_level=ledger_stats["by_risk_level"],
        ),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )
