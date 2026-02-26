"""
app/video_routes.py

SENTRY-X Video Analysis API Routes

Mounted onto the main FastAPI app as a router.
Adds two endpoints:
  POST /v1/analyze/video   → Full video analysis pipeline
  GET  /v1/analyze/video/supported → Supported formats + limits

Designed to be imported and included in app/main.py.
"""

import io
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from models.proven_ensemble import get_proven_ensemble
from video.processor import extract_frames, get_video_metadata, frames_to_jpeg_bytes
from video.analyzer import FrameResult, aggregate_frame_results, build_timeline_data
from utils.fingerprint import generate_fingerprints, sha256_file
from utils.ledger import lookup_sha256, register_threat

router = APIRouter()

SUPPORTED_VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska",
}

MAX_VIDEO_SIZE = int(os.getenv("MAX_VIDEO_SIZE", 200 * 1024 * 1024))  # 200MB
DEFAULT_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1.0"))
MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "60"))


# ── Response schemas ──────────────────────────────────────────────────────────

class FrameResultSchema(BaseModel):
    frame_index      : int
    timestamp_s      : float
    confidence       : float
    risk_level       : str
    verdict          : str
    action           : str
    forensic_signals : List[str]


class VideoVerdictSchema(BaseModel):
    # Overall verdict
    risk_level           : str
    verdict              : str
    action               : str
    description          : str
    color_code           : str

    # Scores
    max_confidence       : float
    mean_confidence      : float
    manipulation_ratio   : float

    # Key moments
    peak_frame_index     : int
    peak_timestamp_s     : float
    flagged_timestamps   : List[float]

    # Frame summary
    total_frames_analyzed: int
    frame_counts         : dict

    # Temporal analysis
    temporal_consistency : str
    temporal_note        : str

    # Processing
    processing_time_ms   : float
    frames_per_second_sampled: float


class VideoAnalysisResponse(BaseModel):
    # File info
    filename             : str
    file_size_bytes      : int

    # Video metadata
    video_metadata       : dict

    # Per-frame results
    frame_results        : List[FrameResultSchema]
    timeline_data        : List[dict]

    # Aggregated verdict
    verdict              : VideoVerdictSchema

    # Fingerprint + ledger
    sha256               : str
    fingerprint_id       : str
    ledger_registered    : bool
    ledger_note          : str

    # Config used
    sample_fps           : float
    max_frames_cap       : int
    device_used          : str
    model                : str
    poc_note             : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/video", response_model=VideoAnalysisResponse, tags=["Video Detection"])
async def analyze_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI, WebM)"),
    platform_id: str = Form(default="unknown"),
    sample_fps: float = Form(default=1.0, description="Frames per second to sample (0.5–5.0)"),
):
    """
    ## Analyze a video file for deepfake manipulation

    Runs the SENTRY-X pipeline on a video by:
    1. Extracting frames at `sample_fps` frames per second
    2. Running AI forensic detection on each frame
    3. Aggregating results into a video-level verdict
    4. Identifying exact timestamps where manipulation occurs
    5. Registering RED/ORANGE videos to the provenance ledger

    **Supported formats:** MP4, MOV, AVI, WebM, MKV  
    **Max file size:** 200MB  
    **Default sampling:** 1 frame/second (max 60 frames)  
    **Tip:** Increase `sample_fps` to 2–3 for shorter videos for more thorough analysis.
    """
    t_start = time.time()

    # ── Validate ──────────────────────────────────────────────────────────────
    content_type = file.content_type or ""
    filename     = file.filename or "upload.mp4"

    # Accept by content type OR by extension (browsers sometimes misreport)
    ext = Path(filename).suffix.lower()
    valid_ext = ext in {".mp4", ".mov", ".avi", ".webm", ".mkv"}

    if content_type not in SUPPORTED_VIDEO_TYPES and not valid_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type: {content_type}. Use MP4, MOV, AVI, WebM, or MKV."
        )

    # Clamp sample_fps to safe range
    sample_fps = max(0.5, min(5.0, sample_fps))

    file_bytes = await file.read()
    file_size  = len(file_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")
    if file_size > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size // 1048576}MB). Max: {MAX_VIDEO_SIZE // 1048576}MB"
        )

    # ── Fingerprint the raw video file ────────────────────────────────────────
    video_sha256 = sha256_file(file_bytes)
    fp_id        = f"VFP-{video_sha256[:12].upper()}"

    known = lookup_sha256(video_sha256)
    if known and known["risk_level"] in ("red", "orange"):
        t_end = time.time()
        # Build valid schema response for known threat to bypass extracting frames
        return VideoAnalysisResponse(
            filename         = filename,
            file_size_bytes  = file_size,
            video_metadata   = {
                "fps": 0, "total_frames": 0, "duration_seconds": 0, 
                "resolution": "Unknown", "width": 0, "height": 0
            },
            frame_results    = [],
            timeline_data    = [],
            verdict          = VideoVerdictSchema(
                risk_level            = known["risk_level"],
                verdict               = known["verdict"],
                action                = "block" if known["risk_level"] == "red" else "restrict",
                description           = "⚡ Fast-path: Known threat exact match in PROVENANCE LEDGER. Content explicitly blocked; no frames require analysis.",
                color_code            = "#ef4444" if known["risk_level"] == "red" else "#f97316",
                max_confidence        = known.get("confidence", 1.0),
                mean_confidence       = known.get("confidence", 1.0),
                manipulation_ratio    = 1.0,
                peak_frame_index      = 0,
                peak_timestamp_s      = 0.0,
                flagged_timestamps    = [],
                total_frames_analyzed = 0,
                frame_counts          = {known["risk_level"]: 1},
                temporal_consistency  = "known_threat",
                temporal_note         = "Identified via cryptographic hash in ledger.",
                processing_time_ms    = round((t_end - t_start) * 1000, 2),
                frames_per_second_sampled = sample_fps,
            ),
            sha256           = video_sha256,
            fingerprint_id   = fp_id,
            ledger_registered= True,
            ledger_note      = "Detected natively via ledger match.",
            sample_fps       = sample_fps,
            max_frames_cap   = MAX_FRAMES,
            device_used      = "ledger_fast_path",
            model            = "SHA-256 Match",
            poc_note         = "Ledger fast-path triggered natively. Zero inference cost."
        )

    # ── Write to temp file (OpenCV needs a file path) ─────────────────────────
    with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── Get video metadata ────────────────────────────────────────────────
        try:
            meta = get_video_metadata(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read video: {e}")

        # ── Extract and analyze frames ────────────────────────────────────────
        frame_results: List[FrameResult] = []
        extracted_data = []

        for frame_idx, timestamp, pil_image in extract_frames(
            tmp_path,
            sample_fps=sample_fps,
            max_frames=MAX_FRAMES,
        ):
            extracted_data.append((frame_idx, timestamp, pil_image))
            
        if not extracted_data:
            raise HTTPException(status_code=422, detail="Could not extract any frames from video.")
            
        t_extract_done = time.time()
            
        # ── Frame Inference using Proven Ensemble ──
        pil_images = [ed[2] for ed in extracted_data]
        proven_detector = get_proven_ensemble()
        
        for (frame_idx, timestamp, _), pil_image in zip(extracted_data, pil_images):
            result = proven_detector.predict(pil_image)
            raw = result["raw_scores"]
            frame_results.append(FrameResult(
                frame_index      = frame_idx,
                timestamp_s      = timestamp,
                confidence       = result["probability"],
                risk_level       = result["risk_level"],
                verdict          = result["verdict"],
                action           = result["action"],
                forensic_signals = [
                    f"ViT Deepfake P(fake)  = {raw.get('vit_deepfake_v2', 0):.4f}",
                    f"Gemini 2.5 Online     = {raw.get('gemini_online_detector', 0):.4f}",
                    f"XceptionNet forensic  = {raw.get('xception_forensic', 0):.4f}",
                    f"GAN fingerprint (FFT) = {raw.get('gan_fingerprint_fft', 0):.4f}",
                ],
            ))

        if not frame_results:
            raise HTTPException(status_code=422, detail="Could not extract any frames from video.")

        # ── Aggregate verdict ─────────────────────────────────────────────────
        t_inference_done = time.time()
        processing_ms    = (t_inference_done - t_start) * 1000

        verdict = aggregate_frame_results(
            frame_results  = frame_results,
            sample_fps     = sample_fps,
            processing_time_ms = processing_ms,
        )

        # ── Register to ledger if RED or ORANGE ──────────────────────────────
        ledger_registered = False
        ledger_note       = "Green/Yellow verdict — not registered to ledger."

        if verdict.risk_level in ("red", "orange"):
            register_threat(
                fingerprints = {
                    "sha256"         : video_sha256,
                    "phash"          : "video_" + video_sha256[:16],
                    "dhash"          : "video_" + video_sha256[16:32],
                    "fingerprint_id" : fp_id,
                },
                verdict = {
                    "risk_level"       : verdict.risk_level,
                    "verdict"          : verdict.verdict,
                    "action"           : verdict.action,
                    "color_code"       : verdict.color_code,
                    "confidence"       : verdict.max_confidence,
                    "forensic_signals" : [verdict.temporal_note],
                },
                filename    = filename,
                platform_id = platform_id,
            )
            ledger_registered = True
            ledger_note = (
                f"Registered to provenance ledger as {fp_id}. "
                "Production: Polygon L2 smart contract write."
            )

        # ── Build timeline data ───────────────────────────────────────────────
        timeline = build_timeline_data(frame_results)

        # ── Device info ───────────────────────────────────────────────────────
        from utils.device import DEVICE

        return VideoAnalysisResponse(
            filename         = filename,
            file_size_bytes  = file_size,
            video_metadata   = meta,
            frame_results    = [
                FrameResultSchema(
                    frame_index      = fr.frame_index,
                    timestamp_s      = fr.timestamp_s,
                    confidence       = round(fr.confidence, 4),
                    risk_level       = fr.risk_level,
                    verdict          = fr.verdict,
                    action           = fr.action,
                    forensic_signals = fr.forensic_signals,
                )
                for fr in frame_results
            ],
            timeline_data    = timeline,
            verdict          = VideoVerdictSchema(
                risk_level            = verdict.risk_level,
                verdict               = verdict.verdict,
                action                = verdict.action,
                description           = verdict.description,
                color_code            = verdict.color_code,
                max_confidence        = verdict.max_confidence,
                mean_confidence       = verdict.mean_confidence,
                manipulation_ratio    = verdict.manipulation_ratio,
                peak_frame_index      = verdict.peak_frame_index,
                peak_timestamp_s      = verdict.peak_timestamp_s,
                flagged_timestamps    = verdict.flagged_timestamps,
                total_frames_analyzed = verdict.total_frames_analyzed,
                frame_counts          = verdict.frame_counts,
                temporal_consistency  = verdict.temporal_consistency,
                temporal_note         = verdict.temporal_note,
                processing_time_ms    = verdict.processing_time_ms,
                frames_per_second_sampled = verdict.frames_per_second_sampled,
            ),
            sha256           = video_sha256,
            fingerprint_id   = fp_id,
            ledger_registered= ledger_registered,
            ledger_note      = ledger_note,
            sample_fps       = sample_fps,
            max_frames_cap   = MAX_FRAMES,
            device_used      = str(DEVICE),
            model            = "V4 Proven Ensemble (Frame-by-Frame)",
            poc_note         = (
                "PoC: Frame-by-frame V4 Proven Ensemble analysis with temporal aggregation. "
                "Production would add TimeSformer/VideoSwin for native video understanding "
                "and Wav2Vec2 for audio-visual sync forensics."
            ),
        )

    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/video/supported", tags=["Video Detection"])
async def video_supported_formats():
    """Returns supported video formats and current configuration limits."""
    return {
        "supported_formats": ["MP4", "MOV", "AVI", "WebM", "MKV"],
        "max_file_size_mb" : MAX_VIDEO_SIZE // 1048576,
        "default_sample_fps": DEFAULT_SAMPLE_FPS,
        "max_frames_per_video": MAX_FRAMES,
        "sample_fps_range": [0.5, 5.0],
        "note": (
            "Higher sample_fps = more frames analyzed = more accurate but slower. "
            "1.0 fps is recommended for PoC demo."
        ),
    }
