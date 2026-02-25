"""
app/v2_pipeline.py  —  SENTRY-X V4 Proven Pipeline Orchestrator
================================================================

Phase 1: Fast hash triage (threat ledger)
Phase 2: Proven multi-model ensemble detection
Phase 3: Contextual intent classification
Phase 4: Risk-based amplification control
Phase 5: Threat memory & registry
"""
import time
import asyncio
import logging
from typing import Dict, Any, Optional

from PIL import Image

from platforms.threat_intelligence import threat_intel_controller
from models.proven_ensemble import get_proven_ensemble
from models.intent_classifier import intent_classifier
from models.amplification_controller import amplification_controller
from platforms.reach_limiter import reach_limiter

logger = logging.getLogger("sentry_pipeline")


def phase1_instant_triage(file_bytes, image):
    return threat_intel_controller.execute_global_triage(file_bytes, image)


def phase2_proven_detection(image: Image.Image) -> Dict[str, Any]:
    """
    Phase 2: Proven Multi-Model Ensemble Detection
    - ViT-base deepfake detector (HuggingFace, fine-tuned on real deepfake data)
    - XceptionNet forensic detector (FaceForensics++ methodology)
    - GAN fingerprint detector (Wang et al. 2020, DCT/FFT frequency analysis)
    - Calibrated GBT meta-ensemble + Youden's J thresholds
    """
    ensemble = get_proven_ensemble()
    result = ensemble.predict(image)

    verdict_color = {
        "green": "#22c55e", "yellow": "#eab308",
        "orange": "#f97316", "red": "#ef4444",
    }

    raw = result.get("raw_scores", {})
    return {
        "calibrated_probability": result["probability"],
        "fusion_threat_score": result["probability"],
        "is_fake": result["action"] in ("restrict", "block"),
        "risk_classification": {
            "risk_level": result["risk_level"],
            "verdict": result["verdict"],
            "action": result["action"],
            "color_code": verdict_color.get(result["risk_level"], "#888"),
            "effective_score": result["probability"],
            "description": (
                f"Proven ensemble P(fake)={result['probability']:.3f}. "
                f"Detectors: ViT-deepfake, XceptionNet-forensic, GAN-fingerprint. "
                f"Thresholds from ROC curve (Youden's J)."
            ),
        },
        "raw_scores": raw,
        "thresholds": result.get("thresholds", {}),
        "forensic_signals": [
            f"ViT Deepfake P(fake)  = {raw.get('vit_deepfake_v2', 0):.4f}",
            f"XceptionNet forensic  = {raw.get('xception_forensic', 0):.4f}",
            f"GAN fingerprint (FFT) = {raw.get('gan_fingerprint_fft', 0):.4f}",
            f"Ensemble P(fake)      = {result['probability']:.4f}",
            f"Decision band         = {result['action']}",
        ],
        "latency_ms": result.get("latency_ms", 0),
    }


def phase3_intent_classification(threat_score: float, metadata: Dict) -> Dict[str, Any]:
    return intent_classifier.analyze_intent(threat_score, metadata)


def phase4_amplification_control(intent_data: Dict) -> Dict[str, Any]:
    policy_tier = amplification_controller.evaluate_risk(intent_data)
    return reach_limiter.apply_limits(policy_tier)


def phase5_threat_memory(policy: Dict, fingerprints: Dict, metadata: Dict) -> Optional[Dict]:
    return threat_intel_controller.execute_global_ban(policy, fingerprints, metadata)


async def execute_v2_pipeline(
    file_bytes: bytes,
    image: Image.Image,
    filename: str,
    platform_id: str,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Orchestrates all 5 phases of the SENTRY-X V4 pipeline."""
    if metadata is None:
        metadata = {}

    profiler = {}
    t_start = time.time()

    # Phase 1: Hash triage
    is_known, known_data, fingerprints = phase1_instant_triage(file_bytes, image)
    profiler["phase1_triage_ms"] = round((time.time() - t_start) * 1000, 2)

    if is_known:
        return {
            "status": "success",
            "pipeline_mode": "fast_path",
            "pipeline_version": "4.0_proven",
            "amplification_policy": {
                "tier": known_data["risk_level"],
                "action": "block" if known_data["risk_level"] == "red" else "restrict",
                "visibility_multiplier": 0.0,
                "policy_enforcement": "Triage layer: Cross-Platform Ledger match.",
            },
            "intent_classification": {"intent": "known_threat_match", "context_risk": 1.0},
            "threat_intelligence": {"node_status": "existing_record", "ledger_hit": True},
            "fingerprints": fingerprints,
            "latency_profile_ms": profiler,
        }

    # Phase 2: Proven ensemble detection
    t0 = time.time()
    detection_signals = await asyncio.to_thread(phase2_proven_detection, image)
    profiler["phase2_proven_detection_ms"] = round((time.time() - t0) * 1000, 2)

    # Phase 3: Intent profiling
    t0 = time.time()
    meta_payload = {
        "filename": filename, "platform_id": platform_id,
        "caption": metadata.get("caption", ""),
        "uploader_id": metadata.get("uploader_id", "user_default"),
    }
    intent_data = phase3_intent_classification(detection_signals["fusion_threat_score"], meta_payload)
    profiler["phase3_intent_ms"] = round((time.time() - t0) * 1000, 2)

    # Phase 4: Amplification control
    t0 = time.time()
    amplification_policy = phase4_amplification_control(intent_data)
    profiler["phase4_amplification_ms"] = round((time.time() - t0) * 1000, 2)

    # Phase 5: Threat memory
    t0 = time.time()
    ledger_action = await asyncio.to_thread(
        phase5_threat_memory, amplification_policy, fingerprints,
        {"filename": filename, "platform_id": platform_id},
    )
    profiler["phase5_ledger_sync_ms"] = round((time.time() - t0) * 1000, 2)
    profiler["total_pipeline_ms"] = round((time.time() - t_start) * 1000, 2)

    return {
        "status": "success",
        "pipeline_mode": "proven_ensemble",
        "pipeline_version": "4.0_proven",
        "amplification_policy": amplification_policy,
        "detection_signals": detection_signals,
        "intent_classification": intent_data,
        "threat_intelligence": ledger_action,
        "fingerprints": fingerprints,
        "latency_profile_ms": profiler,
    }
