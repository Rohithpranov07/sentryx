"""
video/ensemble_video_analyzer.py

SENTRY-X Video-Level Ensemble Detection
=========================================

Extends the multi-model ensemble to video content:
- Frame-level ensemble detection (all 3 models per frame)
- Temporal consistency analysis (confidence delta between frames)
- Frame-to-frame coherence scoring
- Aggregation with temporal forensic signals
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
import time

from models.proven_ensemble import get_proven_ensemble
from models.physiological_detector import physiological_detector

def get_ensemble():
    return get_proven_ensemble()

class TemporalConsistencyAnalyzer:
    """
    Analyzes temporal consistency of detection scores across video frames.
    
    Key insight: Real videos have smooth, consistent appearance.
    Deepfake videos have:
    - Sudden confidence spikes (face-swap fails on certain angles)
    - Frame-to-frame flickering in detection scores
    - Inconsistent edge/noise patterns between adjacent frames
    """

    def analyze_confidence_sequence(self, confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze the temporal pattern of frame-level confidence scores.
        """
        if len(confidences) < 3:
            return {
                "temporal_score": 0.0,
                "consistency": "insufficient_frames",
                "note": "Too few frames for temporal analysis",
            }
        
        confs = np.array(confidences)
        
        # 1. Frame-to-frame deltas
        deltas = np.diff(confs)
        abs_deltas = np.abs(deltas)
        
        # 2. Statistics
        mean_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))
        std_delta = float(np.std(abs_deltas))
        
        # 3. Detect spikes (sudden jumps in confidence)
        spike_threshold = 0.15
        spikes = int(np.sum(abs_deltas > spike_threshold))
        
        # 4. Temporal flickering index
        # Count sign changes in deltas (oscillation)
        sign_changes = int(np.sum(np.diff(np.sign(deltas)) != 0))
        flicker_ratio = sign_changes / max(1, len(deltas) - 1)
        
        # 5. Score temporal inconsistency
        temporal_score = 0.0
        consistency = "consistent"
        
        if spikes > len(confs) * 0.2:
            temporal_score += 0.3
            consistency = "burst"
        
        if flicker_ratio > 0.7:
            temporal_score += 0.2
            consistency = "flickering"
        
        if mean_delta > 0.08:
            temporal_score += 0.15
        
        if max_delta > 0.3:
            temporal_score += 0.1
        
        temporal_score = min(1.0, temporal_score)
        
        return {
            "temporal_score": round(temporal_score, 4),
            "consistency": consistency,
            "mean_delta": round(mean_delta, 4),
            "max_delta": round(max_delta, 4),
            "spike_count": spikes,
            "flicker_ratio": round(flicker_ratio, 4),
            "note": self._generate_note(temporal_score, consistency, spikes),
        }

    def analyze_frame_coherence(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray
    ) -> float:
        """
        Measure visual coherence between adjacent frames.
        Deepfakes can have subtle boundary flickering that real videos don't.
        """
        # Convert to grayscale
        if len(frame_a.shape) == 3:
            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
        else:
            gray_a, gray_b = frame_a, frame_b
        
        # Compute structural similarity (simplified)
        diff = cv2.absdiff(gray_a, gray_b).astype(np.float32)
        edge_a = cv2.Canny(gray_a, 50, 150)
        edge_b = cv2.Canny(gray_b, 50, 150)
        edge_diff = cv2.absdiff(edge_a, edge_b)
        
        # Edge coherence: real videos have smooth edge transitions
        edge_incoherence = float(np.mean(edge_diff)) / 255.0
        
        return edge_incoherence

    def _generate_note(self, score: float, consistency: str, spikes: int) -> str:
        if score > 0.5:
            return (
                f"Temporal analysis strongly indicates manipulation. "
                f"Detected {spikes} confidence spike(s) with {consistency} pattern."
            )
        elif score > 0.2:
            return (
                f"Moderate temporal inconsistency detected ({spikes} spikes). "
                f"Pattern: {consistency}."
            )
        else:
            return "Temporal confidence pattern consistent with authentic video."


def process_video_ensemble(
    frames: List[Tuple[int, float, Image.Image]],
) -> Dict[str, Any]:
    """
    Run ensemble detection on a sequence of video frames.
    
    Args:
        frames: List of (frame_index, timestamp_seconds, PIL.Image)
    
    Returns:
        Comprehensive video-level analysis with per-frame ensemble results.
    """
    t0 = time.time()
    
    frame_results = []
    confidences = []
    prev_frame_np = None
    coherence_scores = []
    
    temporal_analyzer = TemporalConsistencyAnalyzer()
    
    for frame_idx, timestamp, pil_image in frames:
        # Run ensemble on each frame
        ensemble_result = ensemble_detector.predict(pil_image)
        ensemble_score = ensemble_result["ensemble_score"]
        
        # Run physiological on each frame too
        physio_data = physiological_detector.analyze(pil_image)
        physio_conf = physio_data["physiological_confidence"]
        
        # Combine
        frame_score = min(1.0, ensemble_score + physio_conf * 0.1)
        confidences.append(frame_score)
        
        # Frame-to-frame coherence
        frame_np = np.array(pil_image.convert("RGB"))
        if prev_frame_np is not None:
            coherence = temporal_analyzer.analyze_frame_coherence(prev_frame_np, frame_np)
            coherence_scores.append(coherence)
        prev_frame_np = frame_np
        
        frame_results.append({
            "frame_index": frame_idx,
            "timestamp_s": round(timestamp, 3),
            "ensemble_score": round(frame_score, 4),
            "agreement": ensemble_result["agreement"],
            "individual_detectors": [
                {"name": r["detector"], "confidence": r["confidence"]}
                for r in ensemble_result["individual_results"]
            ],
            "physiological_confidence": round(physio_conf, 4),
        })
    
    # Temporal consistency analysis
    temporal_analysis = temporal_analyzer.analyze_confidence_sequence(confidences)
    
    # Aggregate statistics
    all_confs = np.array(confidences)
    max_conf = float(np.max(all_confs))
    mean_conf = float(np.mean(all_confs))
    
    # Count frames by risk level
    high_frames = int(np.sum(all_confs > 0.65))
    medium_frames = int(np.sum((all_confs > 0.35) & (all_confs <= 0.65)))
    low_frames = int(np.sum(all_confs <= 0.35))
    
    # Mean coherence incoherence
    mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    processing_time_ms = (time.time() - t0) * 1000
    
    return {
        "frame_results": frame_results,
        "temporal_analysis": temporal_analysis,
        "aggregate": {
            "max_confidence": round(max_conf, 4),
            "mean_confidence": round(mean_conf, 4),
            "total_frames": len(frame_results),
            "high_risk_frames": high_frames,
            "medium_risk_frames": medium_frames,
            "low_risk_frames": low_frames,
            "mean_coherence_incoherence": round(mean_coherence, 4),
        },
        "processing_time_ms": round(processing_time_ms, 2),
    }
