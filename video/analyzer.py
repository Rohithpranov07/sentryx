"""
video/analyzer.py

Aggregates per-frame detection results into a video-level verdict.

Key insight: A deepfake video is rarely 100% manipulated.
Face-swap techniques often fail on certain frames (motion blur,
extreme angles, occlusion). This temporal inconsistency IS evidence.

Aggregation strategy:
  - MAX confidence → catches even brief manipulation bursts
  - MEAN confidence → overall manipulation level
  - PEAK TIMESTAMP → where manipulation is strongest
  - MANIPULATION RATIO → % of frames flagged

Video-level risk rules:
  - Any frame RED → video is RED (worst-case dominates for safety)
  - >30% frames ORANGE or above → video is ORANGE
  - >50% frames YELLOW → video is YELLOW
  - Otherwise → GREEN
"""

import time
from typing import List
from dataclasses import dataclass, field, asdict


@dataclass
class FrameResult:
    frame_index  : int
    timestamp_s  : float
    confidence   : float
    risk_level   : str          # green | yellow | orange | red
    verdict      : str
    action       : str
    forensic_signals: List[str] = field(default_factory=list)


@dataclass
class VideoVerdict:
    # Overall classification
    risk_level          : str
    verdict             : str
    action              : str
    description         : str
    color_code          : str

    # Aggregate scores
    max_confidence      : float
    mean_confidence     : float
    manipulation_ratio  : float   # 0.0–1.0, % of frames not green

    # Key timestamps
    peak_frame_index    : int
    peak_timestamp_s    : float
    flagged_timestamps  : List[float]   # all timestamps where risk >= orange

    # Frame summary
    total_frames_analyzed: int
    frame_counts        : dict   # {green: N, yellow: N, orange: N, red: N}

    # Temporal forensic signal
    temporal_consistency: str    # "consistent" | "inconsistent" | "burst"
    temporal_note       : str

    # Processing
    processing_time_ms  : float
    frames_per_second_sampled: float


def aggregate_frame_results(
    frame_results: List[FrameResult],
    sample_fps: float,
    processing_time_ms: float,
) -> VideoVerdict:
    """
    Given a list of per-frame results, produce a single VideoVerdict.
    """
    if not frame_results:
        raise ValueError("No frame results to aggregate")

    n = len(frame_results)
    confidences = [f.confidence for f in frame_results]

    max_conf  = max(confidences)
    mean_conf = sum(confidences) / n

    # Frame counts by risk level
    counts = {"green": 0, "yellow": 0, "orange": 0, "red": 0}
    for f in frame_results:
        counts[f.risk_level] = counts.get(f.risk_level, 0) + 1

    # Peak frame (highest confidence)
    peak_frame = max(frame_results, key=lambda f: f.confidence)

    # Flagged timestamps (orange or red)
    flagged_ts = [
        f.timestamp_s for f in frame_results
        if f.risk_level in ("orange", "red")
    ]

    # Manipulation ratio = frames that are NOT green
    not_green = n - counts.get("green", 0)
    manipulation_ratio = not_green / n

    # ── Temporal consistency analysis ──────────────────────────────────────
    # Look at confidence sequence — is manipulation spread out or bursty?
    temporal_consistency, temporal_note = _analyze_temporal_pattern(
        frame_results, counts, manipulation_ratio
    )

    # ── Video-level risk determination ─────────────────────────────────────
    if counts.get("red", 0) > 0:
        risk_level  = "red"
        verdict     = "Deepfake Detected"
        action      = "block"
        color_code  = "#ef4444"
        description = (
            f"{counts['red']} frame(s) contain high-confidence manipulation. "
            f"Peak at {peak_frame.timestamp_s:.1f}s. "
            f"Content blocked and fingerprint registered to ledger."
        )
    elif manipulation_ratio > 0.30:
        risk_level  = "orange"
        verdict     = "Suspicious Video — Restricted"
        action      = "restrict"
        color_code  = "#f97316"
        description = (
            f"{int(manipulation_ratio*100)}% of frames show manipulation artifacts. "
            f"Content restricted pending human review."
        )
    elif manipulation_ratio > 0.10 or counts.get("yellow", 0) > 0:
        risk_level  = "yellow"
        verdict     = "AI-Generated Content"
        action      = "label"
        color_code  = "#eab308"
        description = (
            f"Synthetic generation patterns detected in {not_green}/{n} frames. "
            f"Content may be AI-generated. Publish with disclosure label."
        )
    else:
        risk_level  = "green"
        verdict     = "Authentic & Safe"
        action      = "publish"
        color_code  = "#22c55e"
        description = (
            f"No significant manipulation detected across {n} analyzed frames. "
            f"Content cleared for publishing."
        )

    return VideoVerdict(
        risk_level           = risk_level,
        verdict              = verdict,
        action               = action,
        description          = description,
        color_code           = color_code,
        max_confidence       = round(max_conf, 4),
        mean_confidence      = round(mean_conf, 4),
        manipulation_ratio   = round(manipulation_ratio, 4),
        peak_frame_index     = peak_frame.frame_index,
        peak_timestamp_s     = peak_frame.timestamp_s,
        flagged_timestamps   = sorted(flagged_ts),
        total_frames_analyzed= n,
        frame_counts         = counts,
        temporal_consistency = temporal_consistency,
        temporal_note        = temporal_note,
        processing_time_ms   = round(processing_time_ms, 2),
        frames_per_second_sampled = sample_fps,
    )


def _analyze_temporal_pattern(
    results: List[FrameResult],
    counts: dict,
    manipulation_ratio: float,
) -> tuple[str, str]:
    """
    Detect whether manipulation is:
      - consistent (uniform across video — likely fully synthetic)
      - burst (concentrated in short segment — likely face-swap on specific scene)
      - inconsistent (scattered — could be compression artifact or weak signal)
    """
    if manipulation_ratio < 0.05:
        return "consistent", "Manipulation signal below threshold across all frames — video appears authentic."

    # Get indices of flagged frames
    flagged = [i for i, r in enumerate(results) if r.risk_level in ("orange", "red", "yellow")]

    if not flagged:
        return "consistent", "No flagged frames detected."

    n = len(results)
    flagged_ratio = len(flagged) / n

    # Check if flagged frames are clustered (burst) or spread (consistent/inconsistent)
    if len(flagged) >= 2:
        # Measure gap between flagged frames
        gaps = [flagged[i+1] - flagged[i] for i in range(len(flagged)-1)]
        max_gap = max(gaps)
        mean_gap = sum(gaps) / len(gaps)
    else:
        max_gap = n
        mean_gap = n

    if flagged_ratio > 0.70:
        return (
            "consistent",
            "Manipulation artifacts detected uniformly across video — consistent with fully synthetic generation (GAN/diffusion-produced content)."
        )
    elif max_gap > (n * 0.5) and len(flagged) <= max(3, n * 0.2):
        return (
            "burst",
            f"Manipulation concentrated in short segment around {results[flagged[0]].timestamp_s:.1f}s–{results[flagged[-1]].timestamp_s:.1f}s — "
            "pattern consistent with face-swap or scene-specific deepfake insertion."
        )
    else:
        return (
            "inconsistent",
            "Manipulation signal scattered across non-contiguous frames — "
            "may indicate partial synthetic generation, heavy compression artifacts, or borderline content."
        )


def build_timeline_data(frame_results: List[FrameResult]) -> list:
    """
    Build a chart-friendly list of dicts for Streamlit visualization.
    """
    return [
        {
            "timestamp": r.timestamp_s,
            "confidence": round(r.confidence * 100, 1),
            "risk_level": r.risk_level,
            "frame_index": r.frame_index,
        }
        for r in frame_results
    ]
