"""
video/processor.py

SENTRY-X Video Analysis Pipeline

Strategy:
  - Extract frames at configurable FPS (default: 1 frame/sec for PoC)
  - Run deepfake detector on each frame independently
  - Aggregate per-frame results into a video-level verdict
  - Identify the specific timestamps where manipulation occurs

Why frame-by-frame (not whole-video models):
  - Works with our existing EfficientNet-B4 pipeline — no new model needed
  - Finds WHICH frames are manipulated, not just whether the video is
  - Face-swap deepfakes are rarely consistent across all frames —
    temporal inconsistency is itself a forensic signal
  - Auditable: judges can see exact frame evidence, not a black-box score

Production extension:
  - Add temporal consistency analysis (frame-to-frame delta)
  - Add audio-visual sync check (lip movement vs audio waveform)
  - Use video-native models (TimeSformer, VideoSwin) for better accuracy
"""

import cv2
import io
import time
import tempfile
import os
from pathlib import Path
from typing import Generator
from PIL import Image
import numpy as np


# ── Frame extraction ──────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    """
    Extract metadata from a video file without loading frames.
    Returns fps, frame count, duration, resolution.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_s, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
    }


def extract_frames(
    video_path: str,
    sample_fps: float = 1.0,
    max_frames: int = 60,
) -> Generator[tuple[int, float, Image.Image], None, None]:
    """
    Extract frames from video at `sample_fps` frames per second.

    Yields: (frame_index, timestamp_seconds, PIL.Image)

    Args:
        video_path:  Path to video file
        sample_fps:  How many frames per second to sample (default 1.0)
                     Higher = more thorough but slower
                     Lower  = faster but may miss short manipulation bursts
        max_frames:  Safety cap — never process more than this many frames
                     Prevents runaway processing on long videos

    The interval strategy means:
      - 30fps video, sample_fps=1.0 → every 30th frame
      - 30fps video, sample_fps=2.0 → every 15th frame
      - Always includes frame 0 (first frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many native frames to skip between each sample
    frame_interval = max(1, int(native_fps / sample_fps))

    frame_idx     = 0
    sample_count  = 0

    try:
        while frame_idx < total_frames and sample_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV reads BGR — convert to RGB for PIL/PyTorch
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            timestamp = frame_idx / native_fps

            yield frame_idx, round(timestamp, 3), pil_image

            frame_idx    += frame_interval
            sample_count += 1
    finally:
        cap.release()


def frames_to_jpeg_bytes(image: Image.Image, quality: int = 75) -> bytes:
    """Convert PIL image to JPEG bytes for display in Streamlit."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
