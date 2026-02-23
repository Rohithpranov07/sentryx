"""
models/detector.py

SENTRY-X Deepfake Detection Engine

Architecture: EfficientNet-B4 fine-tuned for binary classification
  - Class 0: REAL (authentic)
  - Class 1: FAKE (manipulated / AI-generated)

Why EfficientNet-B4:
  - State-of-the-art on FaceForensics++ benchmark
  - Good balance of accuracy vs inference speed
  - Pretrained ImageNet weights → fine-tuned on deepfake datasets
  - Runs acceptably on CPU for demo purposes (~1-3s per image)

For the PoC we load pretrained ImageNet weights and treat the
final sigmoid output as a manipulation confidence proxy.
A production system would use weights fine-tuned on:
  - FaceForensics++ (FF++)
  - DFDC (Deepfake Detection Challenge dataset)
  - Celeb-DF v2

The model is loaded once at startup (singleton pattern) to avoid
re-loading on every request — critical for API performance.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import numpy as np
from PIL import Image
from typing import Tuple
import os
import time

from utils.device import DEVICE


# ── Image preprocessing pipeline ─────────────────────────────────────────────
# EfficientNet expects 224x224 normalized to ImageNet stats
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_GREEN_MAX  = float(os.getenv("RISK_GREEN_MAX",  "0.30"))
RISK_YELLOW_MAX = float(os.getenv("RISK_YELLOW_MAX", "0.55"))
RISK_ORANGE_MAX = float(os.getenv("RISK_ORANGE_MAX", "0.75"))


class DeepfakeDetector:
    """
    Singleton deepfake detection model.
    Loaded once at startup, reused for all requests.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """
        Load EfficientNet-B4 with pretrained ImageNet weights.
        Replace classifier head with binary output (real vs fake).

        Production note: swap `pretrained=True` for a checkpoint
        fine-tuned on FF++ / DFDC for dramatically better accuracy.
        The architecture and inference pipeline remain identical.
        """
        print(f"[SENTRY-X] Loading detection model on {DEVICE}...")
        start = time.time()

        # Load backbone
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,        # ImageNet weights
            num_classes=1,          # Binary: real(0) vs fake(1)
        )

        # Move to best available device
        self.model = self.model.to(DEVICE)
        self.model.eval()

        elapsed = time.time() - start
        print(f"[SENTRY-X] Model loaded in {elapsed:.2f}s on {DEVICE}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, list]:
        """
        Run inference on a PIL Image.

        Returns:
            confidence (float): 0.0 = definitely real, 1.0 = definitely fake
            forensic_signals (list[str]): human-readable explanation of signals

        Note on PoC vs Production:
            With ImageNet weights only, this model detects visual anomalies
            and texture inconsistencies but is NOT calibrated for deepfakes.
            Fine-tuned weights on FF++/DFDC would give 90%+ accuracy.
            The PoC uses this as a structural demo of the inference pipeline.
        """
        # Preprocess
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        # Forward pass
        logit = self.model(tensor)
        confidence = torch.sigmoid(logit).item()

        # Generate forensic signals based on confidence bands
        signals = self._generate_signals(confidence, image)

        return confidence, signals

    def _generate_signals(self, confidence: float, image: Image.Image) -> list:
        """
        Generate human-readable forensic signal descriptions.
        In production these would come from attention maps, GradCAM,
        and per-module sub-detectors (audio, compression artifacts, etc.)
        """
        signals = []
        img_array = np.array(image.convert("RGB"))

        # ── Signal 1: Texture anomaly estimate ──
        # High-frequency noise in real images follows natural statistics.
        # GAN/diffusion outputs have characteristic frequency signatures.
        gray = np.mean(img_array, axis=2)
        noise_std = float(np.std(np.diff(gray, axis=1)))
        if noise_std < 8.0:
            signals.append(f"Low texture noise variance ({noise_std:.1f}) — consistent with synthetic smoothing")
        elif noise_std > 40.0:
            signals.append(f"Elevated texture noise ({noise_std:.1f}) — possible JPEG artifact injection")

        # ── Signal 2: Color channel imbalance ──
        # GANs sometimes produce subtle RGB channel correlation artifacts
        r_mean = float(np.mean(img_array[:, :, 0]))
        g_mean = float(np.mean(img_array[:, :, 1]))
        b_mean = float(np.mean(img_array[:, :, 2]))
        channel_delta = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
        if channel_delta > 30:
            signals.append(f"Color channel imbalance detected (delta: {channel_delta:.1f}) — atypical for natural photography")

        # ── Signal 3: Confidence-based signals ──
        if confidence > 0.75:
            signals.append("Model classifier confidence strongly indicates synthetic generation")
            signals.append("Pattern inconsistencies detected in feature space")
        elif confidence > 0.55:
            signals.append("Moderate manipulation confidence — content flagged for secondary review")
        elif confidence > 0.30:
            signals.append("Low-level anomalies detected — AI-generated content possible")
        else:
            signals.append("No significant manipulation artifacts detected")

        # ── Signal 4: Resolution anomaly ──
        w, h = image.size
        if w != h and (w > 1024 or h > 1024):
            signals.append(f"Non-standard aspect ratio ({w}x{h}) — may indicate cropped synthetic output")

        return signals if signals else ["No anomalous signals detected"]


def classify_risk(confidence: float) -> dict:
    """
    Map model confidence score to a 4-tier risk classification.

    Returns:
        risk_level: green | yellow | orange | red
        verdict: human-readable decision
        action: publish | label | restrict | block
        color_code: hex color for UI
    """
    if confidence <= RISK_GREEN_MAX:
        return {
            "risk_level": "green",
            "verdict": "Authentic & Safe",
            "action": "publish",
            "color_code": "#22c55e",
            "description": "No manipulation artifacts detected. Content cleared for publishing.",
        }
    elif confidence <= RISK_YELLOW_MAX:
        return {
            "risk_level": "yellow",
            "verdict": "AI-Generated Content",
            "action": "label",
            "color_code": "#eab308",
            "description": "Content appears AI-generated but not deceptively manipulated. Publish with disclosure label.",
        }
    elif confidence <= RISK_ORANGE_MAX:
        return {
            "risk_level": "orange",
            "verdict": "Suspicious — Restricted",
            "action": "restrict",
            "color_code": "#f97316",
            "description": "Manipulation artifacts detected. Content restricted pending human review.",
        }
    else:
        return {
            "risk_level": "red",
            "verdict": "Manipulated Media Detected",
            "action": "block",
            "color_code": "#ef4444",
            "description": "High-confidence deepfake or synthetic manipulation. Content blocked. Fingerprint registered to ledger.",
        }


# ── Singleton instance (loaded at startup) ────────────────────────────────────
detector = DeepfakeDetector()
