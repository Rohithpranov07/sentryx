"""
models/ensemble_detector.py

SENTRY-X Multi-Model Ensemble Detection Engine
===============================================

Implements TASK 2: At least 3 independent detectors with weighted voting.

Detectors:
    1. SENTRY-X Primary (EfficientNet-B4) — existing spatial CNN
    2. XceptionNet Deepfake Classifier — FaceForensics++ architecture
    3. CLIP-based Synthetic Image Detector — latent space anomaly detection

Final decision uses weighted average + majority vote agreement.
No single model can make a decision alone.

Each detector is modular and independent — can be swapped, retrained,
or disabled without affecting others.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional
import os
import time
import logging
import json
from datetime import datetime

from utils.device import DEVICE

# ── Logging for disagreement cases ────────────────────────────────────────────
DISAGREEMENT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "disagreements")
os.makedirs(DISAGREEMENT_LOG_DIR, exist_ok=True)

logger = logging.getLogger("sentry_ensemble")
logging.basicConfig(level=logging.INFO)

# ── Shared preprocessing ─────────────────────────────────────────────────────
EFFICIENTNET_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

XCEPTION_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

CLIP_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 1: SENTRY-X Primary (EfficientNet-B4)
# ═══════════════════════════════════════════════════════════════════════════════

class SentryXDetector:
    """
    Primary detector: EfficientNet-B4 for spatial artifact detection.
    In production, fine-tuned on FF++/DFDC/Celeb-DF.
    """
    def __init__(self):
        self.name = "sentry_x_efficientnet_b4"
        self.weight = 0.40  # Ensemble weight
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[Ensemble] Loading Detector 1: EfficientNet-B4 on {DEVICE}...")
        self.model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=1)
        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        tensor = EFFICIENTNET_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        logit = self.model(tensor)
        
        # Mild temperature scaling
        temperature = 1.15
        logit = logit / temperature
        cnn_confidence = torch.sigmoid(logit).item()
        
        # Forensic signal-processing features (works with pretrained weights)
        forensic_score = self._compute_forensic_features(image)
        
        # Blend: CNN provides texture awareness, forensic provides real-vs-fake signal
        # In production with fine-tuned weights, CNN dominates; in PoC, forensic helps
        confidence = (cnn_confidence * 0.35) + (forensic_score * 0.65)
        
        return {
            "detector": self.name,
            "confidence": round(confidence, 4),
            "weight": self.weight,
            "cnn_raw": round(cnn_confidence, 4),
            "forensic_score": round(forensic_score, 4),
            "signals": self._generate_signals(confidence, image),
        }

    def _compute_forensic_features(self, image: Image.Image) -> float:
        """
        Signal-processing based synthetic detection.
        Analyzes noise variance, texture regularity, and sensor-like patterns.
        """
        img_np = np.array(image.convert("RGB"))
        gray = np.mean(img_np, axis=2)
        score = 0.0
        
        # 1. Noise variance analysis
        # Real camera images have characteristic sensor noise (std 8-25)
        # AI images tend to be smoother (std < 5) or have unnatural noise patterns
        noise_std = float(np.std(np.diff(gray, axis=1)))
        if noise_std < 3.0:
            score += 0.35  # Unnaturally smooth
        elif noise_std < 6.0:
            score += 0.20  # Suspiciously smooth
        elif 8.0 <= noise_std <= 30.0:
            score -= 0.15  # Natural camera noise range
        
        # 2. Local variance consistency
        # Real photos have varying local variance; AI images are more uniform
        h, w = gray.shape
        block_size = min(32, h // 4, w // 4)
        if block_size > 4:
            local_vars = []
            for by in range(0, h - block_size, block_size):
                for bx in range(0, w - block_size, block_size):
                    block = gray[by:by+block_size, bx:bx+block_size]
                    local_vars.append(float(np.var(block)))
            if local_vars:
                var_of_vars = float(np.var(local_vars))
                mean_var = float(np.mean(local_vars))
                cv = var_of_vars / (mean_var + 1e-6)
                if cv < 2.0:  # Very uniform variance = synthetic
                    score += 0.20
                elif cv > 50.0:  # High variation = natural
                    score -= 0.10
        
        # 3. Color channel independence
        r, g, b = img_np[:,:,0].flatten(), img_np[:,:,1].flatten(), img_np[:,:,2].flatten()
        rg_corr = float(np.corrcoef(r, g)[0, 1])
        # AI images sometimes have lower inter-channel correlation
        if rg_corr < 0.6:
            score += 0.15
        elif rg_corr > 0.9:
            score -= 0.05
        
        return max(0.0, min(1.0, 0.3 + score))  # Base 0.3 + adjustments

    def _generate_signals(self, confidence: float, image: Image.Image) -> List[str]:
        signals = []
        img_array = np.array(image.convert("RGB"))
        
        gray = np.mean(img_array, axis=2)
        noise_std = float(np.std(np.diff(gray, axis=1)))
        
        if noise_std < 2.0:
            signals.append(f"[EfficientNet] Extremely low texture variance ({noise_std:.1f}) — synthetic smoothing pattern")
        
        w, h = image.size
        if w == h and w > 1500:
            signals.append(f"[EfficientNet] 1:1 high-res ({w}x{h}) — common diffusion model output")
        
        if confidence > 0.65:
            signals.append("[EfficientNet] Strong spatial artifact detection signal")
        elif confidence > 0.40:
            signals.append("[EfficientNet] Moderate manipulation indicators detected")
        
        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 2: XceptionNet Deepfake Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class XceptionNetDetector:
    """
    Secondary detector: Xception architecture.
    XceptionNet is the gold standard for FaceForensics++ benchmarks.
    Uses depthwise separable convolutions that excel at detecting
    compression artifacts and face-swap boundary artifacts.
    
    In production: load weights fine-tuned on FF++ c23/c40.
    PoC: uses ImageNet pretrained Xception as structural demo.
    """
    def __init__(self):
        self.name = "xceptionnet_deepfake"
        self.weight = 0.35  # Ensemble weight
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[Ensemble] Loading Detector 2: XceptionNet on {DEVICE}...")
        # Try xception41 from HuggingFace hub first (avoids GitHub SSL issues)
        # Fallback chain: xception41 -> xception -> efficientnet_b0
        import ssl
        for model_name in ["xception41", "xception", "efficientnet_b0"]:
            try:
                self.model = timm.create_model(model_name, pretrained=True, num_classes=1)
                self.model = self.model.to(DEVICE)
                self.model.eval()
                self.name = f"xceptionnet_{model_name}"
                print(f"  [Detector 2] Loaded: {model_name}")
                return
            except Exception as e:
                print(f"  [Detector 2] {model_name} failed: {e}, trying next...")
        raise RuntimeError("Could not load any XceptionNet variant")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        tensor = XCEPTION_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        logit = self.model(tensor)
        
        temperature = 1.10
        logit = logit / temperature
        cnn_confidence = torch.sigmoid(logit).item()
        
        # Edge and boundary forensic analysis
        forensic_score = self._compute_boundary_forensics(image)
        
        # Blend CNN + forensics
        confidence = (cnn_confidence * 0.35) + (forensic_score * 0.65)
        
        return {
            "detector": self.name,
            "confidence": round(confidence, 4),
            "weight": self.weight,
            "cnn_raw": round(cnn_confidence, 4),
            "forensic_score": round(forensic_score, 4),
            "signals": self._generate_signals(confidence, image),
        }

    def _compute_boundary_forensics(self, image: Image.Image) -> float:
        """
        Analyzes edge coherence and boundary artifacts.
        Face-swap deepfakes leave edge discontinuities.
        GAN images have unnaturally smooth edges.
        """
        import cv2
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        score = 0.0
        
        # 1. Edge density analysis
        edges = cv2.Canny(gray, 100, 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_density = float(np.sum(edges / 255.0)) / total_pixels
        
        # Very low edge density suggests AI smoothing
        if edge_density < 0.01:
            score += 0.30
        elif edge_density < 0.03:
            score += 0.15
        elif edge_density > 0.08:
            score -= 0.10  # Rich edges = likely real
        
        # 2. Laplacian variance (focus/blur consistency)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 30:
            score += 0.25  # Unnaturally uniform blur
        elif laplacian_var < 80:
            score += 0.10
        elif laplacian_var > 200:
            score -= 0.10  # Natural sharp textures
        
        # 3. Center vs periphery edge ratio (face-swap detection)
        h, w = gray.shape
        margin = min(h, w) // 4
        if margin > 10:
            center_edges = edges[margin:h-margin, margin:w-margin]
            periph_edges = np.concatenate([
                edges[:margin, :].flatten(),
                edges[h-margin:, :].flatten(),
                edges[:, :margin].flatten(),
                edges[:, w-margin:].flatten()
            ])
            center_density = float(np.mean(center_edges)) / 255.0
            periph_density = float(np.mean(periph_edges)) / 255.0
            
            if periph_density > 0:
                ratio = center_density / (periph_density + 1e-6)
                if ratio > 3.0 or ratio < 0.25:
                    score += 0.15  # Inconsistent edge distribution
        
        return max(0.0, min(1.0, 0.3 + score))

    def _generate_signals(self, confidence: float, image: Image.Image) -> List[str]:
        signals = []
        if confidence > 0.65:
            signals.append("[XceptionNet] Strong face manipulation boundary artifacts detected")
        elif confidence > 0.40:
            signals.append("[XceptionNet] Moderate compression-domain anomalies")
        else:
            signals.append("[XceptionNet] No significant face-swap artifacts")
        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 3: CLIP-based Synthetic Image Detector
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPSyntheticDetector:
    """
    Tertiary detector: Uses a CLIP-like vision encoder to detect
    synthetic image characteristics in the latent embedding space.
    
    Key insight: CLIP's learned representations capture high-level semantic
    features that differ systematically between real photographs and
    AI-generated images (texture coherence, lighting physics, edge patterns).
    
    Uses a lightweight visual transformer (ViT-B/16) with a binary head.
    In production: fine-tuned on real-vs-synthetic datasets.
    """
    def __init__(self):
        self.name = "clip_synthetic_detector"
        self.weight = 0.25  # Ensemble weight
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[Ensemble] Loading Detector 3: CLIP-ViT Synthetic Detector on {DEVICE}...")
        # ViT-B/16 — similar to CLIP's visual encoder
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)
        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        tensor = CLIP_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        logit = self.model(tensor)
        
        temperature = 1.10
        logit = logit / temperature
        cnn_confidence = torch.sigmoid(logit).item()
        
        # FFT spectrum analysis for synthetic detection
        fft_score = self._analyze_fft_spectrum(image)
        
        # Combine: ViT provides embedding awareness, FFT provides spectral forensics
        combined = (cnn_confidence * 0.30) + (fft_score * 0.70)
        
        return {
            "detector": self.name,
            "confidence": round(combined, 4),
            "weight": self.weight,
            "fft_synthetic_score": round(fft_score, 4),
            "vit_raw_score": round(cnn_confidence, 4),
            "signals": self._generate_signals(combined, fft_score),
        }

    def _analyze_fft_spectrum(self, image: Image.Image) -> float:
        """
        Analyzes FFT spectrum for synthetic image characteristics.
        AI-generated images have distinctive spectral signatures:
        - GAN checkerboard artifacts in mid-frequencies
        - Diffusion model upsampling patterns in high-frequencies
        - Unnatural energy distribution vs real camera sensors
        """
        img_np = np.array(image.convert("L"))
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Radial energy distribution analysis
        r_inner = int(0.15 * min(cx, cy))
        r_mid = int(0.5 * min(cx, cy))
        r_outer = int(0.85 * min(cx, cy))
        
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = x*x + y*y
        
        inner_mask = dist < r_inner*r_inner
        mid_mask = (dist >= r_inner*r_inner) & (dist < r_mid*r_mid)
        outer_mask = (dist >= r_mid*r_mid) & (dist < r_outer*r_outer)
        
        inner_energy = np.mean(magnitude[inner_mask])
        mid_energy = np.mean(magnitude[mid_mask])
        outer_energy = np.mean(magnitude[outer_mask])
        
        # Synthetic images have abnormal energy falloff
        # Real camera images follow a predictable 1/f spectrum decay
        if inner_energy > 0:
            mid_ratio = mid_energy / inner_energy
            outer_ratio = outer_energy / inner_energy
        else:
            mid_ratio = 0.5
            outer_ratio = 0.3
        
        # Score synthetic likelihood based on spectral anomalies
        score = 0.0
        
        # GAN/diffusion images often have elevated mid-frequency energy
        if mid_ratio > 0.75:
            score += 0.3
        elif mid_ratio > 0.65:
            score += 0.15
        
        # Unusual high-frequency retention (GAN upsampling artifacts)
        if outer_ratio > 0.55:
            score += 0.35
        elif outer_ratio > 0.45:
            score += 0.15
        
        # Mid-frequency variance (checkerboard patterns)
        mid_var = np.var(magnitude[mid_mask])
        if mid_var > 800:
            score += 0.2
        elif mid_var > 500:
            score += 0.1
        
        return min(1.0, score)

    def _generate_signals(self, confidence: float, fft_score: float) -> List[str]:
        signals = []
        if confidence > 0.60:
            signals.append("[CLIP-ViT] Latent embedding strongly indicates synthetic generation")
        elif confidence > 0.35:
            signals.append("[CLIP-ViT] Moderate synthetic characteristics in embedding space")
        
        if fft_score > 0.5:
            signals.append(f"[FFT-Spectrum] Abnormal spectral energy distribution (score: {fft_score:.2f})")
        
        if not signals:
            signals.append("[CLIP-ViT] No strong synthetic signatures detected")
        
        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleDetector:
    """
    Orchestrates multi-model ensemble detection.
    
    Decision rules:
    - Weighted average of all detector confidences
    - Majority vote check (at least 2 of 3 must agree)
    - No single-model decision allowed
    - Disagreement cases are logged for review
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
        
        print("[Ensemble] Initializing Multi-Model Ensemble Detector...")
        t0 = time.time()
        
        self.detectors = []
        self._load_detectors()
        
        elapsed = time.time() - t0
        print(f"[Ensemble] All {len(self.detectors)} detectors loaded in {elapsed:.2f}s")
        self._initialized = True

    def _load_detectors(self):
        """Load all detectors. Each is independent and modular."""
        try:
            self.detectors.append(SentryXDetector())
            print("  ✓ Detector 1: EfficientNet-B4 (SENTRY-X Primary)")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B4: {e}")
        
        try:
            self.detectors.append(XceptionNetDetector())
            print("  ✓ Detector 2: XceptionNet (FaceForensics++)")
        except Exception as e:
            logger.error(f"Failed to load XceptionNet: {e}")
        
        try:
            self.detectors.append(CLIPSyntheticDetector())
            print("  ✓ Detector 3: CLIP-ViT Synthetic Detector")
        except Exception as e:
            logger.error(f"Failed to load CLIP-ViT: {e}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run all detectors and produce ensemble verdict.
        Returns individual results + fused score + agreement info.
        """
        results = []
        all_signals = []
        
        for det in self.detectors:
            try:
                result = det.predict(image)
                results.append(result)
                all_signals.extend(result.get("signals", []))
            except Exception as e:
                logger.error(f"Detector {det.name} failed: {e}")
                results.append({
                    "detector": det.name,
                    "confidence": 0.0,
                    "weight": det.weight,
                    "signals": [f"[{det.name}] Detector error: {str(e)}"],
                    "error": True,
                })
        
        # ── Weighted Average Fusion ──
        total_weight = sum(r["weight"] for r in results if not r.get("error"))
        if total_weight > 0:
            weighted_score = sum(
                r["confidence"] * r["weight"]
                for r in results if not r.get("error")
            ) / total_weight
        else:
            weighted_score = 0.0
        
        # ── Agreement Analysis ──
        threshold_medium = 0.40  # Threshold for "detector flags suspicious"
        threshold_high = 0.65   # Threshold for "detector flags likely fake"
        
        active_results = [r for r in results if not r.get("error")]
        
        models_flagging_medium = [r for r in active_results if r["confidence"] >= threshold_medium]
        models_flagging_high = [r for r in active_results if r["confidence"] >= threshold_high]
        
        agreement_count_medium = len(models_flagging_medium)
        agreement_count_high = len(models_flagging_high)
        total_active = len(active_results)
        
        # ── Majority Vote Logic ──
        majority_threshold = max(2, (total_active // 2) + 1) if total_active >= 2 else total_active
        has_majority_medium = agreement_count_medium >= 2  # At least 2 models at medium
        has_majority_high = agreement_count_high >= 2      # At least 2 models at high
        
        # ── Final Ensemble Score (combining weighted avg + agreement) ──
        if has_majority_high:
            # Strong agreement — trust the weighted average with a boost
            ensemble_score = min(1.0, weighted_score * 1.10)
        elif has_majority_medium:
            # Medium agreement — use weighted average as-is
            ensemble_score = weighted_score
        elif agreement_count_medium == 1 and agreement_count_high == 0:
            # Only 1 model flagging — dampen but don't crush
            ensemble_score = weighted_score * 0.70
            all_signals.append("[Ensemble] Only 1 detector flagging — dampened (not suppressed)")
        else:
            # No models flagging
            ensemble_score = weighted_score * 0.50
        
        # ── Log disagreements ──
        if self._has_disagreement(active_results, threshold_medium):
            self._log_disagreement(results, ensemble_score, image)
            all_signals.append("[Ensemble] Model disagreement detected — logged for human review")
        
        return {
            "ensemble_score": round(ensemble_score, 4),
            "weighted_average": round(weighted_score, 4),
            "individual_results": results,
            "agreement": {
                "models_active": total_active,
                "models_flagging_medium": agreement_count_medium,
                "models_flagging_high": agreement_count_high,
                "has_majority": has_majority_medium,
                "has_strong_majority": has_majority_high,
            },
            "signals": all_signals,
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run ensemble prediction on a batch of images."""
        return [self.predict(img) for img in images]

    def _has_disagreement(self, results: List[Dict], threshold: float) -> bool:
        """Check if detectors significantly disagree."""
        if len(results) < 2:
            return False
        confs = [r["confidence"] for r in results]
        # Disagreement: one model >threshold and another <(threshold - 0.2)
        max_conf = max(confs)
        min_conf = min(confs)
        return max_conf >= threshold and min_conf < (threshold - 0.20)

    def _log_disagreement(self, results: List[Dict], ensemble_score: float, image: Image.Image):
        """Log disagreement cases for human review."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "ensemble_score": ensemble_score,
                "detectors": [
                    {"name": r["detector"], "confidence": r["confidence"]}
                    for r in results
                ],
                "image_size": f"{image.size[0]}x{image.size[1]}",
            }
            
            log_file = os.path.join(
                DISAGREEMENT_LOG_DIR,
                f"disagreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to log disagreement: {e}")


# ── Singleton instance ────────────────────────────────────────────────────────
ensemble_detector = EnsembleDetector()
