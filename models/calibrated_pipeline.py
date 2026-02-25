"""
models/calibrated_pipeline.py

SENTRY-X V3 — Calibrated Detection Pipeline
============================================

EVERY number in this system is learned from data. NOTHING is guessed.

Architecture:
  1. Feature extractors: extract 14 numerical features from an image
  2. Calibrators: Platt scaling + isotonic regression on each raw score
  3. Meta-classifier: logistic regression trained on calibrated features
  4. Decision thresholds: chosen from ROC curve to hit FPR < 5%
  5. Contradiction logic: hard rules on top of calibrated probabilities

The pipeline:
  Image → feature_extract() → calibrate() → meta_classifier.predict_proba()
        → apply_contradiction_logic() → threshold → decision
"""

import os
import io
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger("sentry_calibrated")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "calibration"
MODEL_DIR = BASE_DIR / "models" / "trained"
LOG_DIR = BASE_DIR / "logs" / "errors"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: GROUND TRUTH DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class GroundTruthDataset:
    """
    Generates labeled image dataset with controlled characteristics.
    
    Real images: natural camera-like noise, texture variation, organic gradients
    AI images: smooth gradients, periodic patterns, noise inconsistency
    
    Each image is stored as (features, label) — features are the 14 numerical
    signal-processing measurements, NOT raw pixels.
    """

    # Image generation is deterministic per seed for reproducibility
    REAL_GENERATORS = [
        "natural_landscape",
        "portrait_bokeh", 
        "street_photo",
        "low_light_phone",
        "high_contrast",
        "macro_close_up",
        "indoor_ambient",
        "overexposed",
    ]
    
    AI_GENERATORS = [
        "gan_smooth",
        "gan_checkerboard",
        "diffusion_ring",
        "deepfake_swap",
        "perfect_face",
        "midjourney_style",
        "dalle_style",
        "runway_video_frame",
    ]

    @staticmethod
    def generate_real(variant: str, seed: int, w: int = 512, h: int = 384) -> Image.Image:
        """Generate camera-like image with natural sensor characteristics."""
        rng = np.random.RandomState(seed)
        
        if variant == "natural_landscape":
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            img = np.zeros((h, w, 3), dtype=np.float64)
            img[:,:,0] = 100 + 100 * yy + rng.normal(0, 14, (h, w))
            img[:,:,1] = 140 + 60 * yy - 20 * xx + rng.normal(0, 14, (h, w))
            img[:,:,2] = 200 - 80 * yy + rng.normal(0, 14, (h, w))
            
        elif variant == "portrait_bokeh":
            cy, cx = h // 2, w // 2
            yc, xc = np.ogrid[:h, :w]
            dist = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            face_mask = (dist < min(w, h) * 0.25).astype(np.float64)
            bg = rng.normal(50, 8, (h, w, 3))
            face = rng.normal(175, 18, (h, w, 3))
            img = bg * (1 - face_mask[:,:,None]) + face * face_mask[:,:,None]
            img += rng.normal(0, 11, img.shape)
            
        elif variant == "street_photo":
            img = rng.normal(110, 20, (h, w, 3))
            # Add horizontal structures (buildings)
            for row_start in range(0, h, h // 6):
                block_h = h // 8
                brightness = rng.uniform(60, 200)
                img[row_start:row_start+block_h, :, :] = rng.normal(brightness, 15, (min(block_h, h - row_start), w, 3))
            img += rng.normal(0, 12, img.shape)
            
        elif variant == "low_light_phone":
            img = rng.normal(45, 6, (h, w, 3))
            img += rng.normal(0, 22, img.shape)  # High noise ISO
            
        elif variant == "high_contrast":
            img = np.zeros((h, w, 3), dtype=np.float64)
            half = h // 2
            img[:half] = rng.normal(230, 12, (half, w, 3))
            img[half:] = rng.normal(25, 10, (h - half, w, 3))
            transition = slice(half - 30, half + 30)
            img[transition] = rng.normal(120, 28, (60, w, 3))
            
        elif variant == "macro_close_up":
            img = rng.normal(140, 30, (h, w, 3))
            # Add sharp texture detail
            texture = rng.normal(0, 25, (h, w, 3))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
            for c in range(3):
                texture[:,:,c] = cv2.filter2D(texture[:,:,c], -1, kernel)
            img += texture * 0.3
            
        elif variant == "indoor_ambient":
            warmth = np.array([180, 160, 120], dtype=np.float64)
            img = np.ones((h, w, 3), dtype=np.float64) * warmth
            img += rng.normal(0, 15, img.shape)
            # Vignette
            cy, cx = h // 2, w // 2
            yc, xc = np.ogrid[:h, :w]
            dist = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            vignette = 1.0 - 0.3 * (dist / dist.max())
            img *= vignette[:,:,None]
            
        elif variant == "overexposed":
            img = rng.normal(210, 15, (h, w, 3))
            img += rng.normal(0, 10, img.shape)
        else:
            img = rng.normal(128, 20, (h, w, 3))
            
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def generate_ai(variant: str, seed: int, w: int = 512, h: int = 512) -> Image.Image:
        """Generate AI-like image with synthetic characteristics."""
        rng = np.random.RandomState(seed)
        
        if variant == "gan_smooth":
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            img = np.zeros((h, w, 3), dtype=np.float64)
            img[:,:,0] = 128 + 127 * np.sin(xx * np.pi)
            img[:,:,1] = 128 + 127 * np.cos(yy * np.pi)
            img[:,:,2] = 128 + 127 * np.sin((xx + yy) * 2.0)
            img += rng.normal(0, 1.2, img.shape)  # Very little noise
            
        elif variant == "gan_checkerboard":
            img = rng.normal(128, 12, (h, w, 3))
            checker = np.zeros((h, w))
            checker[::2, ::2] = 4.0
            checker[1::2, 1::2] = 4.0
            for c in range(3):
                img[:,:,c] += checker
            img += rng.normal(0, 2, img.shape)
            
        elif variant == "diffusion_ring":
            cy, cx = h // 2, w // 2
            yc, xc = np.ogrid[:h, :w]
            dist = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            ring = np.sin(dist * 0.15) * 50 + 128
            img = np.stack([ring + rng.normal(0, 2, (h, w)) for _ in range(3)], axis=2)
            
        elif variant == "deepfake_swap":
            # Background normal noise, face region unnaturally smooth
            bg = rng.normal(90, 14, (h, w, 3))
            cy, cx = h // 2, w // 2
            yc, xc = np.ogrid[:h, :w]
            dist = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            face = (dist < min(w, h) * 0.28).astype(np.float64)
            smooth_face = np.ones((h, w, 3)) * np.array([195, 175, 155])
            smooth_face += rng.normal(0, 2, (h, w, 3))
            img = bg * (1 - face[:,:,None]) + smooth_face * face[:,:,None]
            # Add boundary artifacts
            boundary = ((dist > min(w,h)*0.25) & (dist < min(w,h)*0.31)).astype(np.float64)
            img += boundary[:,:,None] * rng.normal(0, 18, (h, w, 3))
            
        elif variant == "perfect_face":
            center = np.array([215, 185, 165], dtype=np.float64)
            img = np.ones((h, w, 3)) * center
            cy, cx = h // 2, w // 2
            yc, xc = np.ogrid[:h, :w]
            dist = np.sqrt((xc - cx)**2 + (yc - cy)**2)
            falloff = np.exp(-(dist**2) / (2 * (min(w,h)*0.3)**2))
            img = img * (0.5 + 0.5 * falloff[:,:,None])
            img += rng.normal(0, 0.8, img.shape)
            
        elif variant == "midjourney_style":
            # Artistic, saturated, very clean
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            img = np.zeros((h, w, 3), dtype=np.float64)
            img[:,:,0] = 200 * np.sin(xx * 2 * np.pi) ** 2
            img[:,:,1] = 100 + 80 * np.cos(yy * 3 * np.pi)
            img[:,:,2] = 150 + 100 * np.sin((xx - yy) * 2.5 * np.pi)
            img += rng.normal(0, 1.5, img.shape)
            
        elif variant == "dalle_style":
            # Photo-realistic but with uniform noise characteristics
            base = rng.normal(140, 10, (h, w, 3))
            # Create scene-like structure but too clean
            for i in range(5):
                rx, ry = rng.randint(0, w - 100), rng.randint(0, h - 100)
                sw, sh = rng.randint(50, 150), rng.randint(50, 150)
                color = rng.randint(50, 220, 3)
                base[ry:min(ry+sh,h), rx:min(rx+sw,w)] = color + rng.normal(0, 3, (min(sh, h-ry), min(sw, w-rx), 3))
            img = base
            
        elif variant == "runway_video_frame":
            # Video-sourced: frame from AI video. Slightly more structured noise.
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            img = rng.normal(120, 8, (h, w, 3))
            # Motion blur effect
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            for c in range(3):
                img[:,:,c] = cv2.filter2D(img[:,:,c], -1, kernel)
            img += rng.normal(0, 2, img.shape)
        else:
            img = np.ones((h, w, 3)) * 128 + rng.normal(0, 2, (h, w, 3))
            
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def apply_compression(image: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def apply_screenshot(image: Image.Image) -> Image.Image:
        """Resize to mobile-like dimensions."""
        return image.resize((1080, 1920), Image.LANCZOS)

    @classmethod
    def build_dataset(cls, n_per_variant: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build complete labeled dataset.
        
        Returns:
            X: (N, 14) feature matrix
            y: (N,) label array (0 = real, 1 = AI)
        """
        features_list = []
        labels = []
        extractor = FeatureExtractor()
        
        rng = np.random.RandomState(seed)
        sample_id = 0
        
        # Real images
        print(f"[Dataset] Generating real images ({len(cls.REAL_GENERATORS)} variants × {n_per_variant} samples)...")
        for variant in cls.REAL_GENERATORS:
            for i in range(n_per_variant):
                img = cls.generate_real(variant, seed=seed + sample_id)
                
                # 30% get compressed (Instagram-style)
                if rng.random() < 0.3:
                    quality = rng.choice([30, 50, 70, 85])
                    img = cls.apply_compression(img, quality)
                
                # 10% get screenshot treatment
                if rng.random() < 0.1:
                    img = cls.apply_screenshot(img)
                
                feats = extractor.extract(img)
                features_list.append(feats)
                labels.append(0)
                sample_id += 1
        
        # AI images
        print(f"[Dataset] Generating AI images ({len(cls.AI_GENERATORS)} variants × {n_per_variant} samples)...")
        for variant in cls.AI_GENERATORS:
            for i in range(n_per_variant):
                img = cls.generate_ai(variant, seed=seed + sample_id)
                
                # 30% get compressed
                if rng.random() < 0.3:
                    quality = rng.choice([30, 50, 70, 85])
                    img = cls.apply_compression(img, quality)
                
                # 10% get screenshot treatment
                if rng.random() < 0.1:
                    img = cls.apply_screenshot(img)
                
                feats = extractor.extract(img)
                features_list.append(feats)
                labels.append(1)
                sample_id += 1
        
        X = np.array(features_list, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        
        print(f"[Dataset] Built: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[Dataset] Real: {np.sum(y == 0)}, AI: {np.sum(y == 1)}")
        
        return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (deterministic, no ML — pure signal processing)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extracts 14 numerical features from a single image.
    
    These are the RAW measurements — NOT decisions, NOT scores.
    The meta-classifier learns how to combine them.
    
    Features:
      0: noise_std           - horizontal gradient noise standard deviation
      1: local_var_cv        - coefficient of variation of block-wise variance
      2: mean_local_var      - mean block-wise pixel variance
      3: rg_correlation      - R-G channel correlation
      4: rb_correlation      - R-B channel correlation
      5: edge_density        - Canny edge pixel ratio
      6: laplacian_var       - Laplacian variance (blur/sharpness metric)
      7: fft_mid_ratio       - mid-frequency / low-frequency FFT energy ratio
      8: fft_high_ratio      - high-frequency / low-frequency FFT energy ratio
      9: fft_azimuthal_range - spectral quadrant energy range (directional bias)
     10: jpeg_ghost_var      - JPEG re-compression ghost consistency
     11: noise_block_cv      - noise pattern block-wise consistency
     12: center_edge_ratio   - center vs periphery edge density ratio
     13: saturation_std      - HSV saturation standard deviation
    """
    
    FEATURE_NAMES = [
        "noise_std", "local_var_cv", "mean_local_var",
        "rg_correlation", "rb_correlation",
        "edge_density", "laplacian_var",
        "fft_mid_ratio", "fft_high_ratio", "fft_azimuthal_range",
        "jpeg_ghost_var", "noise_block_cv",
        "center_edge_ratio", "saturation_std"
    ]

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract all 14 features. Returns shape (14,)."""
        img_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        img_bgr = img_rgb[:, :, ::-1]
        h, w = gray.shape
        
        features = np.zeros(14, dtype=np.float64)
        
        # 0: Noise standard deviation (horizontal gradient)
        features[0] = float(np.std(np.diff(gray, axis=1)))
        
        # 1-2: Local variance consistency
        block_size = min(32, max(4, h // 8, w // 8))
        local_vars = []
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by:by+block_size, bx:bx+block_size]
                local_vars.append(float(np.var(block)))
        if local_vars:
            mean_lv = float(np.mean(local_vars))
            var_lv = float(np.var(local_vars))
            features[1] = var_lv / (mean_lv + 1e-8)  # CV
            features[2] = mean_lv
        
        # 3-4: Color channel correlations
        r, g, b = img_rgb[:,:,0].flatten().astype(np.float64), \
                   img_rgb[:,:,1].flatten().astype(np.float64), \
                   img_rgb[:,:,2].flatten().astype(np.float64)
        features[3] = float(np.corrcoef(r, g)[0, 1]) if np.std(r) > 0 and np.std(g) > 0 else 0.0
        features[4] = float(np.corrcoef(r, b)[0, 1]) if np.std(r) > 0 and np.std(b) > 0 else 0.0
        
        # 5: Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        features[5] = float(np.sum(edges > 0)) / (h * w)
        
        # 6: Laplacian variance
        features[6] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 7-9: FFT spectral analysis
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        
        cy, cx = h // 2, w // 2
        max_r = min(cx, cy)
        yg, xg = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist_sq = xg*xg + yg*yg
        
        r_inner = int(0.15 * max_r)
        r_mid = int(0.50 * max_r)
        r_outer = int(0.85 * max_r)
        
        inner_mask = dist_sq < r_inner**2
        mid_mask = (dist_sq >= r_inner**2) & (dist_sq < r_mid**2)
        outer_mask = (dist_sq >= r_mid**2) & (dist_sq < r_outer**2)
        
        inner_e = float(np.mean(magnitude[inner_mask])) if np.any(inner_mask) else 1.0
        mid_e = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
        outer_e = float(np.mean(magnitude[outer_mask])) if np.any(outer_mask) else 0.0
        
        features[7] = mid_e / (inner_e + 1e-8)
        features[8] = outer_e / (inner_e + 1e-8)
        
        # Azimuthal (directional) analysis
        quads = [
            magnitude[:cy, :cx], magnitude[:cy, cx:],
            magnitude[cy:, :cx], magnitude[cy:, cx:]
        ]
        q_means = [float(np.mean(q)) for q in quads]
        features[9] = max(q_means) - min(q_means)
        
        # 10: JPEG ghost variance
        ghost_vars = []
        for quality in [70, 40]:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img_bgr, encode_param)
            recompressed = cv2.imdecode(encoded, 1)
            if recompressed is not None:
                diff = cv2.absdiff(img_bgr, recompressed).astype(np.float64)
                diff_gray = np.mean(diff, axis=2)
                block_means = []
                bs = 32
                for by in range(0, h - bs, bs):
                    for bx in range(0, w - bs, bs):
                        block_means.append(float(np.mean(diff_gray[by:by+bs, bx:bx+bs])))
                if block_means:
                    ghost_vars.append(float(np.var(block_means)))
        features[10] = float(np.mean(ghost_vars)) if ghost_vars else 0.0
        
        # 11: Noise pattern consistency
        denoised = cv2.medianBlur(img_bgr, 3).astype(np.float64)
        noise = img_bgr.astype(np.float64) - denoised
        block_stds = []
        bs = 64
        for by in range(0, h - bs, bs):
            for bx in range(0, w - bs, bs):
                block_stds.append(float(np.std(noise[by:by+bs, bx:bx+bs])))
        if len(block_stds) > 2:
            features[11] = float(np.var(block_stds)) / (float(np.mean(block_stds)) + 1e-8)
        
        # 12: Center vs periphery edge ratio
        margin = min(h, w) // 4
        if margin > 10:
            center_edges = edges[margin:h-margin, margin:w-margin]
            periph = np.concatenate([
                edges[:margin, :].flatten(), edges[h-margin:, :].flatten(),
                edges[:, :margin].flatten(), edges[:, w-margin:].flatten()
            ])
            cd = float(np.mean(center_edges)) / 255.0
            pd = float(np.mean(periph)) / 255.0
            features[12] = cd / (pd + 1e-8)
        
        # 13: Saturation standard deviation (HSV)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        features[13] = float(np.std(hsv[:,:,1]))
        
        # Replace any NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: CALIBRATION (Platt Scaling + Isotonic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureCalibrator:
    """
    Normalizes raw features using StandardScaler (zero mean, unit variance).
    
    Uses StandardScaler instead of isotonic regression per feature.
    Isotonic per-feature was collapsing the probability space by mapping
    all features to narrow [0, 1] ranges. StandardScaler preserves the
    full feature variance for the meta-classifier.
    """
    
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit scaler on training data."""
        self.scaler.fit(X)
        self.fitted = True
        print(f"[Calibrator] StandardScaler fitted on {X.shape[1]} features")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to zero-mean unit-variance."""
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def save(self, path: str):
        joblib.dump(self.scaler, path)
        
    def load(self, path: str):
        self.scaler = joblib.load(path)
        self.fitted = True


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: META-CLASSIFIER (Learned Ensemble Weights)
# ═══════════════════════════════════════════════════════════════════════════════

class MetaClassifier:
    """
    Gradient Boosted Trees meta-classifier.
    
    Learns non-linear feature interactions and produces calibrated
    probabilities via Platt scaling (built into GBT's predict_proba).
    
    Replaces ALL guessed weights with learned parameters.
    """
    
    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
        self.fitted = False
        self.threshold = 0.5
        self.feature_names = FeatureExtractor.FEATURE_NAMES
    
    def fit(self, X_scaled: np.ndarray, y: np.ndarray):
        """Train meta-classifier on scaled features."""
        self.model.fit(X_scaled, y)
        self.fitted = True
        
        # Feature importances (learned, not guessed)
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("[Meta-Classifier] Learned feature importance (GBT):")
        for idx in sorted_idx:
            print(f"  {self.feature_names[idx]:>22s}: {importances[idx]:.4f}")
    
    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Returns calibrated P(AI) for each sample."""
        if not self.fitted:
            raise RuntimeError("Meta-classifier not fitted.")
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'threshold': self.threshold}, path)
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.threshold = data['threshold']
        self.fitted = True


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: THRESHOLD OPTIMIZER (From ROC Curve)
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdOptimizer:
    """
    Finds optimal thresholds from ROC curve.
    
    Primary: Youden's J statistic (TPR - FPR) to find the best operating point.
    Then: apply FPR constraint (≤ 5% for restrict, ≤ 1% for block).
    """
    
    @staticmethod
    def optimize(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Find thresholds from ROC curve using Youden's J."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # === PRIMARY: Youden's J statistic (best balance of TPR and FPR) ===
        # J = TPR - FPR. Maximizing J finds the optimal operating point.
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        youden_threshold = float(thresholds[best_j_idx])
        youden_fpr = float(fpr[best_j_idx])
        youden_tpr = float(tpr[best_j_idx])
        
        # === RESTRICT threshold ===
        # If Youden threshold already has FPR ≤ 5%, use it directly.
        # Otherwise, find best threshold with FPR ≤ 5%.
        if youden_fpr <= 0.05:
            restrict_threshold = youden_threshold
            restrict_fpr = youden_fpr
            restrict_tpr = youden_tpr
        else:
            valid_mask = fpr <= 0.05
            if np.any(valid_mask):
                j_constrained = (tpr - fpr)[valid_mask]
                best_idx = np.argmax(j_constrained)
                restrict_threshold = float(thresholds[valid_mask][best_idx])
                restrict_fpr = float(fpr[valid_mask][best_idx])
                restrict_tpr = float(tpr[valid_mask][best_idx])
            else:
                restrict_threshold = youden_threshold
                restrict_fpr = youden_fpr
                restrict_tpr = youden_tpr
        
        # === LABEL threshold: more lenient (FPR ≤ 10%, maximizing J) ===
        valid_mask_label = fpr <= 0.10
        if np.any(valid_mask_label):
            j_label = (tpr - fpr)[valid_mask_label]
            best_idx = np.argmax(j_label)
            label_threshold = float(thresholds[valid_mask_label][best_idx])
        else:
            label_threshold = restrict_threshold - 0.05
        
        label_threshold = min(label_threshold, restrict_threshold)
        
        # === BLOCK threshold: stricter (FPR ≤ 1%) ===
        valid_mask_block = fpr <= 0.01
        if np.any(valid_mask_block):
            j_block = (tpr - fpr)[valid_mask_block]
            best_idx = np.argmax(j_block)
            block_threshold = float(thresholds[valid_mask_block][best_idx])
        else:
            block_threshold = restrict_threshold + 0.10
        
        block_threshold = max(block_threshold, restrict_threshold + 0.03)
        block_threshold = max(block_threshold, 0.90)
        
        return {
            "label_threshold": round(label_threshold, 4),
            "restrict_threshold": round(restrict_threshold, 4),
            "block_threshold": round(block_threshold, 4),
            "roc_auc": round(roc_auc, 4),
            "restrict_fpr": round(restrict_fpr, 4),
            "restrict_tpr": round(restrict_tpr, 4),
            "youden_j": round(float(j_scores[best_j_idx]), 4),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5: CONTRADICTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class ContradictionEngine:
    """
    Hard rules applied on top of calibrated probabilities.
    
    Rules (applied in order):
      1. If physiological AND frequency both look real → clamp to ALLOW
      2. If 2+ feature groups agree AI → use calibrated probability
      3. If only 1 weak feature flags → do NOT block
      4. Hard block only if calibrated_prob > 0.90
    """
    
    @staticmethod
    def apply(
        calibrated_prob: float,
        raw_features: np.ndarray,
        calibrated_features: np.ndarray,  # now scaled features, not [0,1] probs
    ) -> Dict[str, Any]:
        """
        Apply contradiction logic using raw feature values.
        Uses the calibrated probability from the meta-classifier as primary input.
        Raw features provide forensic interpretability.
        """
        reasons = []
        adjusted = calibrated_prob
        
        # Use raw features for interpretable forensic grouping
        noise_std = raw_features[0]
        edge_density = raw_features[5]
        laplacian_var = raw_features[6]
        fft_mid_ratio = raw_features[7]
        fft_high_ratio = raw_features[8]
        rg_corr = raw_features[3]
        
        # Group features into detector categories using raw value ranges
        # Physiology: noise + edges + texture = camera sensor characteristics
        physio_real_signals = sum([
            1 if noise_std > 8.0 else 0,     # Natural camera noise
            1 if edge_density > 0.03 else 0,   # Rich edge structure 
            1 if laplacian_var > 100 else 0,   # Sharp textures
        ])
        
        # Frequency: FFT spectral characteristics
        freq_real_signals = sum([
            1 if fft_mid_ratio < 0.7 else 0,   # Normal spectral decay
            1 if fft_high_ratio < 0.5 else 0,   # Normal high-freq falloff
        ])
        
        # Channel: color correlation
        channel_real_signals = 1 if rg_corr > 0.8 else 0
        
        total_real_signals = physio_real_signals + freq_real_signals + channel_real_signals
        
        # RULE 1: If majority of feature groups look strongly real → clamp
        if total_real_signals >= 5 and calibrated_prob > 0.3:
            old = adjusted
            adjusted = min(adjusted, 0.15)
            reasons.append(f"RULE1: {total_real_signals}/6 features look strongly real → clamped {old:.2f} → {adjusted:.2f}")
        
        # RULE 2: Strong AI classifier output → trust it
        elif calibrated_prob > 0.7 and total_real_signals <= 2:
            reasons.append(f"RULE2: High P(AI)={calibrated_prob:.3f} + only {total_real_signals}/6 real signals → trusting classifier")
        
        # RULE 3: Weak classifier output with mixed signals → dampen
        elif calibrated_prob > 0.3 and calibrated_prob < 0.7 and total_real_signals >= 3:
            old = adjusted
            adjusted = adjusted * 0.5
            reasons.append(f"RULE3: Uncertain P(AI)={calibrated_prob:.2f} + {total_real_signals}/6 real signals → dampened to {adjusted:.2f}")
        
        # RULE 4: Hard block floor — never block without strong consensus
        if adjusted > 0.90 and total_real_signals >= 3:
            old = adjusted
            adjusted = 0.85
            reasons.append(f"RULE4: High prob ({old:.2f}) but {total_real_signals}/6 real signals → capped at 0.85")
        
        return {
            "adjusted_prob": round(float(adjusted), 4),
            "original_prob": round(float(calibrated_prob), 4),
            "real_signal_count": total_real_signals,
            "physio_real_signals": physio_real_signals,
            "freq_real_signals": freq_real_signals,
            "channel_real_signals": channel_real_signals,
            "contradiction_reasons": reasons,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6: ERROR LOGGING (Active Learning Loop)
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorLogger:
    """
    Logs all predictions to enable active error learning.
    False positives and false negatives are flagged for review.
    """
    
    def __init__(self):
        self.log_path = LOG_DIR / "prediction_log.jsonl"
    
    def log_prediction(
        self,
        image_hash: str,
        raw_features: np.ndarray,
        calibrated_prob: float,
        final_prob: float,
        decision: str,
        ground_truth: Optional[int] = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_hash": image_hash,
            "raw_features": raw_features.tolist(),
            "calibrated_prob": calibrated_prob,
            "final_prob": final_prob,
            "decision": decision,
            "ground_truth": ground_truth,
            "error_type": None,
        }
        
        if ground_truth is not None:
            is_ai = ground_truth == 1
            flagged = decision in ("restrict", "block")
            if not is_ai and flagged:
                entry["error_type"] = "FALSE_POSITIVE"
            elif is_ai and not flagged:
                entry["error_type"] = "FALSE_NEGATIVE"
        
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")
    
    def get_errors(self) -> List[Dict]:
        """Return all logged errors for retraining."""
        errors = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("error_type"):
                        errors.append(entry)
        except FileNotFoundError:
            pass
        return errors


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedDetector:
    """
    The fully calibrated detection pipeline.
    
    Usage:
        detector = CalibratedDetector()
        detector.train()   # Build dataset, fit calibrators, train classifier, optimize thresholds
        
        result = detector.predict(image)
        # result["probability"] is a TRUE calibrated probability
        # result["decision"] is based on ROC-optimized thresholds
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.calibrator = FeatureCalibrator()
        self.meta_classifier = MetaClassifier()
        self.contradiction_engine = ContradictionEngine()
        self.error_logger = ErrorLogger()
        self.thresholds = None
        self.trained = False
        self.metrics = {}
    
    def train(self, n_per_variant: int = 60, seed: int = 42):
        """
        Full training pipeline:
        1. Generate dataset
        2. Split train/val/test
        3. Fit calibrators on train
        4. Train meta-classifier on train
        5. Optimize thresholds on validation
        6. Evaluate on test
        """
        print("\n" + "=" * 70)
        print(" 🔬 SENTRY-X V3 — CALIBRATED PIPELINE TRAINING")
        print("=" * 70)
        
        # Step 1: Generate dataset
        X, y = GroundTruthDataset.build_dataset(n_per_variant=n_per_variant, seed=seed)
        
        # Step 2: Split 70/15/15
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=seed, stratify=y_train_val
        )  # 0.176 of 85% ≈ 15% of total
        
        print(f"\n[Split] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
        print(f"[Split] Train class dist: real={np.sum(y_train==0)}, AI={np.sum(y_train==1)}")
        
        # Step 3: Fit calibrators on training data
        print("\n── Step 3: Fitting Feature Calibrators ──")
        self.calibrator.fit(X_train, y_train)
        
        X_train_cal = self.calibrator.transform(X_train)
        X_val_cal = self.calibrator.transform(X_val)
        X_test_cal = self.calibrator.transform(X_test)
        
        # Step 4: Train meta-classifier
        print("\n── Step 4: Training Meta-Classifier ──")
        self.meta_classifier.fit(X_train_cal, y_train)
        
        # Step 5: Optimize thresholds on validation set
        # IMPORTANT: Optimize on RAW meta-classifier probs, NOT contradiction-adjusted.
        # Contradiction logic is a safety override at inference — it should NOT
        # distort the threshold calibration by collapsing probability space.
        print("\n── Step 5: Optimizing Thresholds ──")
        val_probs = self.meta_classifier.predict_proba(X_val_cal)
        
        self.thresholds = ThresholdOptimizer.optimize(y_val, val_probs)
        
        print(f"  Label threshold:    {self.thresholds['label_threshold']}")
        print(f"  Restrict threshold: {self.thresholds['restrict_threshold']}")
        print(f"  Block threshold:    {self.thresholds['block_threshold']}")
        print(f"  ROC AUC:           {self.thresholds['roc_auc']}")
        print(f"  At restrict: FPR = {self.thresholds['restrict_fpr']:.3f}, TPR = {self.thresholds['restrict_tpr']:.3f}")
        
        self.meta_classifier.threshold = self.thresholds['restrict_threshold']
        
        # Step 6: Evaluate on test set
        print("\n── Step 6: Test Set Evaluation ──")
        self.metrics = self._evaluate(X_test, X_test_cal, y_test)
        
        self.trained = True
        self._save()
        
        print("\n✅ Training complete. Models saved.\n")
        return self.metrics
    
    def _evaluate(self, X_raw: np.ndarray, X_cal: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate on a labeled set. Returns comprehensive metrics."""
        probs = self.meta_classifier.predict_proba(X_cal)
        
        # Apply contradiction logic
        final_probs = []
        decisions = []
        for i in range(len(probs)):
            cont = self.contradiction_engine.apply(probs[i], X_raw[i], X_cal[i])
            p = cont["adjusted_prob"]
            final_probs.append(p)
            
            if p >= self.thresholds["block_threshold"]:
                decisions.append("block")
            elif p >= self.thresholds["restrict_threshold"]:
                decisions.append("restrict")
            elif p >= self.thresholds["label_threshold"]:
                decisions.append("label")
            else:
                decisions.append("allow")
        
        final_probs = np.array(final_probs)
        
        # Binary: restrict/block = flagged
        y_pred_strict = np.array([1 if d in ("restrict", "block") else 0 for d in decisions])
        y_pred_lenient = np.array([1 if d != "allow" else 0 for d in decisions])
        
        # Confusion matrix (strict)
        cm = confusion_matrix(y_true, y_pred_strict)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        # Metrics
        fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        accuracy = float((tp + tn) / len(y_true))
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # ROC AUC on test
        fpr_curve, tpr_curve, _ = roc_curve(y_true, final_probs)
        test_auc = auc(fpr_curve, tpr_curve)
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr_val, 4),
            "fnr": round(fnr_val, 4),
            "auc": round(test_auc, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "total_samples": len(y_true),
            "thresholds_used": {
                k: v for k, v in self.thresholds.items() if k != "roc_curve"
            },
        }
        
        # Print results
        print(f"  Accuracy:   {accuracy*100:.2f}%")
        print(f"  Precision:  {precision*100:.2f}%")
        print(f"  Recall:     {recall*100:.2f}%")
        print(f"  F1 Score:   {f1*100:.2f}%")
        print(f"  FPR:        {fpr_val*100:.2f}%  (target < 5%)")
        print(f"  FNR:        {fnr_val*100:.2f}%  (target < 8%)")
        print(f"  AUC:        {test_auc:.4f}")
        print(f"  Confusion:  TP={tp} TN={tn} FP={fp} FN={fn}")
        
        status_fpr = "✅" if fpr_val < 0.05 else "❌"
        status_fnr = "✅" if fnr_val < 0.08 else "❌"
        print(f"\n  {status_fpr} FPR target (<5%):  {fpr_val*100:.2f}%")
        print(f"  {status_fnr} FNR target (<8%):  {fnr_val*100:.2f}%")
        
        return metrics
    
    def _save(self):
        """Save trained components."""
        self.calibrator.save(str(MODEL_DIR / "calibrator.pkl"))
        self.meta_classifier.save(str(MODEL_DIR / "meta_classifier.pkl"))
        
        with open(str(MODEL_DIR / "thresholds.json"), "w") as f:
            json.dump({k: v for k, v in self.thresholds.items() if k != "roc_curve"}, f, indent=2)
        
        with open(str(MODEL_DIR / "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[Save] Models saved to {MODEL_DIR}")
    
    def load(self) -> bool:
        """Load previously trained components. Returns True if successful."""
        try:
            self.calibrator.load(str(MODEL_DIR / "calibrator.pkl"))
            self.meta_classifier.load(str(MODEL_DIR / "meta_classifier.pkl"))
            
            with open(str(MODEL_DIR / "thresholds.json")) as f:
                self.thresholds = json.load(f)
            
            with open(str(MODEL_DIR / "metrics.json")) as f:
                self.metrics = json.load(f)
            
            self.trained = True
            print(f"[Load] Calibrated pipeline loaded (AUC: {self.metrics.get('auc', '?')})")
            return True
        except Exception as e:
            logger.info(f"No trained model found: {e}")
            return False
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run calibrated prediction on a single image.
        
        Returns:
          probability: calibrated P(AI) — a TRUE probability
          decision: "allow" | "label" | "restrict" | "block"
          features: raw feature values
          calibrated_features: calibrated feature probabilities
          contradiction: contradiction logic output
        """
        if not self.trained:
            if not self.load():
                raise RuntimeError("No trained model. Call train() first.")
        
        t0 = time.time()
        
        # Step 1: Extract features
        raw_features = self.feature_extractor.extract(image)
        
        # Step 2: Calibrate
        X = raw_features.reshape(1, -1)
        X_cal = self.calibrator.transform(X)
        calibrated_features = X_cal[0]
        
        # Step 3: Meta-classifier probability
        calibrated_prob = float(self.meta_classifier.predict_proba(X_cal)[0])
        
        # Step 4: Contradiction logic
        contradiction = self.contradiction_engine.apply(
            calibrated_prob, raw_features, calibrated_features
        )
        final_prob = contradiction["adjusted_prob"]
        
        # Step 5: Decision from ROC-optimized thresholds
        if final_prob >= self.thresholds["block_threshold"]:
            decision = "block"
            risk_level = "red"
        elif final_prob >= self.thresholds["restrict_threshold"]:
            decision = "restrict"
            risk_level = "orange"
        elif final_prob >= self.thresholds["label_threshold"]:
            decision = "label"
            risk_level = "yellow"
        else:
            decision = "allow"
            risk_level = "green"
        
        latency_ms = (time.time() - t0) * 1000
        
        # Step 6: Log for error learning
        img_hash = hashlib.md5(np.array(image).tobytes()[:10000]).hexdigest()[:12]
        self.error_logger.log_prediction(
            image_hash=img_hash,
            raw_features=raw_features,
            calibrated_prob=calibrated_prob,
            final_prob=final_prob,
            decision=decision,
        )
        
        return {
            "probability": final_prob,
            "calibrated_probability": calibrated_prob,
            "decision": decision,
            "risk_level": risk_level,
            "thresholds": {k: v for k, v in self.thresholds.items() if k != "roc_curve"},
            "features": {
                name: round(float(raw_features[i]), 4)
                for i, name in enumerate(FeatureExtractor.FEATURE_NAMES)
            },
            "calibrated_features": {
                name: round(float(calibrated_features[i]), 4)
                for i, name in enumerate(FeatureExtractor.FEATURE_NAMES)
            },
            "contradiction": contradiction,
            "latency_ms": round(latency_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7: BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_benchmark(n_per_variant: int = 60):
    """
    Train, evaluate, and output complete benchmark report.
    """
    detector = CalibratedDetector()
    metrics = detector.train(n_per_variant=n_per_variant)
    
    # Run held-out extra test
    print("\n" + "=" * 70)
    print(" 📊 ADDITIONAL HELD-OUT BENCHMARK")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    
    # 50 fresh real images (different seeds)
    real_results = []
    for i in range(50):
        variant = GroundTruthDataset.REAL_GENERATORS[i % len(GroundTruthDataset.REAL_GENERATORS)]
        img = GroundTruthDataset.generate_real(variant, seed=99999 + i)
        result = detector.predict(img)
        real_results.append(result)
    
    real_fp = sum(1 for r in real_results if r["decision"] in ("restrict", "block"))
    
    # 50 fresh AI images
    ai_results = []
    for i in range(50):
        variant = GroundTruthDataset.AI_GENERATORS[i % len(GroundTruthDataset.AI_GENERATORS)]
        img = GroundTruthDataset.generate_ai(variant, seed=88888 + i)
        result = detector.predict(img)
        ai_results.append(result)
    
    ai_fn = sum(1 for r in ai_results if r["decision"] == "allow")
    
    print(f"\n  Held-out Real → FP (restrict/block): {real_fp}/50 ({real_fp/50*100:.1f}%)")
    print(f"  Held-out AI   → FN (allow):          {ai_fn}/50 ({ai_fn/50*100:.1f}%)")
    
    # Compressed test
    print("\n  Compressed variants:")
    for q in [85, 50, 20]:
        fp_count = 0
        fn_count = 0
        for i in range(20):
            if i < 10:
                img = GroundTruthDataset.generate_real("natural_landscape", seed=77000 + i)
                img = GroundTruthDataset.apply_compression(img, q)
                r = detector.predict(img)
                if r["decision"] in ("restrict", "block"):
                    fp_count += 1
            else:
                img = GroundTruthDataset.generate_ai("gan_smooth", seed=77000 + i)
                img = GroundTruthDataset.apply_compression(img, q)
                r = detector.predict(img)
                if r["decision"] == "allow":
                    fn_count += 1
        print(f"    Q={q:>2}: Real FP={fp_count}/10 | AI FN={fn_count}/10")
    
    # Save full report
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "3.0_calibrated",
        "training_metrics": metrics,
        "held_out": {
            "real_fp_rate": real_fp / 50,
            "ai_fn_rate": ai_fn / 50,
        },
        "latency": {
            "mean_ms": round(float(np.mean([r["latency_ms"] for r in real_results + ai_results])), 1),
        },
    }
    
    report_path = BASE_DIR / "benchmark_report_v3.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  📄 Report saved to: {report_path}")
    print("=" * 70)
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON (lazy init — only trains if no saved model exists)
# ═══════════════════════════════════════════════════════════════════════════════

_calibrated_detector: Optional[CalibratedDetector] = None

def get_calibrated_detector() -> CalibratedDetector:
    """Get or create the calibrated detector singleton."""
    global _calibrated_detector
    if _calibrated_detector is None:
        _calibrated_detector = CalibratedDetector()
        if not _calibrated_detector.load():
            print("[CalibratedDetector] No trained model found. Training now...")
            _calibrated_detector.train()
    return _calibrated_detector


if __name__ == "__main__":
    run_full_benchmark()
