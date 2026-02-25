"""
models/verification_layer.py

SENTRY-X External Verification Pass (TASK 3)
=============================================

When internal ensemble detection produces a confidence in the 
uncertain zone (0.35–0.85), this module runs additional verification:

1. Enhanced FFT + Noise Analysis (frequency domain forensics)
2. JPEG Ghost Analysis (re-compression artifact detection)
3. Edge coherence analysis (boundary artifact detection)
4. Noise pattern consistency check

This provides orthogonal evidence to break ties in the
uncertain confidence range without relying on external APIs,
keeping the system free and self-contained.

In production, this could optionally call:
- HuggingFace Inference API (free tier)
- Illuminarty API
- Other external verification services
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("sentry_verification")


class VerificationLayer:
    """
    Additional forensic verification for uncertain cases.
    Uses signal-processing techniques that are orthogonal to CNN-based detection.
    """

    def __init__(self):
        self.name = "verification_layer"
        # Confidence range that triggers verification
        self.trigger_low = 0.35
        self.trigger_high = 0.85

    def should_verify(self, ensemble_score: float) -> bool:
        """Check if the score falls in the uncertain zone."""
        return self.trigger_low <= ensemble_score <= self.trigger_high

    def verify(self, image: Image.Image, ensemble_score: float) -> Dict[str, Any]:
        """
        Run full verification suite.
        Returns adjusted score and verification signals.
        """
        signals = []
        adjustments = []
        
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]  # BGR for OpenCV
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # ── Test 1: Enhanced FFT Spectral Analysis ──
        fft_result = self._enhanced_fft_analysis(gray)
        signals.extend(fft_result["signals"])
        adjustments.append(fft_result["adjustment"])
        
        # ── Test 2: JPEG Ghost Analysis ──
        ghost_result = self._jpeg_ghost_analysis(img_np)
        signals.extend(ghost_result["signals"])
        adjustments.append(ghost_result["adjustment"])
        
        # ── Test 3: Edge Coherence Analysis ──
        edge_result = self._edge_coherence_analysis(gray)
        signals.extend(edge_result["signals"])
        adjustments.append(edge_result["adjustment"])
        
        # ── Test 4: Noise Pattern Consistency ──
        noise_result = self._noise_consistency_analysis(img_np)
        signals.extend(noise_result["signals"])
        adjustments.append(noise_result["adjustment"])
        
        # ── Test 5: Color Channel Correlation Analysis ──
        color_result = self._color_correlation_analysis(img_np)
        signals.extend(color_result["signals"])
        adjustments.append(color_result["adjustment"])
        
        # Calculate net adjustment (average of all test adjustments)
        net_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0.0
        
        # Apply adjustment to ensemble score
        verified_score = max(0.0, min(1.0, ensemble_score + net_adjustment))
        
        # Count how many verification tests flagged suspicious
        suspicious_tests = sum(1 for a in adjustments if a > 0.02)
        clean_tests = sum(1 for a in adjustments if a < -0.02)
        total_tests = len(adjustments)
        
        verification_verdict = "inconclusive"
        if suspicious_tests >= 3:
            verification_verdict = "likely_synthetic"
            verified_score = max(verified_score, ensemble_score + 0.08)
        elif clean_tests >= 3:
            verification_verdict = "likely_authentic"
            verified_score = min(verified_score, ensemble_score - 0.05)
        elif suspicious_tests >= 2:
            verification_verdict = "suspicious"
        
        verified_score = max(0.0, min(1.0, verified_score))
        
        return {
            "verified_score": round(verified_score, 4),
            "original_score": round(ensemble_score, 4),
            "net_adjustment": round(net_adjustment, 4),
            "verification_verdict": verification_verdict,
            "tests_run": total_tests,
            "tests_suspicious": suspicious_tests,
            "tests_clean": clean_tests,
            "signals": signals,
            "individual_adjustments": [round(a, 4) for a in adjustments],
        }

    def _enhanced_fft_analysis(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced frequency domain analysis.
        Looks for:
        - Periodic patterns in mid-frequencies (GAN checkerboard)
        - Unusual spectral energy distribution
        - Azimuthal asymmetry (directional artifacts)
        """
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Radial profile analysis
        signals = []
        adjustment = 0.0
        
        # 1. Check for periodic peaks (GAN fingerprint)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y)
        
        # Sample energy at different radii
        max_radius = min(cx, cy)
        radial_energies = []
        for r in range(10, max_radius, 5):
            ring_mask = (dist >= r - 2) & (dist <= r + 2)
            if np.any(ring_mask):
                radial_energies.append(float(np.mean(magnitude[ring_mask])))
        
        if len(radial_energies) > 5:
            # Check for unusual bumps in radial energy (GAN artifacts)
            diffs = np.diff(radial_energies)
            positive_bumps = sum(1 for d in diffs if d > 0.3)
            
            if positive_bumps > len(diffs) * 0.3:
                adjustment += 0.06
                signals.append("[FFT-Verify] Non-monotonic spectral decay — possible GAN upsampling artifact")
        
        # 2. Azimuthal analysis (directional artifacts)
        # Real images have isotropic frequency content; GANs can have directional bias
        quadrants = [
            magnitude[:cy, :cx],  # top-left
            magnitude[:cy, cx:],  # top-right
            magnitude[cy:, :cx],  # bottom-left
            magnitude[cy:, cx:],  # bottom-right
        ]
        q_means = [float(np.mean(q)) for q in quadrants]
        q_range = max(q_means) - min(q_means)
        
        if q_range > 1.5:
            adjustment += 0.04
            signals.append(f"[FFT-Verify] Azimuthal spectral asymmetry (range: {q_range:.2f}) — directional processing artifact")
        
        if not signals:
            adjustment -= 0.02
            signals.append("[FFT-Verify] Spectral distribution consistent with natural photography")
        
        return {"adjustment": adjustment, "signals": signals}

    def _jpeg_ghost_analysis(self, img_np: np.ndarray) -> Dict[str, Any]:
        """
        JPEG ghost detection: re-compress at various quality levels
        and measure difference. Real images show uniform ghosting;
        manipulated regions show inconsistent ghosting patterns.
        """
        signals = []
        adjustment = 0.0
        
        ghost_variances = []
        for quality in [75, 50, 30]:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img_np, encode_param)
            recompressed = cv2.imdecode(encoded, 1)
            
            diff = cv2.absdiff(img_np, recompressed).astype(np.float32)
            diff_gray = np.mean(diff, axis=2)
            
            # Measure spatial variance of the ghost (should be uniform for untampered)
            block_size = 32
            h, w = diff_gray.shape
            block_means = []
            for by in range(0, h - block_size, block_size):
                for bx in range(0, w - block_size, block_size):
                    block = diff_gray[by:by+block_size, bx:bx+block_size]
                    block_means.append(float(np.mean(block)))
            
            if block_means:
                ghost_variances.append(float(np.var(block_means)))
        
        if ghost_variances:
            avg_ghost_var = sum(ghost_variances) / len(ghost_variances)
            
            if avg_ghost_var > 50.0:
                adjustment += 0.06
                signals.append(f"[JPEG-Ghost] High ghost variance ({avg_ghost_var:.1f}) — inconsistent compression history")
            elif avg_ghost_var > 25.0:
                adjustment += 0.03
                signals.append(f"[JPEG-Ghost] Moderate ghost variance ({avg_ghost_var:.1f}) — possible re-compression")
            else:
                adjustment -= 0.02
                signals.append(f"[JPEG-Ghost] Uniform ghost pattern ({avg_ghost_var:.1f}) — consistent compression")
        
        return {"adjustment": adjustment, "signals": signals}

    def _edge_coherence_analysis(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Analyze edge coherence across the image.
        Deepfakes often have inconsistent edge sharpness between
        the manipulated face region and the background.
        """
        signals = []
        adjustment = 0.0
        
        # Multi-scale edge analysis
        edges_fine = cv2.Canny(gray, 100, 200)
        edges_coarse = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        
        h, w = gray.shape
        
        # Compare edge density in center vs periphery
        margin = min(h, w) // 4
        center_edges = edges_fine[margin:h-margin, margin:w-margin]
        
        top_edges = edges_fine[:margin, :]
        bottom_edges = edges_fine[h-margin:, :]
        left_edges = edges_fine[:, :margin]
        right_edges = edges_fine[:, w-margin:]
        
        center_density = float(np.mean(center_edges)) / 255.0
        peripheral_arrays = [top_edges, bottom_edges, left_edges, right_edges]
        periph_density = float(np.mean(np.concatenate([a.flatten() for a in peripheral_arrays]))) / 255.0
        
        # Large discrepancy in edge density between regions suggests manipulation
        if center_density > 0 and periph_density > 0:
            density_ratio = center_density / (periph_density + 1e-6)
            
            if density_ratio > 3.0 or density_ratio < 0.3:
                adjustment += 0.05
                signals.append(f"[Edge-Coherence] Significant edge density discrepancy (ratio: {density_ratio:.2f})")
            elif density_ratio > 2.0 or density_ratio < 0.5:
                adjustment += 0.02
                signals.append(f"[Edge-Coherence] Minor edge density inconsistency (ratio: {density_ratio:.2f})")
            else:
                adjustment -= 0.01
                signals.append("[Edge-Coherence] Consistent edge distribution")
        
        return {"adjustment": adjustment, "signals": signals}

    def _noise_consistency_analysis(self, img_np: np.ndarray) -> Dict[str, Any]:
        """
        Analyze noise pattern consistency across the image.
        Real camera images have consistent sensor noise patterns.
        Synthetic images or manipulated regions have inconsistent noise.
        """
        signals = []
        adjustment = 0.0
        
        # Extract noise by subtracting median-filtered version
        img_float = img_np.astype(np.float32)
        denoised = cv2.medianBlur(img_np, 3).astype(np.float32)
        noise = img_float - denoised
        
        # Analyze noise in blocks
        h, w = noise.shape[:2]
        block_size = 64
        block_stds = []
        
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block_noise = noise[by:by+block_size, bx:bx+block_size]
                block_stds.append(float(np.std(block_noise)))
        
        if len(block_stds) > 4:
            noise_var = float(np.var(block_stds))
            mean_noise = float(np.mean(block_stds))
            cv = noise_var / (mean_noise + 1e-6)  # coefficient of variation
            
            if cv > 10.0:
                adjustment += 0.05
                signals.append(f"[Noise-Pattern] Highly inconsistent noise (CV: {cv:.1f}) — suggests compositing")
            elif cv > 5.0:
                adjustment += 0.02
                signals.append(f"[Noise-Pattern] Moderately inconsistent noise (CV: {cv:.1f})")
            else:
                adjustment -= 0.02
                signals.append(f"[Noise-Pattern] Consistent sensor noise pattern (CV: {cv:.1f})")
        
        return {"adjustment": adjustment, "signals": signals}

    def _color_correlation_analysis(self, img_np: np.ndarray) -> Dict[str, Any]:
        """
        Analyze inter-channel color correlations.
        Real photographs maintain consistent color physics.
        GAN-generated images can have subtle correlation anomalies.
        """
        signals = []
        adjustment = 0.0
        
        b, g, r = cv2.split(img_np)
        
        # Compute channel-pair correlations
        rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
        rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
        gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
        
        # Real images typically have high inter-channel correlation (>0.8)
        avg_corr = (rg_corr + rb_corr + gb_corr) / 3.0
        
        # Very low correlation can indicate synthetic generation
        if avg_corr < 0.5:
            adjustment += 0.04
            signals.append(f"[Color-Corr] Unusual inter-channel correlation ({avg_corr:.2f}) — atypical for natural photography")
        elif avg_corr < 0.7:
            adjustment += 0.01
            signals.append(f"[Color-Corr] Slightly unusual color correlations ({avg_corr:.2f})")
        else:
            adjustment -= 0.01
            signals.append(f"[Color-Corr] Natural color channel correlations ({avg_corr:.2f})")
        
        return {"adjustment": adjustment, "signals": signals}


# ── Singleton ─────────────────────────────────────────────────────────────────
verification_layer = VerificationLayer()
