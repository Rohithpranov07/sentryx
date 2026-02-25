"""
models/dynamic_threshold.py

SENTRY-X Dynamic Thresholding Engine (TASK 4)
==============================================

Replaces static 0.5 cutoffs with context-adaptive thresholds.

Rules:
- High physiological inconsistency → lower the detection threshold
- Strong natural camera signals → raise the threshold
- Evasion indicators present → lower the threshold
- Multiple model agreement → use standard thresholds
- Single weak signal → raise the threshold

The threshold is computed per-image based on supporting evidence,
not as a global constant.
"""

from typing import Dict, Any, Tuple


class DynamicThresholdEngine:
    """
    Computes adaptive detection thresholds based on contextual signals.
    """

    def __init__(self):
        # Base thresholds (starting point before adjustment)
        self.base_thresholds = {
            "safe_max": 0.30,       # Below this = definitely safe
            "label_max": 0.50,      # Below this = label only
            "restrict_max": 0.75,   # Below this = restrict  
            "block_min": 0.90,      # Above this = may block (requires multi-model)
        }

    def compute_thresholds(
        self,
        physio_data: Dict[str, Any],
        evasion_data: Dict[str, Any],
        ensemble_agreement: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute context-specific thresholds based on supporting signals.
        
        Returns adjusted threshold dictionary.
        """
        adjustment = 0.0
        reasons = []
        
        # ── Physiological Signals ──
        physio_conf = physio_data.get("physiological_confidence", 0.0)
        micro_exp = physio_data.get("micro_expression_suspicion", 0.0)
        eye_susp = physio_data.get("eye_microsaccade_suspicion", 0.0)
        
        if physio_conf > 0.5:
            # High physiological anomaly → LOWER threshold (more sensitive)
            adjustment -= 0.08
            reasons.append(f"High physio anomaly ({physio_conf:.2f}) → threshold lowered")
        elif physio_conf > 0.3:
            adjustment -= 0.04
            reasons.append(f"Moderate physio anomaly ({physio_conf:.2f}) → threshold slightly lowered")
        elif physio_conf < 0.05:
            # Very clean physio → RAISE threshold (less sensitive)
            adjustment += 0.03
            reasons.append("Clean physiological signals → threshold slightly raised")
        
        # ── Evasion Signals ──
        evasion_risk = evasion_data.get("evasion_risk", 0.0)
        
        if evasion_data.get("screenshot_detected", False):
            adjustment -= 0.06
            reasons.append("Screenshot bypass detected → threshold lowered")
        
        if evasion_data.get("adversarial_noise", False):
            adjustment -= 0.10
            reasons.append("Adversarial noise detected → threshold significantly lowered")
        
        if evasion_data.get("metadata_stripped", False) and evasion_risk > 0.5:
            adjustment -= 0.03
            reasons.append("Metadata stripped with high evasion risk → threshold lowered")
        
        # ── Ensemble Agreement ──
        models_active = ensemble_agreement.get("models_active", 0)
        models_flagging = ensemble_agreement.get("models_flagging_medium", 0)
        has_majority = ensemble_agreement.get("has_majority", False)
        
        if has_majority:
            # Strong agreement → use standard thresholds
            reasons.append("Multi-model consensus → standard thresholds")
        elif models_flagging == 1:
            # Single model only → RAISE threshold (need more evidence)
            adjustment += 0.05
            reasons.append("Single-model flag only → threshold raised (need consensus)")
        elif models_flagging == 0:
            # No flags → RAISE threshold significantly
            adjustment += 0.08
            reasons.append("No model flags → high threshold (conservative)")
        
        # ── Apply Adjustment (clamped) ──
        adjustment = max(-0.15, min(0.12, adjustment))
        
        thresholds = {
            "safe_max": max(0.15, self.base_thresholds["safe_max"] + adjustment),
            "label_max": max(0.30, self.base_thresholds["label_max"] + adjustment),
            "restrict_max": max(0.50, self.base_thresholds["restrict_max"] + adjustment),
            "block_min": max(0.75, self.base_thresholds["block_min"] + adjustment),
            "adjustment_applied": round(adjustment, 3),
            "reasons": reasons,
        }
        
        return thresholds


class FinalDecisionEngine:
    """
    TASK 5: Final risk classification with multi-model agreement requirements.
    
    Decision Rules:
    ─────────────────────────────────────────────────────────────────
    HIGH RISK (red/block):
      - confidence > 0.9 AND multi-model agreement (2+ models)
      
    HIGH RISK (red/restrict):
      - At least 2 models agree at confidence > 0.7
      
    MEDIUM RISK (orange/restrict):
      - 1 model high + others medium
      
    LOW RISK (yellow/label):
      - Only 1 weak signal OR ensemble score in label range
      
    SAFE (green/publish):
      - Below all thresholds
    ─────────────────────────────────────────────────────────────────
    """

    def classify(
        self,
        ensemble_score: float,
        thresholds: Dict[str, float],
        agreement: Dict[str, Any],
        verification_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Final risk classification using dynamic thresholds + agreement.
        """
        # Use verified score if available, else ensemble score
        score = ensemble_score
        if verification_data and "verified_score" in verification_data:
            score = verification_data["verified_score"]
        
        safe_max = thresholds["safe_max"]
        label_max = thresholds["label_max"]
        restrict_max = thresholds["restrict_max"]
        block_min = thresholds["block_min"]
        
        has_majority = agreement.get("has_majority", False)
        has_strong_majority = agreement.get("has_strong_majority", False)
        models_high = agreement.get("models_flagging_high", 0)
        models_medium = agreement.get("models_flagging_medium", 0)
        
        # ── DECISION LOGIC ──
        
        # HARD BLOCK: confidence > 0.9 AND multi-model strong agreement
        if score > block_min and has_strong_majority and models_high >= 2:
            return {
                "risk_level": "red",
                "verdict": "Manipulated Media Detected",
                "action": "block",
                "color_code": "#ef4444",
                "description": (
                    f"High-confidence deepfake (score: {score:.2f}). "
                    f"{models_high} independent models confirm manipulation. "
                    f"Content blocked and fingerprint registered."
                ),
                "decision_path": "block_multimodel",
                "effective_score": round(score, 4),
            }
        
        # HIGH RISK: 2+ models agree at high confidence
        if models_high >= 2 and score > restrict_max:
            return {
                "risk_level": "red",
                "verdict": "High-Confidence Synthetic Media",
                "action": "restrict",
                "color_code": "#ef4444",
                "description": (
                    f"Multiple detectors identify synthetic generation (score: {score:.2f}). "
                    f"Content restricted from amplification."
                ),
                "decision_path": "high_risk_multimodel",
                "effective_score": round(score, 4),
            }
        
        # MEDIUM RISK: Need at least 1 model at high confidence + others supporting
        # This prevents real images with borderline ensemble scores from being restricted
        if models_high >= 1 and models_medium >= 2 and score > label_max:
            return {
                "risk_level": "orange",
                "verdict": "Suspicious — Restricted",
                "action": "restrict",
                "color_code": "#f97316",
                "description": (
                    f"Manipulation indicators detected (score: {score:.2f}). "
                    f"{models_high} detector(s) at high confidence. "
                    f"Content restricted pending human review."
                ),
                "decision_path": "medium_risk",
                "effective_score": round(score, 4),
            }
        
        # LABEL: Score exceeds safe threshold but no strong model agreement
        # This is the balanced zone — flag but don't restrict
        if score > safe_max:
            return {
                "risk_level": "yellow",
                "verdict": "AI-Generated Content Possible",
                "action": "label",
                "color_code": "#eab308",
                "description": (
                    f"Possible AI-generated content detected (score: {score:.2f}). "
                    f"Published with disclosure label."
                ),
                "decision_path": "label_weak_signal",
                "effective_score": round(score, 4),
            }
        
        # SAFE: below all thresholds
        return {
            "risk_level": "green",
            "verdict": "Authentic & Safe",
            "action": "publish",
            "color_code": "#22c55e",
            "description": (
                f"No significant manipulation detected (score: {score:.2f}). "
                f"Content cleared for publishing."
            ),
            "decision_path": "safe",
            "effective_score": round(score, 4),
        }


# ── Singleton instances ──────────────────────────────────────────────────────
dynamic_threshold_engine = DynamicThresholdEngine()
final_decision_engine = FinalDecisionEngine()
