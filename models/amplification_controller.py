"""
models/amplification_controller.py
Maps complex risk profiles into unified Policy Tiers for platforms.
"""
from typing import Dict, Any

class AmplificationController:
    def evaluate_risk(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates final contextual risk from Phase 3 into strict policy tiers.
        """
        final_risk = intent_data.get("context_risk", 0.0)
        
        # Calibrated FP-Resistant Thresholds
        if final_risk >= 0.85:
            tier = "red"
            action = "block"
        elif final_risk >= 0.70:
            tier = "red"       # High confidence fake
            action = "restrict" # But no hard block (FP mitigation)
        elif final_risk >= 0.50:
            tier = "orange"
            action = "restrict"
        elif final_risk >= 0.30:
            tier = "yellow"
            action = "label"
        else:
            tier = "green"
            action = "publish"
            
        return {
            "tier": tier,
            "action": action,
            "final_risk_score": final_risk
        }

amplification_controller = AmplificationController()
