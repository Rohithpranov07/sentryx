"""
platforms/reach_limiter.py
Translates Policy Tiers into physical API limits and visibility multipliers.
"""
from typing import Dict, Any

class ReachLimiter:
    def apply_limits(self, policy_tier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the policy tier and outputs precise limits for the platform's feed algorithm.
        """
        tier = policy_tier["tier"]
        action = policy_tier["action"]
        
        if tier == "red":
            return {
                "tier": tier,
                "action": action,
                "visibility_multiplier": 0.0,
                "reach_limits": {"feed": False, "search": False, "recommendations": False},
                "policy_enforcement": "Hard block. Payload removed from indexing queue."
            }
        elif tier == "orange":
            return {
                "tier": tier,
                "action": action,
                "visibility_multiplier": 0.05,
                "reach_limits": {"feed": True, "search": False, "recommendations": False},
                "policy_enforcement": "Shadow suppression applied. K-factor velocity bounded."
            }
        elif tier == "yellow":
            return {
                "tier": tier,
                "action": action,
                "visibility_multiplier": 0.70,
                "reach_limits": {"feed": True, "search": True, "recommendations": False},
                "policy_enforcement": "AI-generated label applied. Soft reach reduction (Out of recs)."
            }
        else:
            return {
                "tier": tier,
                "action": action,
                "visibility_multiplier": 1.0,
                "reach_limits": {"feed": True, "search": True, "recommendations": True},
                "policy_enforcement": "Authentic media. Standard recommendation tier unlocked."
            }

reach_limiter = ReachLimiter()
