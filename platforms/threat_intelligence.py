"""
platform/threat_intelligence.py

Connects Platform-Level moderation actions to the 
Global Threat Intelligence architecture (Phase 1 + Phase 5).
Ensures that if TikTok spots a fake, Instagram's API responds instantly.
"""
from typing import Dict, Any, Tuple
from utils.robust_fingerprint import generate_robust_fingerprint
from blockchain.threat_ledger import global_ledger
from PIL import Image

class ThreatIntelligenceController:
    """
    Handles Cross-Platform Threat Sharing Hooks before Deep Analysis.
    """
    
    def execute_global_triage(self, file_bytes: bytes, image: Image.Image) -> Tuple[bool, Dict[str, Any], Dict[str, str]]:
        """
        Phase 1: Checks the global node network for known variants.
        Replaces the old, localized exact-match system.
        """
        # 1. Compute robust, multi-modal identifiers designed entirely 
        #    to track bad actors scaling images dynamically.
        fingerprints = generate_robust_fingerprint(image)
        
        # 2. Query Decentralized Network (Represented locally by SQLite in PoC)
        network_hit = global_ledger.check_global_network(
            sha256=fingerprints["sha256"], 
            phash=fingerprints["phash"]
        )
        
        if network_hit["found"]:
            threat_data = network_hit["data"]
            # Enforce global blocking only if the threat is high/critical
            if threat_data["risk_level"] in ("red", "orange"):
                threat_data["verdict"] = f"{threat_data['verdict']} ({network_hit['type'].title()} Variant Match - Cross-Platform Node)"
                return True, threat_data, fingerprints
                
        # Clean! Proceed to Pipeline Phase 2 (Biological / Evasion Checks)
        return False, {}, fingerprints
        
    def execute_global_ban(self, policy: Dict[str, Any], fingerprints: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 5: Pushes newly discovered local threats up to the Global ledger.
        """
        if policy["tier"] in ("red", "orange"):
             return global_ledger.sync_to_network(
                 fingerprints=fingerprints, 
                 policy=policy, 
                 metadata=metadata
             )
        return None

threat_intel_controller = ThreatIntelligenceController()
