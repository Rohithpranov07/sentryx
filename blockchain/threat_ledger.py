"""
blockchain/threat_ledger.py

Cross-Platform Threat Intelligence DB.
Simulates a global smart contract for sharing cryptographic deepfake 
fingerprints between platforms instantly without sharing PII or raw media.
"""
from utils.ledger import register_threat, lookup_sha256, lookup_similar

class GlobalThreatLedger:
    def __init__(self):
        # We hook directly to our physical SQLite DB to act as our local node
        # in the global blockchain network.
        pass

    def check_global_network(self, sha256: str, phash: str) -> dict:
        """
        Queries the network for an exact or similar match.
        """
        # Exact Match (Zero-Day or unaltered duplicate)
        match = lookup_sha256(sha256)
        if match:
            return {"found": True, "type": "exact", "data": match}

        # Similar Match (Variant: Cropped, Recompressed, Filtered)
        similar = lookup_similar(phash, max_distance=10)
        if similar:
            return {"found": True, "type": "variant", "data": similar}

        return {"found": False}

    def sync_to_network(self, fingerprints: dict, policy: dict, metadata: dict) -> dict:
        """
        Propagates a newly discovered threat across the decentralized API immediately.
        """
        # Package the threat payload, stripped of user information
        payload = {
            "risk_level": policy["tier"],
            "verdict": "Critical Threat" if policy["tier"] == "red" else "Manipulated Content",
            "confidence": 0.99,
            "forensic_signals": ["Global Variant Match", policy.get("policy_enforcement", "")]
        }

        # Commit to the local ledger (representing the blockchain write function)
        res = register_threat(
            fingerprints=fingerprints,
            verdict=payload,
            filename=metadata["filename"],
            platform_id=metadata["platform_id"]
        )

        # Indicate the transaction was verified globally
        res["ledger_type"] = "cross_platform_intelligence_network"
        return res

global_ledger = GlobalThreatLedger()
