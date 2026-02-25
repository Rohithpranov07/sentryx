"""
nlp/multilingual_analyzer.py
Scans provided text against cross-lingual harm ontologies.
"""
import re
from nlp.fraud_patterns.multilingual_dict import RISK_PATTERNS, SAFE_PATTERNS

class MultilingualAnalyzer:
    def detect_category(self, caption: str) -> dict:
        """
        Lightweight cross-lingual keyword scan for contextual intent.
        Falls back to 'suspicious_manipulation' if no clear flags map to 
        declared safe patterns.
        """
        if not caption:
             return {"category": "unknown", "detected_language_clue": "none"}
             
        caption_lower = caption.lower()
        
        # 1. Check for declared Safe/Harmless intent first
        for category, lang_dict in SAFE_PATTERNS.items():
            for language, keywords in lang_dict.items():
                for kw in keywords:
                    if kw.lower() in caption_lower:
                        return {"category": category, "detected_language_clue": language}
                        
        # 2. Check for Malicious Framing across all languages
        for category, lang_dict in RISK_PATTERNS.items():
            for language, keywords in lang_dict.items():
                for kw in keywords:
                    if kw.lower() in caption_lower:
                        return {"category": category, "detected_language_clue": language}
                        
        return {"category": "suspicious_manipulation", "detected_language_clue": "unknown"}

multilingual_analyzer = MultilingualAnalyzer()
