"""
models/intent_classifier.py
Contextual Semantic Reasoner. Evaluates captions, user credibility, and behaviors
to determine the *why* behind a media upload.
"""
from typing import Dict, Any
from utils.credibility_db import credibility_db
from nlp.multilingual_analyzer import multilingual_analyzer

class IntentClassifier:
    def _analyze_language(self, caption: str) -> dict:
        """
        Multilingual keyword scan for contextual intent.
        Falls back to 'suspicious_manipulation' if unmatched.
        """
        return multilingual_analyzer.detect_category(caption)


    def analyze_intent(self, threat_score: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Fuses spatial detection with human context.
        """
        # If the pipeline score is very low, it's authentic. No intent scan needed.
        if threat_score < 0.4:
            return {
                "category": "authentic", 
                "intent_multiplier": 1.0,
                "context_risk": 0.0,
                "signals": ["Safe threat threshold bypassed intent scan."]
            }
            
        caption = metadata.get("caption", "")
        uploader_id = metadata.get("uploader_id", "user_default")

        # Extract Intent Category & Language Clue
        lang_data = self._analyze_language(caption)
        semantic_category = lang_data["category"]
        detected_language = lang_data["detected_language_clue"]
        
        # Pull Credibility History
        credibility = credibility_db.get_user_credibility(uploader_id)
        
        # Harmless Context overriding the spatial threshold
        if semantic_category in ["satire", "art", "education"] or credibility["is_satire_account"]:
             return {
                 "category": semantic_category,
                 "intent_multiplier": 0.2, # Crushes risk (Allows publishing with a label)
                 "context_risk": min(1.0, threat_score * 0.2),
                 "signals": [f"Declared context: {semantic_category.title()}", "Threat score lowered manually by Policy Intent Engine."]
             }
             
        # Malicious Context acting as a multiplier
        base_multiplier = credibility["credibility_multiplier"]
        
        # Specific harms carry heavier weight
        if semantic_category in ["fraud", "ncii", "political_disinfo", "medical_misinfo"]:
             base_multiplier += 0.5 # Compound specifically for critical harms
             
        final_risk = min(1.0, threat_score * base_multiplier)
        
        signals = [
            f"Account Risk Multiplier: {credibility['credibility_multiplier']:.1f}x",
            f"Linguistic Intent Prediction: {semantic_category.upper()}",
            f"Matched Context Language: {detected_language.title()}"
        ]
        
        if final_risk > 0.9:
            signals.append("Contextual Intent confirms extreme threat vector.")

        return {
            "category": semantic_category,
            "intent_multiplier": round(base_multiplier, 2),
            "context_risk": round(final_risk, 4),
            "signals": signals
        }

intent_classifier = IntentClassifier()
