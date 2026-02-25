"""
utils/credibility_db.py
Simulates a behavioral database storing user credibility scores
and historical strike rates.
"""

class CredibilityDB:
    def __init__(self):
        # Database mocking social media accounts
        self.users = {
            "user_new_account": {"account_age_days": 1, "strikes": 0, "follower_count": 5, "verified": False},
            "user_verified_news": {"account_age_days": 1800, "strikes": 0, "follower_count": 500000, "verified": True},
            "user_repeat_offender": {"account_age_days": 120, "strikes": 3, "follower_count": 1200, "verified": False},
            "user_satire_bot": {"account_age_days": 600, "strikes": 0, "follower_count": 45000, "verified": False, "category": "comedy"},
            "user_default": {"account_age_days": 300, "strikes": 0, "follower_count": 150, "verified": False}
        }

    def get_user_credibility(self, user_id: str) -> dict:
        """
        Calculates a real-time credibility multiplier based on account history.
        Multiplier > 1.0 increases threat risk (suspicious users).
        Multiplier < 1.0 decreases threat risk (trusted publishers).
        """
        user = self.users.get(user_id, self.users["user_default"])
        
        # Base multiplier
        multiplier = 1.0
        
        # New accounts have slightly higher risk of being burner bots
        if user["account_age_days"] < 7:
            multiplier += 0.2
            
        # Strikes compound risk heavily
        if user["strikes"] > 0:
            multiplier += (user["strikes"] * 0.3)
            
        # Verified organizational accounts generally have lower misinformation risk
        if user["verified"]:
            multiplier -= 0.3
            
        # Account categorical context (e.g., marked as satire/comedy natively)
        is_satire = user.get("category") == "comedy"
            
        return {
            "credibility_multiplier": max(0.5, min(2.5, multiplier)), # Bound between 0.5x and 2.5x
            "is_satire_account": is_satire,
            "strikes": user["strikes"]
        }

credibility_db = CredibilityDB()
