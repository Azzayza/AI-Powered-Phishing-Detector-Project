# src/use_cases/phishing_analyzer.py
import re
from src.core.entities import Email, PredictionResult, LABEL_PHISHING, LABEL_PROMOTIONAL, LABEL_SAFE

# High-confidence phishing keyword patterns (rule-based safety net).
PHISHING_PATTERNS = [
    r'\bclick\s+here\b',
    r'\bverify\s+(your\s+)?(account|identity|information|details|email)\b',
    r'\b(urgent|immediate(ly)?|action\s+required|final\s+warning)\b',
    r'\b(free\s+iphone|free\s+ipad|free\s+gift|free\s+prize|free\s+vacation)\b',
    r'\b(won|winner|winning|lottery|jackpot|prize)\b',
    r'\b(bank\s+(account|details)|credit\s+card\s+number)\b',
    r'\b(password\s+(expire|reset|verify)|reset\s+your\s+password)\b',
    r'\b(account\s+(suspend|locked|comprom|restrict|terminat))\b',
    r'\b(update\s+(your\s+)?(billing|payment|account|details))\b',
    r'\b(claim\s+(your\s+)?(prize|reward|gift|winnings))\b',
    r'\b(earn\s+\$?\d+|make\s+money|work\s+from\s+home)\b',
    r'\b(pre.?approved|no\s+credit\s+check|guaranteed\s+(return|profit))\b',
    r'\b(suspended|permanently\s+deleted|lose\s+access)\b',
    r'\b(shipping\s+fee|customs?\s+(fee|duty)|pay\s+to\s+release)\b',
]

RULE_OVERRIDE_THRESHOLD = 2
CONFIDENCE_BOOST = 0.10


def _check_phishing_rules(text: str) -> tuple:
    """Returns (match_count, list of matched keywords)."""
    text_lower = text.lower()
    matches = []
    for pattern in PHISHING_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            matches.append(m.group(0).strip())
    return len(matches), matches


class PhishingAnalyzer:
    def __init__(self, preprocessor, extractor, classifier):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.classifier = classifier

    def analyze(self, email: Email) -> PredictionResult:
        cleaned_text = self.preprocessor.clean(email.raw_text)
        features = self.extractor.transform([cleaned_text])
        ml_label, ml_confidence, feature_importances = self.classifier.predict(features)

        # --- Rule-based phishing safety net ---
        rule_count, rule_matches = _check_phishing_rules(email.raw_text)
        rule_triggered = rule_count >= RULE_OVERRIDE_THRESHOLD

        # Decision logic:
        # 1. Rules strongly flag phishing AND ML missed it → override to phishing
        # 2. Both ML and rules agree it's phishing → boost confidence
        # 3. Otherwise, trust ML (handles promotional vs safe natively)
        if rule_triggered and ml_label != LABEL_PHISHING:
            label = LABEL_PHISHING
            confidence = min(0.5 + (rule_count * 0.08), 0.95)
            detection_method = "rule_override"
        elif rule_triggered and ml_label == LABEL_PHISHING:
            label = LABEL_PHISHING
            confidence = min(ml_confidence + CONFIDENCE_BOOST, 0.99)
            detection_method = "ml_and_rules"
        else:
            label = ml_label
            confidence = ml_confidence
            detection_method = "ml_only"

        return PredictionResult(
            label=label,
            confidence=confidence,
            explanations={
                "top_features": feature_importances,
                "cleaned_text": cleaned_text,
                "rule_matches": rule_matches,
                "detection_method": detection_method,
            }
        )
