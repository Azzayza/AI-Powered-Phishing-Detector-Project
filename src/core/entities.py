# src/core/entities.py
from dataclasses import dataclass
from typing import Dict, Any

# Label constants
LABEL_SAFE        = 0
LABEL_PHISHING    = 1
LABEL_PROMOTIONAL = 2

LABEL_NAMES = {
    LABEL_SAFE:        "safe",
    LABEL_PHISHING:    "phishing",
    LABEL_PROMOTIONAL: "promotional",
}

@dataclass
class Email:
    raw_text: str

@dataclass
class PredictionResult:
    label: int          # 0 = safe, 1 = phishing, 2 = promotional
    confidence: float
    explanations: Dict[str, Any]

    @property
    def is_phishing(self) -> bool:
        return self.label == LABEL_PHISHING

    @property
    def is_promotional(self) -> bool:
        return self.label == LABEL_PROMOTIONAL

    @property
    def label_name(self) -> str:
        return LABEL_NAMES.get(self.label, "unknown")
