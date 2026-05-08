# src/adapters/feature_extraction/tfidf_extractor.py
from typing import List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from src.core.interfaces import IFeatureExtractor

class TfidfExtractor(IFeatureExtractor):
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def transform(self, texts: List[str]) -> Any:
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before calling transform.")
        return self.vectorizer.transform(texts)
        
    def get_feature_names(self) -> List[str]:
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
