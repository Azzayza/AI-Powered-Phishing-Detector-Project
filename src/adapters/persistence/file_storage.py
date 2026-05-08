# src/adapters/persistence/file_storage.py
import joblib
import os
from src.core.interfaces import IModelRepository, IFeatureExtractor, IClassifier

class JoblibModelRepository(IModelRepository):
    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save_extractor(self, extractor: IFeatureExtractor, path: str) -> None:
        self._ensure_dir(path)
        joblib.dump(extractor, path)

    def load_extractor(self, path: str) -> IFeatureExtractor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature extractor file not found at: {path}")
        return joblib.load(path)

    def save_classifier(self, classifier: IClassifier, path: str) -> None:
        self._ensure_dir(path)
        joblib.dump(classifier, path)

    def load_classifier(self, path: str) -> IClassifier:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier file not found at: {path}")
        return joblib.load(path)
