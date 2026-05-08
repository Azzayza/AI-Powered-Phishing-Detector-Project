# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict
from src.core.entities import Email, PredictionResult

class IPreprocessor(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        pass

class IFeatureExtractor(ABC):
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> Any:
        pass
        
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass

class IClassifier(ABC):
    @abstractmethod
    def train(self, features: Any, labels: List[int]) -> None:
        pass

    @abstractmethod
    def predict(self, features: Any) -> Tuple[bool, float, Dict[str, float]]:
        pass

    @abstractmethod
    def set_feature_names(self, feature_names: List[str]) -> None:
        pass

class IModelRepository(ABC):
    @abstractmethod
    def save_extractor(self, extractor: IFeatureExtractor, path: str) -> None:
        pass
        
    @abstractmethod
    def load_extractor(self, path: str) -> IFeatureExtractor:
        pass

    @abstractmethod
    def save_classifier(self, classifier: IClassifier, path: str) -> None:
        pass

    @abstractmethod
    def load_classifier(self, path: str) -> IClassifier:
        pass
