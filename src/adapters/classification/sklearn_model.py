# src/adapters/classification/sklearn_model.py
from typing import Any, List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.core.interfaces import IClassifier

class LogisticRegressionClassifier(IClassifier):
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=1.5)
        self.is_trained = False
        self.feature_names = None

    def train(self, features: Any, labels: List[int]) -> None:
        self.model.fit(features, labels)
        self.is_trained = True

    def set_feature_names(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def predict(self, features: Any) -> Tuple[int, float, Dict[str, float]]:
        if not self.is_trained:
            raise ValueError("Classifier must be trained before predicting.")
            
        prediction = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        
        explanations = {}
        if self.feature_names is not None and hasattr(self.model, "coef_"):
            feature_array = features.toarray()[0]
            # For multiclass, use the coef row of the predicted class
            coef_row = self.model.coef_[prediction] if self.model.coef_.shape[0] > 1 else self.model.coef_[0]
            contributions = feature_array * coef_row
            
            top_indices = np.argsort(contributions)
            important_features = {}
            
            # Extract top 5 most influential words (positive contributions)
            for idx in top_indices[-5:]:
                if contributions[idx] > 0:
                    important_features[self.feature_names[idx]] = float(contributions[idx])
            # Extract top 3 counter-indicators (negative contributions)
            for idx in top_indices[:3]:
                if contributions[idx] < 0:
                    important_features[self.feature_names[idx]] = float(contributions[idx])
                    
            explanations["influencing_words"] = important_features

        return prediction, confidence, explanations
