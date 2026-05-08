# tests/test_pipeline.py
import unittest
from src.core.entities import Email
from src.adapters.preprocessing.text_cleaner import NLTKPreprocessor
from src.adapters.feature_extraction.tfidf_extractor import TfidfExtractor
from src.adapters.classification.sklearn_model import LogisticRegressionClassifier
from src.use_cases.phishing_analyzer import PhishingAnalyzer

class TestPhishingPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = NLTKPreprocessor()
        self.extractor = TfidfExtractor()
        self.classifier = LogisticRegressionClassifier()
        
        texts = [
            "free money urgent click now",
            "hello meeting scheduled for tomorrow"
        ]
        labels = [1, 0]
        
        cleaned_texts = [self.preprocessor.clean(t) for t in texts]
        self.extractor.fit(cleaned_texts)
        features = self.extractor.transform(cleaned_texts)
        
        self.classifier.set_feature_names(self.extractor.get_feature_names())
        self.classifier.train(features, labels)
        
        self.analyzer = PhishingAnalyzer(self.preprocessor, self.extractor, self.classifier)

    def test_phishing_prediction(self):
        email = Email(raw_text="urgent free money claim here")
        result = self.analyzer.analyze(email)
        
        self.assertTrue(result.is_phishing)
        self.assertGreaterEqual(result.confidence, 0.5)

    def test_ham_prediction(self):
        email = Email(raw_text="hello lets schedule the meeting for tomorrow")
        result = self.analyzer.analyze(email)
        
        self.assertFalse(result.is_phishing)
        self.assertGreaterEqual(result.confidence, 0.5)

if __name__ == '__main__':
    unittest.main()
