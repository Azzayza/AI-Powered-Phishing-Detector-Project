# src/adapters/preprocessing/text_cleaner.py
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.core.interfaces import IPreprocessor

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class NLTKPreprocessor(IPreprocessor):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)

    def clean(self, text: str) -> str:
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned_tokens = [
            word for word in tokens 
            if word not in self.punctuation and word not in self.stop_words and word.isalpha()
        ]
        return " ".join(cleaned_tokens)
