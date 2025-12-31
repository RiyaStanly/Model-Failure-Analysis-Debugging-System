import joblib
import logging
import os
import pickle
from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class TFIDFModel:
    def __init__(self, max_features: int = 10000, C: float = 1.0):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=C, max_iter=1000))
        ])
        self.is_trained = False

    def train(self, texts: pd.Series, labels: pd.Series):
        logger.info("Training TF-IDF + Logistic Regression model...")
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        logger.info("Training complete.")

    def predict(self, texts: pd.Series) -> pd.Series:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: pd.Series) -> Any:
        return self.pipeline.predict_proba(texts)

    def evaluate(self, texts: pd.Series, labels: pd.Series) -> str:
        preds = self.predict(texts)
        return classification_report(labels, preds)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
