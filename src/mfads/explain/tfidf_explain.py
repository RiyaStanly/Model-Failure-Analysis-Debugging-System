from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def explain_tfidf_prediction(text: str, pipeline: Pipeline, top_k: int = 5) -> Dict[str, float]:
    """
    Explains a single prediction for the TF-IDF model.
    Returns top positive contributors.
    """
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    # Vectorize text
    vec = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    
    # Get coefficients for the predicted class
    prediction = clf.predict(vec)[0]
    class_idx = list(clf.classes_).index(prediction)
    coeffs = clf.coef_[class_idx]
    
    # Find active features in this text
    active_indices = vec.indices
    active_coeffs = coeffs[active_indices]
    active_features = feature_names[active_indices]
    
    # Sort by contribution
    sorted_idx = active_coeffs.argsort()[::-1]
    
    explanation = {}
    for i in sorted_idx[:top_k]:
        explanation[active_features[i]] = float(active_coeffs[i])
        
    return explanation

def get_slice_explanations(df: pd.DataFrame, pipeline: Pipeline) -> List[str]:
    """Returns top keywords contributing to errors in a slice."""
    # This is a simplified version of global explanation for a slice
    all_explanations = []
    for text in df["text"]:
        expl = explain_tfidf_prediction(text, pipeline)
        all_explanations.extend(expl.keys())
    
    from collections import Counter
    return [word for word, count in Counter(all_explanations).most_common(10)]
