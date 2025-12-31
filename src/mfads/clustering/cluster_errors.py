from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_misclassifications(
    error_df: pd.DataFrame, 
    embeddings: np.ndarray, 
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Clusters misclassified examples using their embeddings.
    Adds a 'cluster' column to the DataFrame.
    """
    if len(error_df) < n_clusters:
        n_clusters = max(1, len(error_df))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    error_df = error_df.copy()
    error_df["cluster"] = clusters
    return error_df

def get_cluster_keywords(texts: pd.Series, top_k: int = 5) -> List[str]:
    """Extracts top keywords for a cluster using TF-IDF."""
    if len(texts) == 0:
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        indices = scores.argsort()[-top_k:][::-1]
        features = vectorizer.get_feature_names_out()
        return [features[i] for i in indices]
    except Exception:
        return ["error_extracting_keywords"]

def summarize_clusters(error_df: pd.DataFrame) -> List[Dict]:
    """Generates a summary for each error cluster."""
    summaries = []
    
    for cluster_id in error_df["cluster"].unique():
        cluster_data = error_df[error_df["cluster"] == cluster_id]
        keywords = get_cluster_keywords(cluster_data["text"])
        
        # Take a representative example (e.g., the first one for now)
        example = cluster_data["text"].iloc[0]
        
        summaries.append({
            "cluster_id": int(cluster_id),
            "size": len(cluster_data),
            "keywords": keywords,
            "example": example
        })
        
    return summaries
