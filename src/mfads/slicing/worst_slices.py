from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def compute_slice_metrics(slices: Dict[str, pd.DataFrame], y_true_col: str = "label", y_pred_col: str = "prediction") -> pd.DataFrame:
    """Computes metrics (F1, Accuracy) for each slice."""
    results = []
    
    for slice_name, slice_df in slices.items():
        if slice_df.empty:
            continue
            
        y_true = slice_df[y_true_col]
        y_pred = slice_df[y_pred_col]
        
        f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        support = len(slice_df)
        
        results.append({
            "slice": slice_name,
            "f1": f1,
            "accuracy": acc,
            "support": support
        })
        
    return pd.DataFrame(results)

def find_worst_slices(metrics_df: pd.DataFrame, min_support: int = 20, top_k: int = 5) -> pd.DataFrame:
    """Ranks slices by lowest F1 score."""
    filtered_df = metrics_df[metrics_df["support"] >= min_support]
    return filtered_df.sort_values(by="f1", ascending=True).head(top_k)

def detect_bias_gaps(metrics_df: pd.DataFrame, overall_f1: float, threshold: float = 0.1) -> pd.DataFrame:
    """Flags slices with significant performance gaps vs overall."""
    metrics_df = metrics_df.copy()
    metrics_df["gap"] = overall_f1 - metrics_df["f1"]
    return metrics_df[metrics_df["gap"] >= threshold]
