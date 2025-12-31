import re
from typing import List, Set

import pandas as pd
import numpy as np
from collections import Counter

def get_text_length(text: str) -> int:
    """Returns number of tokens in text."""
    return len(text.split())

def get_rare_token_ratio(text: str, rare_tokens: Set[str]) -> float:
    """Calculates ratio of rare tokens in text."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    rare_count = sum(1 for t in tokens if t in rare_tokens)
    return rare_count / len(tokens)

def get_corpus_rare_tokens(texts: pd.Series, threshold_quantile: float = 0.01) -> Set[str]:
    """Identifies rare tokens in a corpus based on frequency quantile."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    
    counts = Counter(all_tokens)
    freq_values = list(counts.values())
    rare_threshold = np.quantile(freq_values, threshold_quantile)
    
    rare_tokens = {token for token, count in counts.items() if count <= rare_threshold}
    return rare_tokens

def extract_slicing_features(df: pd.DataFrame, rare_tokens: Set[str] = None) -> pd.DataFrame:
    """Extracts features used for slicing."""
    df = df.copy()
    df["text_length"] = df["text"].apply(get_text_length)
    if rare_tokens:
        df["rare_token_ratio"] = df["text"].apply(lambda x: get_rare_token_ratio(x, rare_tokens))
    return df
