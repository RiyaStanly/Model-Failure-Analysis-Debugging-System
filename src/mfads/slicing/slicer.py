from typing import Dict, List, Any

import pandas as pd
import numpy as np

class Slicer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["slices"]

    def slice_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creates slices of data based on the configuration.
        Returns a dictionary mapping slice names to DataFrames.
        """
        slices = {}
        
        # 1. Label slices
        if self.config.get("label", {}).get("enabled", True):
            for label in df["label"].unique():
                label_name = df[df["label"] == label]["label_text"].iloc[0] if "label_text" in df.columns else str(label)
                slices[f"label_{label_name}"] = df[df["label"] == label]

        # 2. Text length slices
        if self.config.get("text_length", {}).get("enabled", True):
            bins = self.config["text_length"]["bins"]
            labels = self.config["text_length"]["labels"]
            
            # Use pd.cut to create buckets
            df["length_bucket"] = pd.cut(df["text_length"], bins=bins, labels=labels, include_lowest=True)
            
            for label in labels:
                slice_df = df[df["length_bucket"] == label]
                if not slice_df.empty:
                    slices[f"length_{label}"] = slice_df

        # 3. Rare token ratio slices
        if self.config.get("rare_token_ratio", {}).get("enabled", True):
            thresholds = self.config["rare_token_ratio"]["thresholds"]
            
            # Simple binary split for now: low vs high rarity
            # In a real tool, this could be more granular
            mid_threshold = thresholds[1] if len(thresholds) > 1 else 0.1
            slices["rarity_low"] = df[df["rare_token_ratio"] <= mid_threshold]
            slices["rarity_high"] = df[df["rare_token_ratio"] > mid_threshold]

        return slices
