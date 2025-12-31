import logging
from typing import Dict, Tuple

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_data(dataset_name: str = "ag_news") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset using Hugging Face datasets and return as pandas DataFrames.
    
    Args:
        dataset_name: Name of the dataset to load. Default is 'ag_news'.
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Map labels to text if available (for AG News)
    if dataset_name == "ag_news":
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        train_df["label_text"] = train_df["label"].map(label_map)
        test_df["label_text"] = test_df["label"].map(label_map)
        
    logger.info(f"Loaded {len(train_df)} train samples and {len(test_df)} test samples.")
    return train_df, test_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train, test = load_data()
    print(train.head())
