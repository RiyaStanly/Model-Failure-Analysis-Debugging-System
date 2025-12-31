import pytest
import pandas as pd
import numpy as np
from mfads.slicing.features import extract_slicing_features
from mfads.slicing.slicer import Slicer

def test_extract_slicing_features():
    df = pd.DataFrame({"text": ["Hello world", "Short", "This is a longer sentence for testing"]})
    df_features = extract_slicing_features(df)
    
    assert "text_length" in df_features.columns
    assert df_features["text_length"].iloc[0] == 2
    assert df_features["text_length"].iloc[1] == 1
    assert df_features["text_length"].iloc[2] == 7

def test_slicing_logic():
    df = pd.DataFrame({
        "text": ["A"]*10 + ["B"]*10,
        "label": [0]*10 + [1]*10,
        "text_length": [10]*5 + [60]*5 + [10]*5 + [60]*5,
        "rare_token_ratio": [0.0]*20
    })
    
    config = {
        "slices": {
            "text_length": {
                "enabled": True,
                "bins": [0, 50, 100, 200, 500, 10000],
                "labels": ["very_short", "short", "medium", "long", "very_long"]
            },
            "label": {"enabled": True},
            "rare_token_ratio": {"enabled": False}
        }
    }
    
    slicer = Slicer(config)
    slices = slicer.slice_data(df)
    
    assert "length_very_short" in slices
    assert "length_short" in slices
    assert len(slices["length_very_short"]) == 10
    assert len(slices["length_short"]) == 10
