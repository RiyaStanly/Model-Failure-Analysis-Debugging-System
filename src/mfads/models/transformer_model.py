import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

class TransformerModel:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "models/distilbert"):
        logger.info(f"Fine-tuning {self.model_name}...")
        
        train_ds = Dataset.from_pandas(train_df)
        test_ds = Dataset.from_pandas(test_df)
        
        tokenized_train = train_ds.map(self._tokenize_function, batched=True)
        tokenized_test = test_ds.map(self._tokenize_function, batched=True)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1, # Low epochs for fast demonstration
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Transformer training complete.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        
        # Simple inference using pipeline for convenience in analysis
        classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1, top_k=None)
        
        results = classifier(texts)
        # Convert list of dicts to probability matrix
        proba = []
        for res in results:
            # Sort by label index to ensure correct order
            sorted_res = sorted(res, key=lambda x: int(x['label'].split('_')[-1]))
            proba.append([r['score'] for r in sorted_res])
            
        return np.array(proba)

    def load(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Transformer model loaded from {model_path}")
