import argparse
import logging
import os
import yaml
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from mfads.datasets.load_data import load_data
from mfads.models.tfidf_model import TFIDFModel
from mfads.models.transformer_model import TransformerModel
from mfads.slicing.features import extract_slicing_features, get_corpus_rare_tokens
from mfads.slicing.slicer import Slicer
from mfads.slicing.worst_slices import compute_slice_metrics, find_worst_slices, detect_bias_gaps
from mfads.clustering.embed import Embedder
from mfads.clustering.cluster_errors import cluster_misclassifications, summarize_clusters
from mfads.report.build_report import generate_markdown_report, generate_recommendations, save_artifacts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(args, settings):
    train_df, test_df = load_data(settings["dataset"])
    
    # Use sample size from settings if provided, else default to 2000/500
    train_sample = settings.get("train_sample_size", 2000)
    test_sample = settings.get("test_sample_size", 500)
    
    train_df = train_df.sample(min(train_sample, len(train_df)), random_state=42)
    test_df = test_df.sample(min(test_sample, len(test_df)), random_state=42)

    if args.model == "tfidf":
        model = TFIDFModel(
            max_features=settings["models"]["tfidf"]["max_features"],
            C=settings["models"]["tfidf"]["C"]
        )
        model.train(train_df["text"], train_df["label"])
        model.save("models/tfidf_model.pkl")
    elif args.model == "distilbert":
        model = TransformerModel(
            model_name=settings["models"]["distilbert"]["pretrained_model"]
        )
        model.train(train_df, test_df, output_dir="models/distilbert")

def analyze(args, settings, slices_cfg):
    logger.info("Starting failure analysis...")
    
    # Check if model exists
    if args.model == "tfidf" and not os.path.exists("models/tfidf_model.pkl"):
        logger.error("TF-IDF model not found. Run 'mfads train --model tfidf' first.")
        return
    elif args.model == "distilbert" and not os.path.exists("models/distilbert"):
        logger.error("DistilBERT model not found. Run 'mfads train --model distilbert' first.")
        return

    _, test_df = load_data(settings["dataset"])
    test_sample = settings.get("test_sample_size", 500)
    test_df = test_df.sample(min(test_sample, len(test_df)), random_state=42)
    
    # Load model
    if args.model == "tfidf":
        model = TFIDFModel()
        model.load("models/tfidf_model.pkl")
    else:
        model = TransformerModel()
        model.load("models/distilbert")
    
    # 1. Predictions
    logger.info("Generating predictions...")
    test_df["prediction"] = model.predict(test_df["text"])
    
    # 2. Extract Features & Slicing
    logger.info("Slicing data...")
    rare_tokens = get_corpus_rare_tokens(test_df["text"])
    test_df = extract_slicing_features(test_df, rare_tokens)
    
    slicer = Slicer(slices_cfg)
    slices = slicer.slice_data(test_df)
    
    # 3. Metrics & Worst Slices
    overall_f1 = f1_score(test_df["label"], test_df["prediction"], average="weighted")
    overall_acc = accuracy_score(test_df["label"], test_df["prediction"])
    
    metrics_df = compute_slice_metrics(slices)
    worst_slices = find_worst_slices(metrics_df, min_support=slices_cfg["thresholds"]["min_support"])
    bias_gaps = detect_bias_gaps(metrics_df, overall_f1, threshold=slices_cfg["thresholds"]["f1_gap_warning"])
    
    # 4. Error Clustering
    logger.info("Clustering errors...")
    error_df = test_df[test_df["label"] != test_df["prediction"]].copy()
    if not error_df.empty:
        embedder = Embedder(model_name=slices_cfg["clustering"]["embedding_model"])
        embeddings = embedder.embed(error_df["text"].tolist())
        error_df = cluster_misclassifications(error_df, embeddings, n_clusters=slices_cfg["clustering"].get("min_cluster_size", 5))
        cluster_sum = summarize_clusters(error_df)
    else:
        cluster_sum = []
    
    # 5. Report
    recommendations = generate_recommendations(worst_slices, bias_gaps)
    
    report_dir = f"reports/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    generate_markdown_report(
        {"f1": overall_f1, "accuracy": overall_acc},
        worst_slices,
        bias_gaps,
        cluster_sum,
        recommendations,
        os.path.join(report_dir, "report.md")
    )
    
    save_artifacts({
        "overall": {"f1": overall_f1, "accuracy": overall_acc},
        "metrics": metrics_df,
        "worst_slices": worst_slices,
        "bias_gaps": bias_gaps,
        "clusters": cluster_sum,
        "recommendations": recommendations,
        "errors": error_df.head(100)
    }, report_dir)

def main():
    parser = argparse.ArgumentParser(description="MFADS CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["tfidf", "distilbert"], default="tfidf")
    
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--model", choices=["tfidf", "distilbert"], default="tfidf")
    
    args = parser.parse_args()
    
    if not os.path.exists("config/settings.yaml") or not os.path.exists("config/slices.yaml"):
        logger.error("Configuration files not found in config/ directory.")
        return

    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/slices.yaml") as f:
        slices_cfg = yaml.safe_load(f)
        
    if args.command == "train":
        train(args, settings)
    elif args.command == "analyze":
        analyze(args, settings, slices_cfg)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
