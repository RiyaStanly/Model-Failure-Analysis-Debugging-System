# MFADS: Model Failure Analysis & Debugging System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MFADS is a production-grade diagnostic framework designed to automate the discovery, clustering, and explanation of ML model failures. Instead of looking at aggregate metrics, MFADS enables engineers to deep-dive into **systematic failure modes** and **unseen biases**.

## 🚀 Problem Statement
Traditional ML evaluation often hides critical performance gaps in high-level metrics (e.g., 90% F1 might mask 0% accuracy on rare tokens). MFADS automates the "Model Unit Testing" process to find these gaps before deployment.

## ✨ Core Features
- 🔍 **Automated Slicing**: Identifies performance drops based on metadata (length, rarity, labels).
- 🧩 **Error Clustering**: Uses `all-MiniLM-L6-v2` embeddings to group misclassified examples semantically.
- 💡 **Interpretable Signals**: Feature importance for baseline and attribution heuristics for transformers.
- 📉 **Bias Detection**: Statistical identification of significant subgroup performance gaps.
- 🛠️ **Actionable Reports**: Generates automated "Next Steps" to fix the model (data collection, augmentation, calibration).

## 🏗️ Architecture
- **Analytical Core**: Modular slicing engine and error clusterer.
- **Explainability Layer**: Local and Global explanations for both Linear and Transformer models.
- **Reporting Engine**: Markdown/JSON artifact generator for CI/CD integration.
- **Interactive UI**: Streamlit-based failure exploration dashboard.

---

## 🛠️ Getting Started

### 1. Installation
```bash
# Clone and install in editable mode
pip install -e .
```

### 2. Run Pipeline
```bash
# Train the baseline (TF-IDF + LogReg)
mfads train --model tfidf

# Run deep analysis and generate report
mfads analyze --model tfidf

# Explore results interactively
streamlit run streamlit_app/app.py
```

---

## 🧠 What I Learned
- **The "Slicing" Challenge**: Defining meaningful slices is harder than it looks. Heuristics like "rare token ratio" are effective but require careful corpus-level statistics.
- **Clustering Interpretability**: Raw clusters aren't useful without keywords. Using TF-IDF per cluster to extract "failure themes" makes the analysis much more actionable for product teams.
- **MLOps for Debugging**: Real-world ML is 90% debugging and 10% modeling. Building an automated system for this saves weeks of manual inspection.

## ⚖️ Design Trade-offs
- **KMeans vs HDBSCAN**: Chose KMeans for simplicity and deterministic speed in a demo environment, though HDBSCAN would better handle noise in higher-dimensional embedding spaces.
- **TF-IDF Explanations**: Used native coefficients for speed. While SHAP is more universal, coefficient inspection is instantaneous and sufficient for linear debugging.

---

## 📝 Resume Highlights
- **Engineered** an automated model diagnostic framework (MFADS) that reduced manual error analysis time by **[XX]%** through automated slice discovery and semantic error clustering.
- **Implemented** a multi-modal interpretability layer for both Transformer and Linear models, facilitating the identification of **[XX]** distinct failure modes in text classification tasks.
- **Developed** an interactive Streamlit dashboard for stakeholder reporting, visualizing subgroup performance gaps and providing automated data collection recommendations.

---

## 📂 Repository Structure
```text
mfads/
├── config/             # YAML configs for slicing thresholds
├── src/mfads/
│   ├── clustering/     # Embedding and KMeans logic
│   ├── slicing/        # Feature extraction and ranking
│   ├── explain/        # Model interpretability
│   └── report/         # Automated recommendation engine
├── streamlit_app/      # Interactive Dashboard
└── tests/              # Correctness verification
```
