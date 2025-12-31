import json
import os
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

def generate_markdown_report(
    overall_metrics: Dict[str, float],
    worst_slices: pd.DataFrame,
    bias_gaps: pd.DataFrame,
    cluster_summaries: List[Dict],
    recommendations: List[str],
    output_path: str
):
    """Generates a comprehensive Markdown report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# MFADS Failure Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Overall Performance\n")
        for metric, value in overall_metrics.items():
            f.write(f"- **{metric}**: {value:.4f}\n")
        f.write("\n")
        
        f.write("## 2. Worst Performing Slices\n")
        f.write(worst_slices.to_markdown(index=False))
        f.write("\n\n")
        
        if not bias_gaps.empty:
            f.write("## 3. Detected Bias / Performance Gaps\n")
            f.write(bias_gaps.to_markdown(index=False))
            f.write("\n\n")
        
        f.write("## 4. Failure Mode Clusters\n")
        for cluster in cluster_summaries:
            f.write(f"### Cluster {cluster['cluster_id']} (Size: {cluster['size']})\n")
            f.write(f"- **Top Keywords**: {', '.join(cluster['keywords'])}\n")
            f.write(f"- **Representative Example**: _\"{cluster['example']}\"_\n\n")
            
        f.write("## 5. Actionable Recommendations\n")
        for rec in recommendations:
            f.write(f"- [ ] {rec}\n")
            
    print(f"Report generated at {output_path}")

def generate_recommendations(worst_slices: pd.DataFrame, bias_gaps: pd.DataFrame) -> List[str]:
    """Automated recommendation engine based on analysis."""
    recommendations = []
    
    if not worst_slices.empty:
        top_worst = worst_slices.iloc[0]["slice"]
        recommendations.append(f"Collect more diverse training data for the '{top_worst}' slice.")
        
    for slice_name in bias_gaps["slice"]:
        recommendations.append(f"Analyze feature distribution in '{slice_name}' to check for label imbalance.")
        
    recommendations.append("Apply threshold calibration to improve precision on high-error slices.")
    recommendations.append("Consider data augmentation (e.g., back-translation) for short-text failure cases.")
    
    return recommendations

def save_artifacts(data: Dict[str, Any], output_dir: str):
    """Saves all analysis artifacts as JSON for the dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle pandas objects for JSON serialization
    serialized_data = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            serialized_data[k] = v.to_dict(orient="records")
        else:
            serialized_data[k] = v
            
    with open(os.path.join(output_dir, "artifacts.json"), 'w') as f:
        json.dump(serialized_data, f, indent=4)
