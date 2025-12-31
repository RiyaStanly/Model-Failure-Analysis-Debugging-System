import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MFADS Dashboard", layout="wide")

def load_latest_report():
    # Look for reports relative to this file's location
    report_base = os.path.join(os.path.dirname(__file__), "..", "reports")
    if not os.path.exists(report_base):
        return None
    
    runs = sorted([d for d in os.listdir(report_base) if d.startswith("run_")], reverse=True)
    if not runs:
        return None
    
    with open(os.path.join(report_base, runs[0], "artifacts.json")) as f:
        return json.load(f)

st.title("🛡️ MFADS: Model Failure Analysis & Debugging")

data = load_latest_report()

if not data:
    st.warning("No analysis reports found. Run `python -m mfads.cli analyze` first.")
else:
    # 1. Overview Metrics
    st.header("1. Performance Overview")
    col1, col2 = st.columns(2)
    col1.metric("Overall F1 Score", f"{data['overall']['f1']:.4f}")
    col2.metric("Overall Accuracy", f"{data['overall']['accuracy']:.4f}")

    # 2. Worst Slices
    st.header("2. Sliced Performance Breakdown")
    metrics_df = pd.DataFrame(data["metrics"])
    
    fig = px.bar(
        metrics_df, 
        x="slice", 
        y="f1", 
        color="f1", 
        title="F1 Score by Slice",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Worst Performing Slices")
    st.table(pd.DataFrame(data["worst_slices"]))

    # 3. Bias Gaps
    if data["bias_gaps"]:
        st.header("3. Bias & Performance Gaps")
        st.info("The following slices show significant drops compared to overall performance.")
        st.table(pd.DataFrame(data["bias_gaps"]))

    # 4. Cluster Browser
    st.header("4. Error Mode Clustering")
    clusters = data["clusters"]
    
    if clusters:
        selected_cluster = st.selectbox(
            "Select a cluster to explore failure mode:",
            options=[c["cluster_id"] for c in clusters],
            format_func=lambda x: f"Cluster {x} ({next(c['size'] for c in clusters if c['cluster_id'] == x)} cases)"
        )
        
        c_info = next(c for c in clusters if c["cluster_id"] == selected_cluster)
        st.write(f"**Keywords**: {', '.join(c_info['keywords'])}")
        st.info(f"**Representative Example**: {c_info['example']}")
        
        # Show more examples from this cluster if they exist in artifact
        errors_df = pd.DataFrame(data["errors"])
        if "cluster" in errors_df.columns:
            cluster_examples = errors_df[errors_df["cluster"] == selected_cluster]
            st.subheader("Examples in this cluster")
            st.dataframe(cluster_examples[["text", "label", "prediction"]])

    # 5. Recommendations
    st.header("5. Actionable Next Steps")
    for rec in data["recommendations"]:
        st.write(f"- [ ] {rec}")
