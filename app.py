from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "breast_cancer_wisconsin_diagnostic.csv"

st.set_page_config(page_title="Random Forest Classification", page_icon="🌲", layout="wide")
st.title("🌲 Predictive Classification using Random Forest Ensembles")
st.caption("AIML Minor Project 4 | Breast Cancer Wisconsin (Diagnostic) Dataset")
st.info("Educational ML demonstration only. This model is not a medical diagnostic tool.")

if not (OUT / "metrics.json").exists():
    st.warning("Run `python train_model.py` first to create evaluation outputs.")
    st.code("python train_model.py\nstreamlit run app.py")
    st.stop()

metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
c2.metric("Precision (M)", f"{metrics['precision_malignant']:.3f}")
c3.metric("Recall (M)", f"{metrics['recall_malignant']:.3f}")
c4.metric("F1-score (M)", f"{metrics['f1_malignant']:.3f}")
c5.metric("ROC-AUC", f"{metrics['roc_auc_malignant']:.3f}")

st.subheader("Dataset overview")
df = pd.read_csv(DATA)
a, b, c = st.columns(3)
a.metric("Records", len(df))
b.metric("Predictor features", len(df.columns)-1)
c.metric("Missing values", int(df.isnull().sum().sum()))
st.write("Target labels: B = benign, M = malignant. The model treats malignant (M) as the positive class.")
st.dataframe(df.head(10), use_container_width=True)

tabs = st.tabs(["EDA", "Confusion Matrix", "ROC Curve", "Feature Importance", "Reports"])
with tabs[0]:
    for name in ["class_distribution.png", "feature_distributions.png", "correlation_heatmap.png"]:
        p = OUT / name
        if p.exists():
            st.image(str(p), caption=name.replace("_", " ").replace(".png", "").title(), use_container_width=True)
with tabs[1]:
    st.image(str(OUT / "confusion_matrix.png"), use_container_width=True)
    st.write({k: metrics[k] for k in ["TN", "FP", "FN", "TP"]})
with tabs[2]:
    st.image(str(OUT / "roc_curve.png"), use_container_width=True)
with tabs[3]:
    st.image(str(OUT / "feature_importance.png"), use_container_width=True)
    st.dataframe(pd.read_csv(OUT / "feature_importance.csv").head(15), use_container_width=True)
with tabs[4]:
    st.code((OUT / "classification_report.txt").read_text(encoding="utf-8"))
    for fname in ["metrics.json", "eda_summary.csv", "missing_values.csv", "test_predictions.csv"]:
        p = OUT / fname
        if p.exists():
            st.download_button(f"Download {fname}", p.read_bytes(), file_name=fname)
