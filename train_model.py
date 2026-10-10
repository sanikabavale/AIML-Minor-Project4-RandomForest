"""
AIML Minor Project 4
Predictive Classification using Random Forest Ensembles
Dataset: Breast Cancer Wisconsin (Diagnostic), Kaggle/UCI benchmark.
Run: python train_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin_diagnostic.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)
TARGET = "diagnosis"
POSITIVE_LABEL = "M"  # malignant is treated as the positive class
NEGATIVE_LABEL = "B"  # benign is treated as the negative class

def normalize_kaggle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the original Kaggle data.csv schema to readable feature names."""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    # Original Kaggle file contains an ID and an empty Unnamed: 32 column.
    drop_cols = [c for c in frame.columns if c.lower().startswith("unnamed:")]
    if "id" in [c.lower() for c in frame.columns]:
        drop_cols.extend([c for c in frame.columns if c.lower() == "id"])
    frame = frame.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")
    rename = {}
    for c in frame.columns:
        low = c.lower().strip()
        if low == "diagnosis":
            rename[c] = "diagnosis"
        elif low.endswith("_mean"):
            rename[c] = "mean " + low[:-5].replace("_", " ")
        elif low.endswith("_se"):
            rename[c] = low[:-3].replace("_", " ") + " error"
        elif low.endswith("_worst"):
            rename[c] = "worst " + low[:-6].replace("_", " ")
        else:
            rename[c] = low.replace("_", " ")
    frame = frame.rename(columns=rename)
    # Remove columns that are entirely empty, including the common Kaggle Unnamed: 32 field.
    frame = frame.dropna(axis=1, how="all")
    return frame

df = normalize_kaggle_columns(pd.read_csv(DATA_PATH))
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' missing. Columns: {list(df.columns)}")
if df[TARGET].nunique() != 2:
    raise ValueError("This report script expects a binary target with two classes.")

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
transformers = []
if numeric_cols:
    transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_cols))
# If future datasets have categoricals, they are handled via one-hot encoding.
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
if categorical_cols:
    cat_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("categorical", cat_pipe, categorical_cols))
preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1
    ))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
classes = list(model.named_steps["classifier"].classes_)
proba = model.predict_proba(X_test)
positive_index = classes.index(POSITIVE_LABEL)
y_binary = (y_test == POSITIVE_LABEL).astype(int)
pred_binary = (pd.Series(pred, index=y_test.index) == POSITIVE_LABEL).astype(int)
p_pos = proba[:, positive_index]

metrics = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic) Dataset",
    "dataset_rows": int(len(df)),
    "predictor_features": int(X.shape[1]),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "target": TARGET,
    "positive_class": POSITIVE_LABEL,
    "negative_class": NEGATIVE_LABEL,
    "n_estimators": 200,
    "max_depth": 10,
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_malignant": float(precision_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
    "recall_malignant": float(recall_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
    "f1_malignant": float(f1_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
    "roc_auc_malignant": float(roc_auc_score(y_binary, p_pos)),
    "missing_values_total": int(df.isnull().sum().sum()),
    "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
    "feature_scaling": "Not used: Random Forest is tree-based and scale-insensitive."
}
cm = confusion_matrix(y_test, pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
tn, fp, fn, tp = confusion_matrix(y_binary, pred_binary, labels=[0, 1]).ravel()
metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "confusion_matrix_labels_B_M": cm.tolist()})
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
df.describe().T.to_csv(OUTPUT / "eda_summary.csv")
df.isnull().sum().rename("missing_values").to_csv(OUTPUT / "missing_values.csv")
y.value_counts().rename_axis("diagnosis").reset_index(name="count").to_csv(OUTPUT / "class_distribution.csv", index=False)

plt.figure(figsize=(7, 5))
counts = y.value_counts().reindex(["B", "M"])
plt.bar(["Benign (B)", "Malignant (M)"], counts.values)
plt.title("Target Class Distribution"); plt.ylabel("Number of samples")
plt.tight_layout(); plt.savefig(OUTPUT / "class_distribution.png", dpi=240); plt.close()

selected = ["mean radius", "mean texture", "mean perimeter", "mean area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), selected):
    ax.hist(df.loc[df.diagnosis == "B", col], bins=22, alpha=.7, label="Benign")
    ax.hist(df.loc[df.diagnosis == "M", col], bins=22, alpha=.7, label="Malignant")
    ax.set_title(col.title()); ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
axes[0, 0].legend(); fig.suptitle("Selected Feature Distributions by Diagnosis", y=1.02)
fig.tight_layout(); fig.savefig(OUTPUT / "feature_distributions.png", dpi=240, bbox_inches="tight"); plt.close(fig)

numeric = df.select_dtypes(include="number")
plt.figure(figsize=(12, 10))
plt.imshow(numeric.corr(), aspect="auto", interpolation="nearest")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(numeric.columns)), numeric.columns, rotation=90, fontsize=6)
plt.yticks(range(len(numeric.columns)), numeric.columns, fontsize=6)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout(); plt.savefig(OUTPUT / "correlation_heatmap.png", dpi=240); plt.close()

plt.figure(figsize=(6.5, 5.2)); plt.imshow(cm, interpolation="nearest"); plt.colorbar()
plt.xticks([0, 1], ["Predicted B", "Predicted M"]); plt.yticks([0, 1], ["Actual B", "Actual M"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]): plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
plt.title("Confusion Matrix"); plt.xlabel("Predicted label"); plt.ylabel("Actual label")
plt.tight_layout(); plt.savefig(OUTPUT / "confusion_matrix.png", dpi=240); plt.close()

fpr, tpr, _ = roc_curve(y_binary, p_pos)
plt.figure(figsize=(7, 5.5)); plt.plot(fpr, tpr, label=f"Random Forest (AUC = {metrics['roc_auc_malignant']:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random baseline")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — Malignant Class as Positive"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUTPUT / "roc_curve.png", dpi=240); plt.close()

# Feature importances for numeric features; categorical one-hot features can be added as needed.
rf = model.named_steps["classifier"]
feature_names = numeric_cols.copy()
if categorical_cols:
    enc = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encoder"]
    feature_names += enc.get_feature_names_out(categorical_cols).tolist()
fi = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
fi.to_csv(OUTPUT / "feature_importance.csv", index=False)
top = fi.head(15).sort_values("importance")
plt.figure(figsize=(9, 6)); plt.barh(top["feature"], top["importance"])
plt.title("Top 15 Feature Importances"); plt.xlabel("Importance"); plt.tight_layout()
plt.savefig(OUTPUT / "feature_importance.png", dpi=240); plt.close()

(OUTPUT / "classification_report.txt").write_text(
    classification_report(y_test, pred, labels=["B", "M"], target_names=["Benign (B)", "Malignant (M)"], zero_division=0),
    encoding="utf-8")
pd.DataFrame({"actual_diagnosis": y_test.values, "predicted_diagnosis": pred, "malignant_probability": p_pos}).to_csv(OUTPUT / "test_predictions.csv", index=False)
joblib.dump(model, OUTPUT / "random_forest_pipeline.joblib")
print("Training complete. Outputs saved to:", OUTPUT)
print(json.dumps(metrics, indent=2))
