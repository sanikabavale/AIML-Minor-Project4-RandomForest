# AIML Minor Project 4
## Predictive Classification using Random Forest Ensembles

**Student:** Sanika Balaji Bavale  
**Date:** 11 October 2026

### Project overview
This project trains a Random Forest classifier to classify the Breast Cancer Wisconsin (Diagnostic) dataset into:
- `B`: benign (negative class)
- `M`: malignant (positive class)

The included dataset has **569 records** and **30 numeric predictor features**. The train/test split is stratified 80:20 with `random_state=42`.

### Dataset source and transparency
The selected benchmark is **Breast Cancer Wisconsin (Diagnostic)**, which is available on Kaggle:
- Kaggle dataset page: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The project includes the same 569-row Wisconsin Diagnostic benchmark in a cleaned, reproducible CSV format. **Transparency note:** the copy bundled in this ZIP was prepared from scikit-learn's bundled version because this runtime could not fetch Kaggle's file directly. The project now includes `download_kaggle_dataset.py` to download the original Kaggle `data.csv` using the Kaggle API, normalize its columns, and replace the project CSV. If your teacher requires proof of direct Kaggle download, run that helper before submission and keep the downloaded source archive as evidence.

Official Kaggle page: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

To download the original Kaggle file (requires a Kaggle account/API token):
1. Install the Kaggle API: `pip install kaggle`
2. Configure your Kaggle API token as described at https://www.kaggle.com/docs/api
3. Run `python download_kaggle_dataset.py`
4. Run `python train_model.py` again to regenerate all charts and metrics.

### Main requirements covered
- Exploratory data analysis and feature distributions
- Target class distribution and correlation heatmap
- Missing-value report and median imputation pipeline
- Categorical encoding support (the selected predictors are all numeric, so predictor encoding is not needed)
- Stratified 80:20 train/test split
- `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced")`
- Confusion matrix with TN, FP, FN, TP
- Precision, recall, F1-score and ROC-AUC for malignant class
- High-resolution ROC curve and feature-importance charts
- Technical report and Streamlit dashboard

### Why these hyperparameters?
- `n_estimators=200`: a moderately large forest helps stabilize predictions while remaining practical to train.
- `max_depth=10`: limits tree depth to reduce overfitting compared with unconstrained trees.
- `class_weight="balanced"`: compensates for the unequal benign/malignant class counts.
- `random_state=42`: makes the split and model reproducible.

### Run on a laptop
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open a terminal/Command Prompt in this folder.
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Recreate all results:
   ```bash
   python train_model.py
   ```
6. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

### Important educational and safety note
This is a machine-learning coursework demonstration, **not a medical diagnostic system**. The outputs must not be used to make healthcare decisions. Reported metrics are based on one held-out 20% test split and do not prove clinical performance.

### Key files
- `data/breast_cancer_wisconsin_diagnostic.csv` — data used for the model
- `train_model.py` — reproducible training, evaluation, and plot generation
- `app.py` — dashboard
- `outputs/` — metrics, charts, model, predictions, and data summaries
- `Project_Report.pdf` and `Project_Report.docx` — completed report
