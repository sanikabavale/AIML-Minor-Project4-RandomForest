"""Download the official Kaggle dataset and normalize it for this project.

Prerequisite: install Kaggle API (`pip install kaggle`) and configure Kaggle API credentials.
Kaggle dataset: uciml/breast-cancer-wisconsin-data
Run from this folder: python download_kaggle_dataset.py
"""
from pathlib import Path
import subprocess
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DOWNLOAD_DIR = DATA / "kaggle_download"
DOWNLOAD_DIR.mkdir(exist_ok=True)

subprocess.run([
    "kaggle", "datasets", "download", "-d", "uciml/breast-cancer-wisconsin-data",
    "-p", str(DOWNLOAD_DIR), "--force"
], check=True)
zip_files = list(DOWNLOAD_DIR.glob("*.zip"))
if not zip_files:
    raise FileNotFoundError("Kaggle download did not produce a ZIP file.")
with zipfile.ZipFile(zip_files[0]) as zf:
    candidates = [n for n in zf.namelist() if Path(n).name.lower() == "data.csv"]
    if not candidates:
        raise FileNotFoundError("Could not find data.csv in downloaded Kaggle archive.")
    with zf.open(candidates[0]) as f:
        df = pd.read_csv(f)

def normalize(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    drop_cols = [c for c in frame.columns if c.lower().startswith("unnamed:") or c.lower() == "id"]
    frame = frame.drop(columns=drop_cols, errors="ignore")
    rename = {}
    for c in frame.columns:
        low = c.lower().strip()
        if low == "diagnosis": rename[c] = "diagnosis"
        elif low.endswith("_mean"): rename[c] = "mean " + low[:-5].replace("_", " ")
        elif low.endswith("_se"): rename[c] = low[:-3].replace("_", " ") + " error"
        elif low.endswith("_worst"): rename[c] = "worst " + low[:-6].replace("_", " ")
        else: rename[c] = low.replace("_", " ")
    frame = frame.rename(columns=rename).dropna(axis=1, how="all")
    return frame

out = normalize(df)
if "diagnosis" not in out.columns or len(out) == 0:
    raise ValueError("Downloaded file did not match the expected Kaggle dataset.")
out.to_csv(DATA / "breast_cancer_wisconsin_diagnostic.csv", index=False)
print(f"Saved {len(out)} rows and {len(out.columns)-1} predictors to {DATA / 'breast_cancer_wisconsin_diagnostic.csv'}")
print("Now run: python train_model.py")
