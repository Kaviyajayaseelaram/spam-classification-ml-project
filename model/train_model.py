# ============================================================
#  model/train_model.py  –  Training Pipeline
#  Trains Naive Bayes on TF-IDF features and saves model + metrics
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline

from model.preprocess import preprocess_text

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH    = "data/spam.csv"
MODEL_PATH   = "model/model.pkl"
METRICS_PATH = "model/metrics.pkl"


def load_data() -> pd.DataFrame:
    """
    Load and clean the SMS Spam Collection dataset.
    Expected columns after rename: label (ham/spam), text (message).
    """
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Dataset ships with 5 columns; keep only the first two
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]

    # Drop duplicates and missing values
    df = df.drop_duplicates().dropna()

    print(f"[INFO] Loaded {len(df)} messages  |  Spam: {(df.label=='spam').sum()}  Ham: {(df.label=='ham').sum()}")
    return df


def build_pipeline() -> Pipeline:
    """
    Build a Scikit-learn Pipeline:
      TF-IDF Vectoriser  →  Multinomial Naive Bayes

    TF-IDF settings
    ---------------
    - ngram_range (1,2): unigrams + bigrams  (e.g. 'free prize' as one feature)
    - max_features 10000: vocabulary cap to avoid overfitting
    - min_df 2: ignore terms that appear in < 2 documents
    - sublinear_tf True: apply log(1 + tf) to dampen high-frequency terms
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        min_df=2,
        sublinear_tf=True,
        analyzer="word",
    )

    # alpha=0.1 — mild Laplace smoothing (lower than default 1.0 gave better precision)
    classifier = MultinomialNB(alpha=0.1)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("nb",    classifier),
    ])
    return pipeline


def train_and_save() -> dict:
    """
    Full training run:
      1. Load data
      2. Preprocess all messages
      3. Split 80/20 train/test
      4. Build and fit the pipeline
      5. Evaluate on test set
      6. Save model + metrics to disk

    Returns
    -------
    dict  Metrics dictionary (accuracy, precision, recall, f1, ...)
    """
    # ── 1. Load ────────────────────────────────────────────────
    df = load_data()

    # ── 2. Preprocess ──────────────────────────────────────────
    print("[INFO] Preprocessing messages...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Encode labels:  ham → 0,  spam → 1
    df["label_enc"] = (df["label"] == "spam").astype(int)

    X = df["clean_text"]
    y = df["label_enc"]

    # ── 3. Split ───────────────────────────────────────────────
    # Stratify to preserve class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 4. Train ───────────────────────────────────────────────
    print("[INFO] Training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── 5. Evaluate ────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    # 5-fold cross-validation on full dataset for robustness
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

    print(f"\n{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*40}\n")

    metrics = {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "cv_f1_mean":       float(cv_scores.mean()),
        "cv_f1_std":        float(cv_scores.std()),
        "confusion_matrix": cm,
        "report":           report,
    }

    # ── 6. Save ────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics,  METRICS_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    print(f"[INFO] Metrics saved → {METRICS_PATH}")

    return metrics


# Allow running standalone:  python -m model.train_model
if __name__ == "__main__":
    train_and_save()
