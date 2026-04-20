"""Train a TF-IDF + Logistic Regression classifier for fake news detection.

Data source priority:
  1. Local CSV at data/train.csv with columns: text, label (1=REAL, 0=FAKE)
  2. HuggingFace dataset 'GonzaloA/fake_news' (auto-downloaded)

Outputs:
  - models/fake_news_model.joblib
  - reports/confusion_matrix.png
  - reports/metrics.txt
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    local = ROOT / "data" / "train.csv"
    if local.exists():
        df = pd.read_csv(local)
        assert {"text", "label"}.issubset(df.columns), (
            "data/train.csv must have columns: text, label"
        )
        return df[["text", "label"]]

    print("Local data/train.csv not found — fetching 'GonzaloA/fake_news' from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("GonzaloA/fake_news")
    parts = []
    for split in ("train", "validation"):
        part = ds[split].to_pandas()[["title", "text", "label"]]
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    return df[["text", "label"]]


def main() -> None:
    df = load_data().dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1)),
            ("mnb", MultinomialNB(alpha=0.3)),
            ("cnb", ComplementNB(alpha=0.3)),
        ],
        voting="soft",
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            stop_words="english",
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", ensemble),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"])
    auc = roc_auc_score(y_test, y_proba)

    print("=== Classification Report ===")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    (REPORTS / "metrics.txt").write_text(
        f"{report}\nROC-AUC: {auc:.4f}\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Fake News Detection — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)

    model_path = MODELS / "fake_news_model.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nModel saved -> {model_path}")
    print(f"Confusion matrix saved -> {REPORTS / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
