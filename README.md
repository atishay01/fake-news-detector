# Fake News Detector

A lightweight fake/real news classifier with a Streamlit dashboard. Paste a headline or article, get a prediction with confidence and the tokens that drove the decision.

**Live demo:** _add your Streamlit Cloud URL after deployment_
**Author:** Atishay Jain

---

## What it does

- Classifies English news text as **FAKE** or **REAL** with a probability score.
- Surfaces the **top signal tokens** — the words whose TF-IDF × model weight pushed the decision the hardest. This makes the prediction auditable instead of a black box.
- Ships with a reproducible training script and held-out test metrics.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Features | TF-IDF (1-2 grams, 50K vocab, sublinear TF) | Fast, transparent, strong baseline on news text |
| Model (baseline) | Soft-voting **ensemble** of Logistic Regression + MultinomialNB + ComplementNB | Real ensemble — LogReg gives calibrated probabilities + interpretable coefficients, NB variants add stability on word-count features |
| Model (accurate) | Pre-trained RoBERTa (`hamzab/roberta-fake-news-classification`) | Strong OOD behaviour; user can toggle to this in the UI |
| UI | Streamlit | Zero-boilerplate dashboard, free hosting |
| Hosting | Streamlit Community Cloud | Free, auto-deploys on `git push` |
| Training data | `GonzaloA/fake_news` (HuggingFace) | Labeled English fake/real news, ~32K rows |

## Project layout

```
.
├── app.py                 # Streamlit dashboard (3 input modes: text / URL / image)
├── train.py               # Trains the model and saves artifacts
├── requirements.txt       # Runtime deps (used by Streamlit Cloud)
├── requirements-train.txt # Training deps (datasets, matplotlib)
├── packages.txt           # System deps for Streamlit Cloud (tesseract-ocr)
├── models/
│   └── fake_news_model.joblib   # Trained pipeline (created by train.py)
├── reports/
│   ├── confusion_matrix.png     # Held-out confusion matrix
│   └── metrics.txt              # Classification report + ROC-AUC
├── notebooks/
│   └── baseline_eda.ipynb       # Original exploratory notebook
├── docs/
│   ├── project_report.pdf       # Full project report
│   ├── presentation.pptx        # Project presentation
│   └── bibtex.txt               # Citations
└── data/                  # Local-only (gitignored); drop train.csv here to override HF
```

## Quickstart (local)

```bash
# 1. Set up a virtual env
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux

# 2. Train the model (one time, ~2 minutes on laptop CPU)
pip install -r requirements-train.txt
python train.py

# 3. Run the dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Deploy to the web (Streamlit Community Cloud — free)

1. Create a public GitHub repo and push this folder:
   ```bash
   git init
   git add app.py train.py requirements.txt requirements-train.txt README.md .gitignore models reports
   git commit -m "Initial commit: fake news detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/fake-news-detector.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io> → **New app** → pick the repo → main file = `app.py` → **Deploy**.
3. Streamlit Cloud installs `requirements.txt` and loads `models/fake_news_model.joblib` directly — no retraining on deploy.
4. Copy the resulting URL back into the **Live demo** line at the top of this README, and put the same URL on the resume next to this project.

## Why not Google Colab?

Colab is great for exploration and one-off training, but it is a poor choice for a demo you want to show an interviewer:

- Sessions expire, re-uploading data gets tedious, and sharing a Colab link means the reviewer has to run cells.
- Colab notebooks are not a deployed product — recruiters expect a live URL they can click.
- If the dataset was uploaded mid-session, the link breaks the moment the runtime disconnects, which matches what you observed.

**Use this instead:** train locally (or on Colab if you want the GPU), commit the saved `.joblib` to GitHub, let Streamlit Cloud serve it. The reviewer clicks one URL and gets a working app.

## Results

Run `python train.py` and the metrics land in `reports/metrics.txt`. Typical numbers on this dataset:

| Metric | Value |
|---|---|
| Accuracy | 0.96 |
| Precision (FAKE) | 0.94 |
| Recall (FAKE) | 0.97 |
| ROC-AUC | 0.994 |

> The `GonzaloA/fake_news` dataset is relatively clean and on-distribution, so numbers are high. For a more realistic stress test, see the Model Card below — OOD drift is real and measured, not glossed over.

## Model Card (honest evaluation)

**Training data:** `GonzaloA/fake_news` on HuggingFace, ~32K English news items.
**Baseline:** TF-IDF + soft-voting ensemble (LogReg + MultinomialNB + ComplementNB).
**Held-out test accuracy:** 96%  ·  **ROC-AUC:** 0.994

**Measured out-of-distribution behaviour** (real samples, baseline ensemble):

| Input style | Ensemble says | RoBERTa says |
|---|---|---|
| Reuters-style US wire copy | REAL, 99.4% | REAL, 100% |
| Modern Indian / BBC / AP-captioned | REAL, 72.9% | REAL, 100% |
| Peer-reviewed science news | REAL, 63.5% | REAL, 100% |
| Classic clickbait / sensational copy | FAKE, 96.2% | FAKE, 100% |

**Why this happens:** The REAL class in training is dominated by Reuters wire copy. The model learned stylistic cues (datelines, formal quotes) rather than semantic truth. Factually real articles in a different style drift toward FAKE.

**UNCERTAIN band:** The dashboard shows predictions below 70% confidence as `UNCERTAIN` rather than forcing a FAKE / REAL call. This is honest and matches how the model actually behaves on OOD input.

**Next iteration (roadmap step 1):** DistilBERT fine-tuned on a diverse mix (LIAR + FakeNewsNet + ISOT) to reduce the stylistic bias and raise OOD performance.

## Roadmap (good interview talking points)

- [ ] Evaluate on an out-of-distribution set (LIAR or FakeNewsNet) to quantify drift
- [ ] Add a DistilBERT fine-tune as a second model, compare latency vs accuracy
- [ ] Add a "why this prediction" LIME/SHAP panel alongside the current signal tokens
- [ ] Log predictions to SQLite so the dashboard can show a usage history

## License

MIT
