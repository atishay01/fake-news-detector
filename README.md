# Multimodal Fake News Detector

A **multimodal** fake/real news classifier with a Streamlit dashboard. Paste an article, optionally attach an image and a source URL — the system produces an explainable verdict by combining three independent signals: the article text, the image, and article metadata.

**Live demo:** <https://atishay-fake-news-detector.streamlit.app>
**Author:** Atishay Jain

---

## What it does

- Classifies English news as **FAKE / REAL / UNCERTAIN** using **three independent modalities**, then fuses them:
  1. **Text** — RoBERTa transformer (default) or TF-IDF + soft-voting ensemble baseline.
  2. **Image** — CLIP image↔text cosine similarity ("does the photo actually match the article?") plus lightweight image-origin forensics (EXIF, dimensions, format).
  3. **Metadata** — clickbait markers, linguistic features (caps ratio, sentence length, named-entity density) and source-domain reputation from a hand-curated list of 50 outlets.
- Uses **late fusion** — each modality is scored independently, then combined with a confidence-weighted average. The UI exposes per-modality scores and the exact weights used, so the final verdict is fully auditable.
- Sidebar controls let the user **toggle modalities on/off** and **re-weight** them live — useful for demoing ablations.
- Ships with a reproducible training script for the text baseline and held-out test metrics.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Text features | TF-IDF (1-2 grams, 50K vocab, sublinear TF) | Fast, transparent, strong baseline on news text |
| Text model (baseline) | Soft-voting **ensemble** of Logistic Regression + MultinomialNB + ComplementNB | Real ensemble — LogReg gives calibrated probabilities + interpretable coefficients, NB variants add stability on word-count features |
| Text model (accurate) | Pre-trained RoBERTa (`hamzab/roberta-fake-news-classification`) | Strong OOD behaviour; user can toggle to this in the UI |
| Image modality | CLIP `openai/clip-vit-base-patch32` zero-shot similarity | Catches decontextualised images — the classic "same photo reused for a different story" fake-news pattern |
| Image forensics | Pillow (EXIF, dimensions, format) | Cheap origin heuristics — distinguishes wire photo from screenshot/meme |
| Metadata module | Pure-Python heuristics + `data/source_credibility.json` | Fast, interpretable, no model to retrain |
| Fusion | Confidence-weighted late fusion | Simpler and more interpretable than early fusion; each modality's contribution is visible |
| UI | Streamlit (wide layout, tabs, sliders) | Zero-boilerplate dashboard, free hosting |
| Hosting | Streamlit Community Cloud | Free, auto-deploys on `git push` |
| Training data | `GonzaloA/fake_news` (HuggingFace) | Labeled English fake/real news, ~32K rows |

## Project layout

```
.
├── app.py                 # Streamlit dashboard (multimodal input + fusion UI)
├── multimodal.py          # CLIP image-text match, image forensics,
│                          #   metadata heuristics, late-fusion layer
├── train.py               # Trains the TF-IDF baseline and saves artifacts
├── requirements.txt       # Runtime deps (used by Streamlit Cloud)
├── requirements-train.txt # Training deps (datasets, matplotlib)
├── packages.txt           # System deps for Streamlit Cloud (tesseract-ocr)
├── models/
│   └── fake_news_model.joblib   # Trained TF-IDF pipeline (created by train.py)
├── reports/
│   ├── confusion_matrix.png     # Held-out confusion matrix
│   └── metrics.txt              # Classification report + ROC-AUC
├── notebooks/
│   └── baseline_eda.ipynb       # Original exploratory notebook
├── docs/
│   ├── project_report.pdf       # Full project report
│   ├── presentation.pptx        # Project presentation
│   └── bibtex.txt               # Citations
└── data/
    └── source_credibility.json  # Hand-curated domain reputation list (50 outlets)
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

### Note on the live demo

The CLIP image–text modality is **disabled by default** on the live Streamlit Cloud deployment because the combined footprint of RoBERTa (~500 MB) + CLIP (~600 MB) exceeds the free tier's ~1 GB memory limit. All other modalities (TF-IDF / RoBERTa text, image forensics, metadata) run unchanged. To exercise the full multimodal pipeline including CLIP, run the app locally with the command above, or toggle it on in the sidebar and accept that the hosted app may OOM. This is a deployment constraint, not a code constraint — on any instance with ≥ 2 GB RAM everything runs together.

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

## Multimodal architecture

```
                  ┌────────────────────┐
  text  ────────► │ RoBERTa / TF-IDF   │ ──► text score + confidence
                  └────────────────────┘
                  ┌────────────────────┐
  image ────────► │ CLIP text↔image    │ ──► semantic-match score
                  │  cosine similarity │
                  └────────────────────┘
                  ┌────────────────────┐
  image ────────► │ Pillow forensics   │ ──► origin score
                  │  (EXIF / dim / fmt)│
                  └────────────────────┘
                  ┌────────────────────┐
  text + url ───► │ Metadata heuristics│ ──► credibility score
                  │  + source DB       │
                  └────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  confidence-weighted   │ ──►  REAL / FAKE / UNCERTAIN
                 │    late fusion         │      + per-modality breakdown
                 └────────────────────────┘
```

Each modality returns a `ModalitySignal(score, confidence, label, details)`. A signal with `confidence=0` (e.g. no image provided) is dropped from the fusion and the remaining weights are renormalised, so the system degrades gracefully when fewer modalities are available.

## Roadmap (good interview talking points)

- [x] Add multimodal signals (image + metadata) and a late-fusion layer
- [x] Expose per-modality scores in the UI for auditability
- [ ] Evaluate on an out-of-distribution set (LIAR or FakeNewsNet) to quantify drift
- [ ] Replace heuristic metadata with a small learned model trained on labelled URL features
- [ ] Add a "why this prediction" SHAP panel alongside the current TF-IDF signal tokens
- [ ] Log predictions to SQLite so the dashboard can show a usage history

## License

MIT
