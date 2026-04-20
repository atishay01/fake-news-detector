"""Streamlit dashboard for fake news detection.

Two models:
  - Accurate (RoBERTa, pre-trained on HuggingFace)  <-- default
  - Fast (TF-IDF + Logistic Regression, trained by train.py)

Three input modes: Paste text · Fetch from URL · Upload image (OCR).

Run locally:
    streamlit run app.py
"""
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

MODEL_PATH = Path(__file__).parent / "models" / "fake_news_model.joblib"
TRANSFORMER_MODEL = "hamzab/roberta-fake-news-classification"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

st.title("Fake News Detector")
st.caption(
    "RoBERTa transformer (default) · TF-IDF baseline (toggle) · "
    "text / URL / image input"
)


@st.cache_resource(show_spinner="Loading TF-IDF baseline...")
def load_tfidf():
    if not MODEL_PATH.exists():
        st.error(
            f"Baseline model not found at `{MODEL_PATH}`. "
            "Run `python train.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading RoBERTa — first run downloads ~500 MB...")
def load_transformer():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=TRANSFORMER_MODEL,
        truncation=True,
        max_length=512,
    )


def predict_tfidf(pipe, text: str):
    proba = pipe.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(proba[pred])
    return label, confidence


def predict_transformer(pipe, text: str):
    r = pipe(text)[0]
    raw = r["label"].upper()
    label = "REAL" if raw in {"TRUE", "REAL", "LABEL_1"} else "FAKE"
    return label, float(r["score"])


def top_signal_tokens(pipe, text: str, k: int = 6):
    """Interpret via the LR coefficients (works whether clf is LR or a VotingClassifier
    containing LR)."""
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "named_estimators_"):
        lr = clf.named_estimators_.get("lr")
        if lr is None or not hasattr(lr, "coef_"):
            return []
        coefs = lr.coef_[0]
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
    else:
        return []

    X = vec.transform([text])
    contribs = X.multiply(coefs).toarray().ravel()
    feature_names = np.array(vec.get_feature_names_out())
    nz = np.where(contribs != 0)[0]
    if len(nz) == 0:
        return []
    order = nz[np.argsort(np.abs(contribs[nz]))[::-1]][:k]
    return [(feature_names[i], float(contribs[i])) for i in order]


def fetch_article_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    article = soup.find("article")
    container = article if article else soup
    blocks = container.find_all(["h1", "h2", "p"])
    text = "\n".join(b.get_text(strip=True) for b in blocks if b.get_text(strip=True))
    if not text or len(text) < 80:
        raise ValueError(
            "Could not extract enough article text from this URL. "
            "The site may be JavaScript-rendered or paywalled. "
            "Copy the article body and paste it directly instead."
        )
    return text[:10_000]


def clean_ocr_text(raw: str) -> str:
    kept = []
    for line in raw.splitlines():
        s = line.strip()
        if len(s) < 20:
            continue
        alpha = sum(c.isalpha() for c in s)
        if alpha / max(len(s), 1) < 0.55:
            continue
        kept.append(s)
    return "\n".join(kept)


def ocr_image(file_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "OCR dependencies missing. Run: pip install pytesseract Pillow"
        ) from e

    win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if win_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(win_path)

    try:
        img = Image.open(BytesIO(file_bytes))
        raw = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract binary not found. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki, "
            "then fully close and reopen the terminal running Streamlit."
        ) from e

    text = clean_ocr_text(raw).strip()
    if len(text) < 80:
        raise ValueError(
            "OCR extracted mostly UI / chrome noise after cleanup. Tip: "
            "crop your screenshot to just the article body (no taskbar, "
            "browser tabs, clock or weather widgets), then try again."
        )
    return text


# -------------------- UI --------------------

if "main_text" not in st.session_state:
    st.session_state.main_text = ""

model_choice = st.radio(
    "Model",
    options=["Accurate (RoBERTa)", "Fast (TF-IDF baseline)"],
    horizontal=True,
    help=(
        "RoBERTa generalises well to modern news. TF-IDF is a fast, interpretable "
        "baseline I trained from scratch — it performs well on Reuters-style copy "
        "but shows measurable out-of-distribution drift (see Model Card)."
    ),
)

tab_text, tab_url, tab_img = st.tabs(["Paste text", "Fetch from URL", "Upload image"])

with tab_text:
    st.caption("Type or paste a news headline or article body into the text box below.")

with tab_url:
    url = st.text_input(
        "Article URL",
        placeholder="https://www.bbc.com/news/world-...",
        key="url_input",
    )
    if st.button("Fetch article", key="fetch_btn"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Fetching..."):
                    fetched = fetch_article_text(url.strip())
                st.session_state.main_text = fetched
                st.success(
                    f"Fetched {len(fetched):,} characters. "
                    "Scroll down to review, then click Check."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Fetch failed: {e}")

with tab_img:
    uploaded = st.file_uploader(
        "Upload a news image (screenshot, photo of an article, etc.)",
        type=["png", "jpg", "jpeg", "webp"],
        key="img_uploader",
    )
    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        if st.button("Extract text with OCR", key="ocr_btn"):
            try:
                with st.spinner("Running OCR..."):
                    extracted = ocr_image(uploaded.getvalue())
                st.session_state.main_text = extracted
                st.success(
                    f"Extracted {len(extracted):,} characters. "
                    "Scroll down to review, then click Check."
                )
                st.rerun()
            except Exception as e:
                st.error(f"OCR failed: {e}")

st.divider()

text = st.text_area(
    "News article text:",
    height=240,
    placeholder=(
        "Paste article text here, or use the 'Fetch from URL' / 'Upload image' "
        "tabs above to populate this automatically."
    ),
    key="main_text",
)

go = st.button("Check", type="primary")

if go:
    current = (text or "").strip()
    if not current:
        st.warning("Please enter some text, fetch a URL, or upload an image first.")
        st.stop()

    if current.startswith(("http://", "https://")) and "\n" not in current:
        st.warning(
            "This looks like a bare URL. Use the 'Fetch from URL' tab above — "
            "the model classifies article text, not links."
        )
        st.stop()

    using_transformer = model_choice.startswith("Accurate")

    if using_transformer:
        pipe = load_transformer()
        label, confidence = predict_transformer(pipe, current)
        signals = []
    else:
        pipe = load_tfidf()
        label, confidence = predict_tfidf(pipe, current)
        signals = top_signal_tokens(pipe, current)

    UNCERTAIN_THRESHOLD = 0.70

    if confidence < UNCERTAIN_THRESHOLD:
        st.warning(
            f"Prediction: **UNCERTAIN**  ·  leaning {label}, confidence {confidence:.1%}\n\n"
            "The model is not confident enough to commit. See the **Model Card** below "
            "for known limitations."
        )
    elif label == "REAL":
        st.success(f"Prediction: **REAL**   ·   confidence {confidence:.1%}")
    else:
        st.error(f"Prediction: **FAKE**   ·   confidence {confidence:.1%}")
    st.progress(confidence)

    st.caption(f"Model used: `{TRANSFORMER_MODEL}`" if using_transformer
               else "Model used: TF-IDF + Logistic Regression (trained from scratch)")

    if signals:
        st.subheader("Top signal tokens (TF-IDF baseline only)")
        for token, weight in signals:
            direction = "-> REAL" if weight > 0 else "-> FAKE"
            st.write(f"`{token}`  {direction}  (weight {weight:+.3f})")

    with st.expander("How this works"):
        st.markdown(
            "**Two-model setup:**\n\n"
            "- **RoBERTa** (accurate): a pre-trained transformer from HuggingFace "
            f"(`{TRANSFORMER_MODEL}`). Strong out-of-distribution behaviour because it "
            "learned semantic representations from a much larger corpus.\n"
            "- **TF-IDF + Logistic Regression** (fast): trained from scratch on "
            "`GonzaloA/fake_news`. Interpretable — the *signal tokens* show which "
            "words pushed the decision.\n\n"
            "Held-out metrics for the TF-IDF baseline: 98% accuracy, ROC-AUC 0.9971 "
            "(see `reports/metrics.txt`)."
        )

    with st.expander("Model Card — honest evaluation"):
        st.markdown(
            "**Baseline:** TF-IDF + soft-voting ensemble of LogReg + MultinomialNB + "
            "ComplementNB. Test accuracy 96%, ROC-AUC 0.994.\n\n"
            "**Measured out-of-distribution behaviour** (real samples through both models):\n\n"
            "| Input style | Ensemble says | RoBERTa says |\n"
            "|---|---|---|\n"
            "| Reuters-style US wire copy | REAL 99.4% | REAL 100% |\n"
            "| Modern Indian / BBC / AP-captioned | REAL 72.9% | REAL 100% |\n"
            "| Peer-reviewed science news | REAL 63.5% | REAL 100% |\n"
            "| Classic clickbait | FAKE 96.2% | FAKE 100% |\n\n"
            "**Why the ensemble helps:** LogReg alone fit the Reuters-dominated REAL "
            "class closely and drifted on modern news. Adding MultinomialNB + "
            "ComplementNB shifts the aggregated probability away from single-model "
            "stylistic overfit.\n\n"
            "**Why RoBERTa does even better:** pre-trained on a much larger, more "
            "diverse corpus, so it latches onto semantic content rather than surface "
            "style.\n\n"
            "**UNCERTAIN band:** predictions below 70% confidence are shown as "
            "UNCERTAIN rather than forcing a FAKE / REAL call."
        )

st.divider()
st.caption(
    "Limitations: English news only. May misclassify satire, opinion pieces, or "
    "topics far outside both models' training distributions. OCR quality depends "
    "on image clarity. Treat predictions as a signal, not a verdict."
)
