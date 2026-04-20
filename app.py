"""Streamlit dashboard for the *multimodal* fake-news detector.

Three independent signals, fused late:

    TEXT        - RoBERTa (default) or TF-IDF + Logistic-Regression baseline.
    IMAGE       - CLIP image/text cosine similarity ("does the photo actually
                  match the article?") + cheap image-origin forensics.
    METADATA    - clickbait markers, linguistic features, URL-domain
                  reputation. Pure Python, no model.

A fusion layer combines whichever modalities the user supplied and reports
a per-modality breakdown so the final verdict is explainable.

Run locally:
    streamlit run app.py
"""
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

import multimodal as mm

MODEL_PATH = Path(__file__).parent / "models" / "fake_news_model.joblib"
TRANSFORMER_MODEL = "hamzab/roberta-fake-news-classification"

st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("Multimodal Fake News Detector")
st.caption(
    "Text classifier (RoBERTa / TF-IDF) · CLIP image–text consistency · "
    "metadata & source heuristics · late-fusion verdict"
)


# =============================================================================
# Model loaders (cached)
# =============================================================================


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


@st.cache_resource(show_spinner="Loading CLIP — first run downloads ~600 MB...")
def load_clip_bundle():
    return mm._load_clip()


# =============================================================================
# Text predictors → ModalitySignal
# =============================================================================


def _length_based_confidence(text: str, base: float) -> tuple[float, str]:
    """Scale a text-model's confidence by input length.

    Both classifiers were trained on full article bodies. Headline-only
    or very short inputs are out-of-distribution — the model still returns
    a confident-looking 100% score but it shouldn't dominate the fusion.
    Lowering confidence lets metadata & image signals carry more weight.
    """
    n = len(text.split())
    if n < 20:
        return base * 0.1, f"headline-only ({n} words) — heavily downweighted"
    if n < 50:
        return base * 0.45, f"short text ({n} words) — partially downweighted"
    if n < 100:
        return base * 0.75, f"medium text ({n} words) — slightly downweighted"
    return base, f"full article ({n} words) — full confidence"


def text_signal_tfidf(pipe, text: str) -> mm.ModalitySignal:
    proba = pipe.predict_proba([text])[0]
    real_p = float(proba[1])
    label = "REAL" if real_p >= 0.5 else "FAKE"
    conf, note = _length_based_confidence(text, base=0.9)
    return mm.ModalitySignal(
        name="text",
        score=real_p,
        confidence=conf,
        label=f"TF-IDF says {label} ({max(real_p, 1 - real_p):.1%})",
        details={
            "real_probability": round(real_p, 4),
            "model": "tfidf_ensemble",
            "confidence_note": note,
        },
    )


def text_signal_transformer(pipe, text: str) -> mm.ModalitySignal:
    r = pipe(text)[0]
    raw = r["label"].upper()
    is_real = raw in {"TRUE", "REAL", "LABEL_1"}
    conf_raw = float(r["score"])
    real_p = conf_raw if is_real else (1.0 - conf_raw)
    conf, note = _length_based_confidence(text, base=0.95)
    return mm.ModalitySignal(
        name="text",
        score=real_p,
        confidence=conf,
        label=f"RoBERTa says {'REAL' if is_real else 'FAKE'} ({conf_raw:.1%})",
        details={
            "real_probability": round(real_p, 4),
            "model": TRANSFORMER_MODEL,
            "confidence_note": note,
        },
    )


def top_signal_tokens(pipe, text: str, k: int = 6):
    """Interpret TF-IDF pipeline via its LR coefficients."""
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
    names = np.array(vec.get_feature_names_out())
    nz = np.where(contribs != 0)[0]
    if len(nz) == 0:
        return []
    order = nz[np.argsort(np.abs(contribs[nz]))[::-1]][:k]
    return [(names[i], float(contribs[i])) for i in order]


# =============================================================================
# Content helpers (URL fetch + OCR autofill)
# =============================================================================


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
    for tag in soup(["script", "style", "noscript", "header", "footer",
                     "nav", "aside", "form"]):
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
    from PIL import Image
    import pytesseract

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
    return text


# =============================================================================
# UI
# =============================================================================


# --- Session state defaults --------------------------------------------------
for key, default in [
    ("main_text", ""),
    ("article_url", ""),
    ("image_bytes", None),
    ("image_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Streamlit forbids writing to a widget-bound key AFTER the widget is
# instantiated. The URL-fetch and OCR buttons therefore stash their result
# in `_pending_text`, call st.rerun(), and on the next run (before the
# text_area widget is redrawn) we copy it into `main_text`.
if "_pending_text" in st.session_state:
    st.session_state["main_text"] = st.session_state.pop("_pending_text")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.subheader("Model settings")
    model_choice = st.radio(
        "Text model",
        options=["Accurate (RoBERTa)", "Fast (TF-IDF baseline)"],
        help=(
            "RoBERTa is the pre-trained transformer — strong on modern news.\n"
            "TF-IDF is the interpretable baseline I trained from scratch."
        ),
    )

    st.subheader("Modalities to use")
    use_text = st.checkbox("📝 Text classifier", value=True)
    use_image_match = st.checkbox(
        "🖼️ CLIP image–text match",
        value=False,
        help=(
            "Off by default on the free Streamlit Cloud tier because CLIP "
            "is ~600 MB on top of RoBERTa's 500 MB and the combined footprint "
            "can exceed the 1 GB memory limit. Turn on locally (or on a paid "
            "tier) to activate the image-text semantic-match signal."
        ),
    )
    use_image_forensics = st.checkbox("🔍 Image forensics", value=True)
    use_metadata = st.checkbox("📊 Metadata & source", value=True)

    st.caption(
        "Uncheck a modality to see how the verdict changes without it — "
        "useful when demo-ing ablation to an interviewer."
    )

    st.divider()
    st.subheader("Fusion weights")
    st.caption("Rebalance the late-fusion layer.")
    w_text = st.slider("Text", 0.0, 1.0, 0.55, 0.05)
    w_clip = st.slider("Image–text match", 0.0, 1.0, 0.20, 0.05)
    w_forensics = st.slider("Image forensics", 0.0, 1.0, 0.05, 0.05)
    w_metadata = st.slider("Metadata", 0.0, 1.0, 0.20, 0.05)


# --- Main input panel --------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("1. Article text")
    st.caption(
        "Paste directly, or populate from a URL / image on the right."
    )
    st.text_area(
        "Article text",
        height=300,
        placeholder="Paste article text here...",
        key="main_text",
        label_visibility="collapsed",
    )

with right:
    st.subheader("2. Optional: URL")
    url_in = st.text_input(
        "Article URL (used for source-credibility score + fetch)",
        placeholder="https://www.bbc.com/news/...",
        key="article_url",
    )
    if st.button("Fetch article body", use_container_width=True):
        if not url_in.strip():
            st.warning("Enter a URL first.")
        else:
            try:
                with st.spinner("Fetching..."):
                    fetched = fetch_article_text(url_in.strip())
                # Stash, then rerun — the top-of-script handler copies
                # _pending_text into main_text before the widget redraws.
                st.session_state["_pending_text"] = fetched
                st.success(f"Fetched {len(fetched):,} characters.")
                st.rerun()
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    st.subheader("3. Optional: image")
    uploaded = st.file_uploader(
        "Image that accompanies the article",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        st.session_state.image_bytes = uploaded.getvalue()
        st.session_state.image_name = uploaded.name
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        if st.button("Also OCR this image into the text box", use_container_width=True):
            try:
                with st.spinner("Running OCR..."):
                    extracted = ocr_image(st.session_state.image_bytes)
                if len(extracted) < 40:
                    st.warning(
                        "OCR extracted very little text. The image will still "
                        "be analysed for image–text match and forensics."
                    )
                else:
                    st.session_state["_pending_text"] = extracted
                    st.success(f"Extracted {len(extracted):,} characters.")
                    st.rerun()
            except Exception as e:
                st.error(f"OCR failed: {e}")
    elif st.session_state.image_bytes is not None:
        st.image(
            st.session_state.image_bytes,
            caption=st.session_state.image_name or "(previous upload)",
            use_container_width=True,
        )
        if st.button("Clear image"):
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.rerun()


st.divider()
go = st.button("🔎 Analyze (multimodal)", type="primary", use_container_width=True)


# =============================================================================
# Analysis
# =============================================================================


if go:
    current_text = (st.session_state.main_text or "").strip()
    current_url = (st.session_state.article_url or "").strip() or None
    current_image = st.session_state.image_bytes

    if not current_text and current_image is None:
        st.warning(
            "Please provide at least some text (paste, fetch from URL, or OCR "
            "an image) OR an image to analyse."
        )
        st.stop()

    if (current_text.startswith(("http://", "https://"))
            and "\n" not in current_text):
        st.warning(
            "This looks like a bare URL pasted into the text box. Use the "
            "URL field instead — the model classifies article text, not links."
        )
        st.stop()

    # Headline-only warning. The text classifiers were trained on full
    # article bodies, so one-line inputs are unreliable. The fusion layer
    # downweights them automatically, but the user should know.
    n_words = len(current_text.split()) if current_text else 0
    if 0 < n_words < 40:
        st.info(
            f"ℹ️ Short text ({n_words} words). The text classifier was trained "
            "on full articles — for headlines only, its confidence is "
            "automatically reduced and the verdict leans more on metadata / "
            "image signals. For best results, paste the full article body or "
            "fetch it from the URL."
        )

    # --- Run each modality the user selected --------------------------------
    signals = []

    # 1. Text
    using_transformer = model_choice.startswith("Accurate")
    token_signals = []
    if use_text and current_text:
        with st.spinner("Scoring text..."):
            if using_transformer:
                sig = text_signal_transformer(load_transformer(), current_text)
            else:
                pipe = load_tfidf()
                sig = text_signal_tfidf(pipe, current_text)
                token_signals = top_signal_tokens(pipe, current_text)
        signals.append(sig)

    # 2a. CLIP image–text match
    if use_image_match and current_image is not None and current_text:
        with st.spinner("Running CLIP image–text match..."):
            try:
                clip_bundle = load_clip_bundle()
                sig = mm.analyze_image_text_match(
                    current_image, current_text, clip_bundle=clip_bundle
                )
                signals.append(sig)
            except Exception as e:
                st.warning(f"CLIP modality failed, skipping: {e}")

    # 2b. Image forensics
    if use_image_forensics and current_image is not None:
        sig = mm.analyze_image_forensics(current_image)
        signals.append(sig)

    # 3. Metadata
    if use_metadata and current_text:
        sig = mm.analyze_metadata(current_text, url=current_url)
        signals.append(sig)

    if not signals:
        st.warning("No modalities produced a usable signal. Check your inputs.")
        st.stop()

    # --- Fuse ---------------------------------------------------------------
    weights = {
        "text": w_text,
        "image_text_match": w_clip,
        "image_forensics": w_forensics,
        "metadata": w_metadata,
    }
    verdict = mm.fuse_signals(signals, weights=weights)

    # -----------------------------------------------------------------------
    # Result panel  (minimalist: verdict + per-modality table, rest collapsed)
    # -----------------------------------------------------------------------
    st.divider()

    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        if verdict.label == "REAL":
            st.success(f"### {verdict.label}")
        elif verdict.label == "FAKE":
            st.error(f"### {verdict.label}")
        else:
            st.warning(f"### {verdict.label}")
    with v_col2:
        st.metric("Fused REAL-probability", f"{verdict.score:.1%}",
                  help=f"Fusion confidence: {verdict.confidence:.1%}")
        st.progress(verdict.score)

    # Per-modality breakdown — one compact table, no duplicate chart.
    rows = []
    for s in verdict.signals:
        rows.append({
            "Modality": s.name,
            "REAL-score": round(s.score, 3) if s.confidence > 0 else "—",
            "Weight": verdict.weights.get(s.name, 0.0),
            "Verdict": s.label,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Drill-down + methodology hidden behind a single collapsed expander
    # so interviewers who want depth can click, and casual viewers don't
    # have to scroll past a wall of text.
    with st.expander("Show reasoning & methodology"):
        for s in verdict.signals:
            tag = "" if s.confidence > 0 else "  _(not used)_"
            st.markdown(f"**`{s.name}` → {s.label}**{tag}")
            if s.details:
                for k, v in s.details.items():
                    if isinstance(v, list):
                        st.markdown(f"- *{k}:* " + "; ".join(str(x) for x in v))
                    else:
                        st.markdown(f"- *{k}:* {v}")
            if s.name == "text" and token_signals:
                toks = ", ".join(
                    f"`{t}` ({'+' if w > 0 else ''}{w:.2f})" for t, w in token_signals
                )
                st.markdown(f"- *top TF-IDF tokens:* {toks}")
            st.write("")

        st.markdown("---")
        st.markdown(
            "**How fusion works.** Each modality outputs a REAL-score and "
            "confidence; the final score is a confidence-weighted average "
            "using the sidebar weights. Modalities without input drop out and "
            "remaining weights renormalise. Score ≥ 0.6 → REAL, ≤ 0.4 → FAKE, "
            "otherwise UNCERTAIN. This is **late fusion** — each signal is "
            "scored independently and combined, so every contribution is "
            "auditable."
        )
        st.markdown(
            f"**Models.** Text: `{TRANSFORMER_MODEL}` or TF-IDF ensemble "
            "(LogReg + MultinomialNB + ComplementNB), 96% held-out accuracy on "
            "`GonzaloA/fake_news`. Image: CLIP `openai/clip-vit-base-patch32` "
            "for text-image similarity + Pillow heuristics for origin. "
            "Metadata: pure-Python heuristics + 50-domain credibility list."
        )


st.divider()
st.caption(
    "Multimodal fake-news detection · text + image + metadata · "
    "late-fusion · GitHub: atishay01/fake-news-detector"
)
