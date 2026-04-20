"""Multimodal signal extractors for the fake-news detector.

Three independent signals, fused late:

    1. TEXT           - existing RoBERTa / TF-IDF classifier (handled in app.py).
    2. IMAGE          - (a) CLIP cosine similarity between image and article text
                        ("does the photo actually match the story?"), and
                        (b) lightweight image forensics (EXIF presence, size,
                        aspect ratio, format) - cheap sanity checks on origin.
    3. METADATA       - heuristic scoring from clickbait markers, linguistic
                        features, and source-domain reputation. Pure Python,
                        no model.

A `fuse_signals()` helper combines the three into a single REAL/FAKE verdict
with a per-modality breakdown so the final decision is explainable -- which is
the whole point of going multimodal instead of piling more layers on the text
model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class ModalitySignal:
    """One modality's contribution to the final verdict.

    `score` is in [0, 1] where 1.0 = strongly REAL and 0.0 = strongly FAKE.
    `confidence` is in [0, 1] -- how much weight fusion should place on this
    signal. A modality that wasn't provided returns confidence=0.
    """
    name: str
    score: float
    confidence: float
    label: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusedVerdict:
    label: str            # "REAL" | "FAKE" | "UNCERTAIN"
    score: float          # fused REAL-probability in [0, 1]
    confidence: float     # total fusion confidence in [0, 1]
    signals: list         # list[ModalitySignal]
    weights: dict         # modality name -> weight actually used

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "weights": self.weights,
            "signals": [s.to_dict() for s in self.signals],
        }


# =============================================================================
# MODALITY 2a: image-text consistency (CLIP)
# =============================================================================

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def _load_clip():
    """Lazy-import so the rest of the app works on machines without torch."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def analyze_image_text_match(
    image_bytes: bytes,
    article_text: str,
    clip_bundle=None,
) -> ModalitySignal:
    """CLIP cosine similarity between the image and the article.

    A genuine news photo usually matches its caption/body semantically. A
    decontextualised image (the classic "same photo reused for different
    stories" fake-news pattern) gets a low similarity score.

    Similarity is mapped to a REAL-score:
        sim >= 0.30  -> image matches article  -> pushes REAL
        sim <= 0.15  -> image is unrelated     -> pushes FAKE
        in between   -> mild / uncertain
    """
    if clip_bundle is None:
        clip_bundle = _load_clip()
    model, processor = clip_bundle

    from PIL import Image
    import torch

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    # CLIP's text encoder is hard-capped at 77 tokens -> truncate to ~300 chars
    trimmed = (article_text or "").strip()[:300]
    if not trimmed:
        return ModalitySignal(
            name="image_text_match",
            score=0.5,
            confidence=0.0,
            label="N/A",
            details={"reason": "No article text provided to compare against."},
        )

    inputs = processor(
        text=[trimmed],
        images=[img],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        out = model(**inputs)
    # cosine sim between image and text embeddings
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sim = float((img_emb @ txt_emb.T).item())

    # Map [-0.1, 0.4] -> [0, 1]
    score = float(np.clip((sim - 0.10) / 0.30, 0.0, 1.0))
    if score >= 0.7:
        label = "Image matches article"
    elif score <= 0.3:
        label = "Image looks unrelated"
    else:
        label = "Weak match"

    return ModalitySignal(
        name="image_text_match",
        score=score,
        confidence=0.8,
        label=label,
        details={"clip_cosine_similarity": round(sim, 4)},
    )


# =============================================================================
# MODALITY 2b: image forensics (no model, cheap sanity checks)
# =============================================================================


def analyze_image_forensics(image_bytes: bytes) -> ModalitySignal:
    """Cheap origin/authenticity heuristics on the image file.

    Not a deepfake detector -- but real wire-service photos tend to carry
    EXIF, sensible dimensions, and JPEG compression; while memes, screenshots,
    and heavily re-shared images usually do not.
    """
    try:
        from PIL import Image, ExifTags  # noqa: F401
    except ImportError:
        return ModalitySignal(
            name="image_forensics",
            score=0.5,
            confidence=0.0,
            label="Pillow missing",
            details={},
        )

    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return ModalitySignal(
            name="image_forensics",
            score=0.3,
            confidence=0.3,
            label="Could not decode image",
            details={"error": str(e)},
        )

    width, height = img.size
    fmt = (img.format or "").upper()
    has_exif = False
    try:
        exif = img.getexif()
        has_exif = bool(exif) and len(exif) > 0
    except Exception:
        has_exif = False

    # Start at neutral and nudge.
    score = 0.5
    reasons = []

    # Very small images are almost always thumbnails / memes.
    if width < 300 or height < 300:
        score -= 0.15
        reasons.append(f"Low resolution {width}x{height}")
    elif width >= 800 and height >= 500:
        score += 0.10
        reasons.append(f"Decent resolution {width}x{height}")

    # Extreme aspect ratios -> screenshot / banner, not a news photo.
    ar = width / max(height, 1)
    if ar > 3 or ar < 0.33:
        score -= 0.10
        reasons.append(f"Unusual aspect ratio {ar:.2f}")

    # Wire photos usually carry EXIF. Screenshots / memes strip it.
    if has_exif:
        score += 0.15
        reasons.append("EXIF metadata present (camera origin likely)")
    else:
        score -= 0.05
        reasons.append("No EXIF metadata (screenshot/edited/shared image)")

    # PNG is the native screenshot format; not disqualifying, but a weak signal.
    if fmt == "PNG":
        score -= 0.05
        reasons.append("PNG format (typical of screenshots)")
    elif fmt in {"JPEG", "JPG"}:
        score += 0.05
        reasons.append("JPEG format (typical of photographs)")

    score = float(np.clip(score, 0.0, 1.0))
    label = "Looks like a photograph" if score >= 0.55 else (
        "Looks shared/edited" if score <= 0.45 else "Inconclusive"
    )

    return ModalitySignal(
        name="image_forensics",
        score=score,
        confidence=0.4,   # heuristic, so we weight it lower than CLIP
        label=label,
        details={
            "width": width,
            "height": height,
            "format": fmt,
            "has_exif": has_exif,
            "reasons": reasons,
        },
    )


# =============================================================================
# MODALITY 3: metadata / linguistic heuristics
# =============================================================================

_CLICKBAIT_WORDS = {
    "shocking", "shocked", "unbelievable", "you won't believe", "wont believe",
    "mind-blowing", "mindblowing", "miracle", "secret", "exposed", "destroyed",
    "slammed", "savage", "insane", "bombshell", "explosive", "stunning",
    "jaw-dropping", "breaks the internet", "gone viral", "must see",
    "what happens next", "doctors hate", "one weird trick",
}

_SENSATIONAL_PUNCT = re.compile(r"[!?]{2,}")


def _load_source_credibility() -> dict:
    """Load the small hand-curated domain reputation list."""
    path = Path(__file__).parent / "data" / "source_credibility.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _score_source(url: Optional[str], creds: dict) -> tuple[Optional[float], str]:
    """Return (score_in_[0,1] or None, human_label) for a URL's domain."""
    if not url:
        return None, "No URL provided"
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return None, "Could not parse URL"
    if not host:
        return None, "Empty host"

    # exact or suffix match
    for domain, tier in creds.items():
        if host == domain or host.endswith("." + domain):
            mapping = {
                "trusted": (0.90, f"{host}: trusted wire / mainstream"),
                "mainstream": (0.75, f"{host}: mainstream outlet"),
                "mixed": (0.50, f"{host}: mixed reliability"),
                "low": (0.20, f"{host}: low-credibility source"),
                "satire": (0.10, f"{host}: satire / parody site"),
            }
            return mapping.get(tier, (0.5, f"{host}: {tier}"))
    return 0.5, f"{host}: unknown source"


def analyze_metadata(text: str, url: Optional[str] = None) -> ModalitySignal:
    """Language + source heuristics. No ML model."""
    txt = (text or "").strip()
    if not txt:
        return ModalitySignal(
            name="metadata",
            score=0.5,
            confidence=0.0,
            label="No text",
            details={},
        )

    # --- clickbait style -----------------------------------------------------
    lower = txt.lower()
    # word-boundary match so "secret" doesn't fire inside "Secretary"; allow
    # multi-word phrases to match as substrings.
    clickbait_hits = 0
    for w in _CLICKBAIT_WORDS:
        if " " in w:
            if w in lower:
                clickbait_hits += 1
        else:
            if re.search(rf"\b{re.escape(w)}\b", lower):
                clickbait_hits += 1
    shout_punct = len(_SENSATIONAL_PUNCT.findall(txt))

    letters = [c for c in txt if c.isalpha()]
    caps_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )

    # --- length / structure --------------------------------------------------
    n_chars = len(txt)
    n_words = len(txt.split())
    sentences = [s for s in re.split(r"[.!?]+", txt) if s.strip()]
    n_sent = len(sentences)
    avg_sent_len = (n_words / n_sent) if n_sent else 0.0

    # --- proper-noun density (cheap NER proxy) ------------------------------
    #   real news has many named entities; opinion/clickbait has few.
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]+\b", txt)
    mid_caps = [
        t for i, t in enumerate(tokens)
        if i > 0 and t[:1].isupper() and not t.isupper()
    ]
    proper_noun_ratio = len(mid_caps) / max(len(tokens), 1)

    # --- source reputation ---------------------------------------------------
    creds = _load_source_credibility()
    source_score, source_label = _score_source(url, creds)

    # --- combine into a credibility score -----------------------------------
    score = 0.55  # slight prior toward REAL (most text users paste is real)
    reasons = []

    if clickbait_hits >= 1:
        score -= 0.10 * min(clickbait_hits, 3)
        reasons.append(f"{clickbait_hits} clickbait phrase(s) found")
    if shout_punct >= 1:
        score -= 0.05 * min(shout_punct, 3)
        reasons.append(f"{shout_punct} group(s) of !!/??")
    if caps_ratio > 0.15:
        score -= 0.10
        reasons.append(f"Excessive capitalisation ({caps_ratio:.0%})")

    if n_words < 60:
        score -= 0.10
        reasons.append(f"Very short article ({n_words} words)")
    elif n_words > 200:
        score += 0.05
        reasons.append(f"Adequate length ({n_words} words)")

    if 12 <= avg_sent_len <= 28:
        score += 0.05
        reasons.append(f"Natural sentence length ({avg_sent_len:.1f} words/sent)")

    if proper_noun_ratio >= 0.08:
        score += 0.10
        reasons.append(f"High named-entity density ({proper_noun_ratio:.1%})")
    elif proper_noun_ratio < 0.02:
        score -= 0.05
        reasons.append("Few named entities")

    if source_score is not None:
        # Blend source reputation in at 40% weight.
        score = 0.6 * score + 0.4 * source_score
        reasons.append(source_label)

    score = float(np.clip(score, 0.0, 1.0))
    label = (
        "Credible style" if score >= 0.6
        else "Clickbait / low-credibility style" if score <= 0.4
        else "Mixed signals"
    )

    return ModalitySignal(
        name="metadata",
        score=score,
        confidence=0.6,
        label=label,
        details={
            "clickbait_hits": clickbait_hits,
            "caps_ratio": round(caps_ratio, 3),
            "word_count": n_words,
            "avg_sentence_len": round(avg_sent_len, 1),
            "proper_noun_ratio": round(proper_noun_ratio, 3),
            "source": source_label,
            "reasons": reasons,
        },
    )


# =============================================================================
# FUSION
# =============================================================================

# Default weights: text still leads (it's the actual classifier), but image
# and metadata meaningfully influence the final call. These sum to 1.0 only
# *after* we drop modalities with confidence=0.
DEFAULT_WEIGHTS = {
    "text":              0.55,
    "image_text_match":  0.20,
    "image_forensics":   0.05,
    "metadata":          0.20,
}


def fuse_signals(
    signals: list,
    weights: Optional[dict] = None,
    uncertain_band: tuple = (0.40, 0.60),
    disagreement_threshold: float = 0.5,
) -> FusedVerdict:
    """Weighted late fusion across whichever modalities are present.

    A signal with confidence=0 is dropped and its weight is redistributed
    proportionally. The fused probability is the confidence-weighted average
    of per-modality REAL-scores.

    **Disagreement override:** if the spread between the highest and lowest
    active modality scores exceeds ``disagreement_threshold`` (default 0.5),
    the verdict is forced to UNCERTAIN regardless of the weighted average.
    This catches the out-of-distribution failure mode where one classifier
    is confidently wrong (e.g. the RoBERTa text model collapsing to 100%
    FAKE on entertainment/box-office news) while another modality correctly
    says otherwise. A well-calibrated multimodal system should admit
    uncertainty here, not rubber-stamp the majority-weighted signal.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)

    active = [s for s in signals if s.confidence > 0]
    if not active:
        return FusedVerdict(
            label="UNCERTAIN", score=0.5, confidence=0.0,
            signals=signals, weights={},
        )

    # effective weight = declared weight * signal confidence
    w_eff = {}
    for s in active:
        base = weights.get(s.name, 0.0)
        w_eff[s.name] = base * s.confidence

    total_w = sum(w_eff.values())
    if total_w == 0:
        return FusedVerdict(
            label="UNCERTAIN", score=0.5, confidence=0.0,
            signals=signals, weights={},
        )

    fused_score = sum(s.score * w_eff[s.name] for s in active) / total_w
    fused_score = float(np.clip(fused_score, 0.0, 1.0))

    # Overall confidence: how concentrated the signal weight is, scaled by
    # the average per-modality confidence.
    avg_conf = float(np.mean([s.confidence for s in active]))
    coverage = total_w / sum(weights.get(s.name, 0.0) for s in active)
    fused_conf = float(np.clip(avg_conf * coverage, 0.0, 1.0))

    normalised_weights = {k: round(v / total_w, 3) for k, v in w_eff.items()}

    # Disagreement override — see docstring.
    scores = [s.score for s in active]
    disagreement = max(scores) - min(scores) if len(scores) >= 2 else 0.0
    if disagreement > disagreement_threshold:
        return FusedVerdict(
            label="UNCERTAIN",
            score=fused_score,
            confidence=fused_conf * 0.5,
            signals=signals,
            weights=normalised_weights,
        )

    lo, hi = uncertain_band
    if fused_score >= hi:
        label = "REAL"
    elif fused_score <= lo:
        label = "FAKE"
    else:
        label = "UNCERTAIN"

    return FusedVerdict(
        label=label,
        score=fused_score,
        confidence=fused_conf,
        signals=signals,
        weights=normalised_weights,
    )
