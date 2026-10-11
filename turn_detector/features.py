"""Feature extraction for a low-latency text-and-timing endpointing model.

This is deliberately a compact, inspectable baseline. Transcript text is assumed
 to be the current streaming ASR hypothesis, not raw audio.
"""
from __future__ import annotations

import re
from typing import Sequence

FEATURE_NAMES: Sequence[str] = (
    "pause_seconds",
    "stable_seconds",
    "word_count_scaled",
    "terminal_punctuation",
    "question_mark",
    "period_or_exclamation",
    "open_punctuation",
    "ends_open_word",
    "ends_pronoun",
    "ends_filler",
    "contains_complete_marker",
    "is_spanish",
)

EN_OPEN = {
    "and", "but", "because", "if", "when", "while", "although", "that",
    "which", "who", "to", "for", "with", "of", "the", "a", "an", "as",
    "than", "so", "or", "unless", "until", "from", "at", "on", "into",
}
ES_OPEN = {
    "y", "pero", "porque", "si", "cuando", "mientras", "aunque", "que",
    "quien", "para", "con", "de", "el", "la", "los", "las", "un", "una",
    "unos", "unas", "como", "o", "hasta", "desde", "en", "por",
}
EN_PRONOUNS = {"i", "we", "you", "they", "he", "she", "it", "this", "that"}
ES_PRONOUNS = {"yo", "nosotros", "nosotras", "tú", "usted", "ustedes", "ellos", "ellas", "esto", "eso"}
EN_FILLERS = {"um", "uh", "erm", "hmm", "well", "like"}
ES_FILLERS = {"eh", "mmm", "este", "pues"}
EN_COMPLETE = {"thanks", "thank", "goodbye", "bye", "done", "today", "tomorrow", "yes", "no", "okay", "ok", "correct", "great"}
ES_COMPLETE = {"gracias", "adiós", "listo", "lista", "hoy", "mañana", "sí", "no", "vale", "perfecto", "perfecta"}


def _normalize_language(language: str) -> str:
    lang = (language or "en").lower().replace("_", "-")
    if lang.startswith("en"):
        return "en"
    if lang.startswith("es"):
        return "es"
    raise ValueError("Supported prototype languages are English (en) and Spanish (es).")


def extract_features(text: str, pause_ms: float, stability_ms: float, language: str = "en") -> list[float]:
    """Return a stable numeric feature vector for a transcript snapshot."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    lang = _normalize_language(language)
    stripped = text.strip()
    lower = stripped.lower()
    words = re.findall(r"[\wáéíóúüñ']+", lower, flags=re.UNICODE)
    last = words[-1] if words else ""
    open_words = EN_OPEN if lang == "en" else ES_OPEN
    pronouns = EN_PRONOUNS if lang == "en" else ES_PRONOUNS
    fillers = EN_FILLERS if lang == "en" else ES_FILLERS
    complete_markers = EN_COMPLETE if lang == "en" else ES_COMPLETE
    terminal = bool(re.search(r"[.!?…][\"')\]]*\s*$", stripped))
    question = "?" in stripped[-4:]
    period_or_exclamation = bool(re.search(r"[.!][\"')\]]*\s*$", stripped))
    open_punctuation = bool(re.search(r"[,;:]|(?:\.\.\.|…|--?)\s*$", stripped))
    # Add phrase-level lexical clues, but do not require a language-specific phrase.
    phrase_markers = (
        ("that's all", "that is all", "all i needed", "thank you", "talk to you later")
        if lang == "en" else
        ("eso es todo", "eso sería todo", "muchas gracias", "hasta luego", "eso era todo")
    )
    has_complete_marker = last in complete_markers or any(p in lower for p in phrase_markers)
    return [
        max(0.0, min(float(pause_ms), 2000.0)) / 1000.0,
        max(0.0, min(float(stability_ms), 1000.0)) / 500.0,
        min(len(words), 20) / 20.0,
        float(terminal),
        float(question),
        float(period_or_exclamation),
        float(open_punctuation),
        float(last in open_words),
        float(last in pronouns),
        float(last in fillers),
        float(has_complete_marker),
        float(lang == "es"),
    ]
