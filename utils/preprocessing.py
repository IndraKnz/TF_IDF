import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


@lru_cache(maxsize=1)
def _resources():
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


def preprocess_text(text):
    """
    Preprocessing Bahasa Indonesia:
    case folding, URL/mention/hashtag removal, number and punctuation removal,
    whitespace normalization, stopword removal, and Sastrawi stemming.
    """
    if text is None:
        return ""

    try:
        text = str(text)
    except Exception:
        return ""

    text = text.strip().lower()
    if not text:
        return ""

    # URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Mention and hashtag marker
    text = re.sub(r"[@#]\w+", " ", text)
    # Numbers
    text = re.sub(r"\d+", " ", text)
    # Keep Indonesian letters and whitespace
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    stemmer, stopword_remover = _resources()

    try:
        text = stopword_remover.remove(text)
    except Exception:
        pass

    try:
        text = stemmer.stem(text)
    except Exception:
        pass

    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_series(series):
    return series.fillna("").map(preprocess_text)
