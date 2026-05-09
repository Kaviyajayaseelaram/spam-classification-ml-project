# ============================================================
#  model/preprocess.py  –  NLP Preprocessing Pipeline
#  Steps: lowercase → remove noise → tokenize → stopwords → stem
# ============================================================

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (runs once, safe to call repeatedly)
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialise stemmer and stopword set once at module level (fast)
_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))

# Words that are actually useful for spam detection — keep them even
# though they appear in the English stopword list
_KEEP_WORDS = {"free", "won", "win", "prize", "urgent", "call", "now", "claim"}


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps
    -----
    1. Lowercase
    2. Remove URLs, email addresses, phone numbers
    3. Remove punctuation and non-alphabetic characters
    4. Tokenise
    5. Remove stopwords (but keep spam-signal words)
    6. Porter stemming
    7. Re-join tokens into a single string

    Parameters
    ----------
    text : str  Raw message text

    Returns
    -------
    str  Cleaned, stemmed text ready for TF-IDF vectorisation
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http/www)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove phone numbers (sequences of 5+ digits)
    text = re.sub(r"\b\d{5,}\b", " ", text)

    # 5. Remove all non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenise
    tokens = word_tokenize(text)

    # 8. Remove stopwords — but keep spam-signal words
    tokens = [
        t for t in tokens
        if t not in _stopwords or t in _KEEP_WORDS
    ]

    # 9. Remove very short tokens (single letters)
    tokens = [t for t in tokens if len(t) > 1]

    # 10. Porter stemming
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
