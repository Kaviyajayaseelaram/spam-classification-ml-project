# ============================================================
#  model/predict.py  –  Inference Helper
#  Loads the saved pipeline and predicts a single message
# ============================================================

import joblib
from model.preprocess import preprocess_text

MODEL_PATH = "model/model.pkl"
_pipeline  = None  # module-level cache so we load only once


def _load_pipeline():
    """Lazy-load the trained pipeline (loaded once per session)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_message(text: str) -> tuple[str, float, float, float]:
    """
    Classify a single message as SPAM or HAM.

    Parameters
    ----------
    text : str  Raw message text (untouched)

    Returns
    -------
    tuple
        label      : "SPAM" or "HAM"
        confidence : probability of the predicted class (0–1)
        ham_prob   : P(ham)
        spam_prob  : P(spam)
    """
    pipeline = _load_pipeline()

    # Preprocess in the same way as training
    clean = preprocess_text(text)

    # predict_proba returns [[P(ham), P(spam)]]
    proba     = pipeline.predict_proba([clean])[0]
    ham_prob  = float(proba[0])
    spam_prob = float(proba[1])

    label      = "SPAM" if spam_prob >= 0.5 else "HAM"
    confidence = spam_prob if label == "SPAM" else ham_prob

    return label, confidence, ham_prob, spam_prob
