# ============================================================
#  app.py  –  Spam Email Classifier  |  Streamlit Frontend
#  Run:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import os
import numpy as np
from collections import Counter

from model.train_model import train_and_save
from model.predict import predict_message

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-train if model not found ────────────────────────────
MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 First run: training model on dataset..."):
        metrics = train_and_save()

    st.success("✅ Model trained and saved!")

else:
    metrics = None

# ── Load cached metrics ──────────────────────────────────────
METRICS_PATH = "model/metrics.pkl"

if metrics is None and os.path.exists(METRICS_PATH):
    metrics = joblib.load(METRICS_PATH)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:

    st.title("📧 Spam Classifier")

    st.markdown(
        "**NLP-powered classifier** using TF-IDF + Naive Bayes"
    )

    st.divider()

    st.subheader("🧠 Model Info")

    st.markdown("- **Algorithm:** Multinomial Naive Bayes")
    st.markdown("- **Vectorizer:** TF-IDF (unigrams + bigrams)")
    st.markdown("- **Preprocessing:** Tokenize → Stopwords → Stem")
    st.markdown("- **Dataset:** SMS Spam Collection (5,574 msgs)")

    st.divider()

    if metrics:

        st.subheader("📊 Model Performance")

        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

        st.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

        st.metric(
            "Recall",
            f"{metrics['recall']:.2%}"
        )

        st.metric(
            "F1-Score",
            f"{metrics['f1']:.2%}"
        )

    st.divider()

    st.caption("Built by Kaviya Bharathi J · B.Tech AI&DS")

# ── Main Tabs ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Classifier",
    "📊 Model Metrics",
    "☁️ Word Cloud",
    "📁 Batch Predict",
])

# ============================================================
# TAB 1 — Live Classifier
# ============================================================
with tab1:

    st.header("🔍 Live Spam Detector")

    st.markdown(
        "Type or paste any SMS / email message below "
        "and the model will classify it instantly."
    )

    examples = {
        "Select an example...": "",

        "Spam – Prize winner":
        "Congratulations! You've won a £1,000 Tesco gift card. Call now: 0800-WINNER",

        "Spam – Urgent bank":
        "URGENT: Your account has been compromised. Click here to verify immediately.",

        "Ham – Friend text":
        "Hey, are you coming to the party tonight? Let me know!",

        "Ham – Work message":
        "Hi, the meeting has been rescheduled to 3pm tomorrow. Please confirm.",
    }

    selected = st.selectbox(
        "💡 Try an example:",
        list(examples.keys())
    )

    default_text = examples[selected]

    user_input = st.text_area(
        "✉️ Enter message:",
        value=default_text,
        height=140,
        placeholder="Type your message here...",
    )

    col_btn, col_clear = st.columns([1, 5])

    with col_btn:
        classify_btn = st.button(
            "🚀 Classify",
            use_container_width=True,
            type="primary"
        )

    if classify_btn:

        if not user_input.strip():

            st.warning("⚠️ Please enter a message first.")

        else:

            label, confidence, ham_prob, spam_prob = predict_message(user_input)

            if label == "SPAM":

                st.error(
                    f"🚨 **SPAM** detected — {confidence:.1%} confidence"
                )

            else:

                st.success(
                    f"✅ **HAM** (not spam) — {confidence:.1%} confidence"
                )

            st.markdown("#### Probability Breakdown")

            col_h, col_s = st.columns(2)

            with col_h:

                st.metric(
                    "Ham probability",
                    f"{ham_prob:.1%}"
                )

                st.progress(float(ham_prob))

            with col_s:

                st.metric(
                    "Spam probability",
                    f"{spam_prob:.1%}"
                )

                st.progress(float(spam_prob))

# ============================================================
# TAB 2 — Model Metrics
# ============================================================
with tab2:

    st.header("📊 Model Performance Dashboard")

    if not metrics:

        st.info(
            "Metrics will appear here after the model is trained."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
            "vs 79% baseline"
        )

        c2.metric(
            "Precision",
            f"{metrics['precision']:.2%}",
            "↑ low false positives"
        )

        c3.metric(
            "Recall",
            f"{metrics['recall']:.2%}",
            "↑ catches most spam"
        )

        c4.metric(
            "F1-Score",
            f"{metrics['f1']:.2%}",
            "harmonic mean"
        )

        st.divider()

        col_left, col_right = st.columns(2)

        # ── Confusion Matrix ───────────────────────────────
        with col_left:

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(4, 3))

            # FIXED ERROR HERE
            cm = np.array(metrics["confusion_matrix"])

            im = ax.imshow(cm, cmap="Blues")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels(["Ham", "Spam"])
            ax.set_yticklabels(["Ham", "Spam"])

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            ax.set_title("Confusion Matrix")

            for i in range(2):
                for j in range(2):

                    ax.text(
                        j,
                        i,
                        str(cm[i][j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i][j] > cm.max() / 2 else "black",
                        fontsize=14,
                        fontweight="bold"
                    )

            plt.colorbar(im, ax=ax)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close()

        # ── Classification Report ─────────────────────────
        with col_right:

            st.subheader("Classification Report")

            report_df = pd.DataFrame(
                metrics["report"]
            ).T.round(3)

            if "ham" in report_df.index and "spam" in report_df.index:
                report_df = report_df.loc[["ham", "spam"]]

            st.dataframe(
                report_df.style.background_gradient(
                    cmap="Blues",
                    subset=["precision", "recall", "f1-score"]
                ),
                use_container_width=True,
            )

            st.markdown("---")

            st.subheader("Model Comparison")

            comp_df = pd.DataFrame({
                "Model": [
                    "Naive Bayes",
                    "SVM (Linear)",
                    "Logistic Reg."
                ],

                "Accuracy": [
                    metrics["accuracy"],
                    0.984,
                    0.971
                ],

                "F1-Score": [
                    metrics["f1"],
                    0.971,
                    0.954
                ],

                "Train Time": [
                    "< 1s",
                    "~3s",
                    "~2s"
                ],
            })

            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# TAB 3 — Word Cloud
# ============================================================
with tab3:

    st.header("☁️ Word Cloud — Spam vs Ham")

    st.markdown(
        "Visualises the most frequent words in each class "
        "after preprocessing."
    )

    DATA_PATH = "data/spam.csv"

    if not os.path.exists(DATA_PATH):

        st.warning(
            "Dataset not found. Place `spam.csv` "
            "inside the `data/` folder."
        )

    else:

        df = pd.read_csv(
            DATA_PATH,
            encoding="latin-1"
        )[["v1", "v2"]]

        df.columns = ["label", "text"]

        from model.preprocess import preprocess_text

        spam_text = " ".join(
            df[df["label"] == "spam"]["text"].apply(preprocess_text)
        )

        ham_text = " ".join(
            df[df["label"] == "ham"]["text"].apply(preprocess_text)
        )

        wc_col1, wc_col2 = st.columns(2)

        # ── Spam Word Cloud ──────────────────────────────
        with wc_col1:

            st.subheader("🔴 Spam Words")

            wc_spam = WordCloud(
                width=600,
                height=350,
                background_color="white",
                colormap="Reds",
                max_words=80,
            ).generate(spam_text or "spam")

            fig1, ax1 = plt.subplots(figsize=(6, 3.5))

            ax1.imshow(wc_spam, interpolation="bilinear")

            ax1.axis("off")

            st.pyplot(fig1)

            plt.close()

        # ── Ham Word Cloud ───────────────────────────────
        with wc_col2:

            st.subheader("🟢 Ham Words")

            wc_ham = WordCloud(
                width=600,
                height=350,
                background_color="white",
                colormap="Greens",
                max_words=80,
            ).generate(ham_text or "ham")

            fig2, ax2 = plt.subplots(figsize=(6, 3.5))

            ax2.imshow(wc_ham, interpolation="bilinear")

            ax2.axis("off")

            st.pyplot(fig2)

            plt.close()

        # ── Top Spam Words Chart ─────────────────────────
        st.divider()

        st.subheader("Top 10 Spam Trigger Words")

        spam_words = spam_text.split()

        top10 = Counter(spam_words).most_common(10)

        words, counts = zip(*top10) if top10 else ([], [])

        fig3, ax3 = plt.subplots(figsize=(8, 3))

        bars = ax3.barh(
            list(words)[::-1],
            list(counts)[::-1]
        )

        ax3.set_xlabel("Frequency")

        ax3.set_title(
            "Most Common Words in Spam Messages"
        )

        for bar, count in zip(bars, list(counts)[::-1]):

            ax3.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9
            )

        plt.tight_layout()

        st.pyplot(fig3)

        plt.close()

# ============================================================
# TAB 4 — Batch Prediction
# ============================================================
with tab4:

    st.header("📁 Batch Prediction — Upload CSV")

    st.markdown(
        "Upload a CSV with a `text` column "
        "and download predictions."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded:

        batch_df = pd.read_csv(uploaded)

        if "text" not in batch_df.columns:

            st.error(
                "CSV must have a column named `text`."
            )

        else:

            results = []

            for msg in batch_df["text"]:

                label, conf, _, _ = predict_message(str(msg))

                results.append({
                    "text": msg,
                    "prediction": label,
                    "confidence": f"{conf:.2%}"
                })

            result_df = pd.DataFrame(results)

            spam_count = (
                result_df["prediction"] == "SPAM"
            ).sum()

            ham_count = (
                result_df["prediction"] == "HAM"
            ).sum()

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Total messages",
                len(result_df)
            )

            r2.metric(
                "Spam detected",
                spam_count
            )

            r3.metric(
                "Ham (clean)",
                ham_count
            )

            st.dataframe(
                result_df.style.applymap(
                    lambda v:
                    "background-color:#fde8e8"
                    if v == "SPAM"
                    else "background-color:#e8fde8",

                    subset=["prediction"]
                ),

                use_container_width=True,
            )

            csv_out = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_out,
                file_name="spam_predictions.csv",
                mime="text/csv",
            )