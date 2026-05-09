 Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Naive%20Bayes-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **NLP-powered classifier** that detects spam messages with 97% accuracy using TF-IDF vectorisation and Multinomial Naive Bayes — deployed as an interactive Streamlit web application.

 Features

| Feature | Description |
|---|---|
| 🔍 Live Classifier | Type any message and get instant spam/ham prediction |
| 📊 Model Metrics | Accuracy, Precision, Recall, F1, confusion matrix |
| ☁️ Word Cloud | Visual of top spam vs ham words |
| 📁 Batch Predict | Upload CSV → download predictions |


 Tech Stack

- **Language:** Python 3.10+
- **NLP:** NLTK (tokenisation, stopwords, Porter stemming)
- **ML:** Scikit-learn — TF-IDF Vectoriser + Multinomial Naive Bayes
- **UI:** Streamlit
- **Visualisation:** Matplotlib, WordCloud
- **Dataset:** [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)


Model Performance

| Metric | Score |
|---|---|
| Accuracy | 97.3% |
| Precision | 98.1% |
| Recall | 94.6% |
| F1-Score | 96.3% |
| CV F1 (5-fold) | 96.1% ± 0.8% |

NLP Pipeline
Raw text
  └── lowercase
  └── remove URLs, emails, phone numbers
  └── remove non-alphabetic characters
  └── tokenise (NLTK word_tokenize)
  └── remove stopwords (keep spam-signal words: free, win, urgent...)
  └── Porter stemming (running → run, prizes → prize)
  └── TF-IDF vectorisation (unigrams + bigrams, top 10,000 features)
  └── Multinomial Naive Bayes classifier

 Future Enhancements

- [ ] Add deep learning model (LSTM / DistilBERT) and compare performance
- [ ] Multi-language spam detection
- [ ] Gmail API integration for automated inbox filtering
- [ ] Browser extension for real-time email scanning
- [ ] Active learning loop — improve from user corrections

