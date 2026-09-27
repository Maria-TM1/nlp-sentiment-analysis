# NLP: From TF-IDF to Transformers · Sentiment Analysis

**Four-stage NLP study on Spanish and English reviews: classical ML, word embeddings, LSTMs and sentence-transformers.**
Master's in Artificial Intelligence · UDIT · Natural Language Processing and Generation · 2026

**Authors:** María Tocado Murillo · María Fernanda Martínez García

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%C2%B7%20MLP-EE4C2C?logo=pytorch&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-0f8f82)
![Transformers](https://img.shields.io/badge/sentence--transformers-MiniLM-FFD21E)

> 🇪🇸 **Resumen:** estudio en cuatro fases sobre clasificación de sentimiento: modelos clásicos (TF-IDF + regresión logística) sobre reseñas de FilmAffinity, embeddings Word2Vec, LSTM en PyTorch y embeddings contextuales con sentence-transformers. Trabajo realizado en equipo.

---

## Overview

| Stage | Task | Data | Techniques | Key result |
|---|---|---|---|---|
| **1 · Classical ML** | 3-class sentiment (neg / neutral / pos) | 8,601 FilmAffinity reviews (ES) | Snowball stemming · BoW / TF-IDF (1–2 grams) · Logistic Regression · LinearSVC · class weighting · BiLSTM (Keras) reference | **Macro F1 = 0.70** (vs 0.19 trivial baseline) |
| **2 · Word embeddings** | Semantic vector spaces | 5,800-sentence Spanish corpus | Word2Vec CBOW vs Skip-gram · lemmatisation (spaCy) · analogies · PCA / t-SNE | Semantic neighbours and analogies evaluated |
| **3 · Deep learning** | Binary sentiment | 6,367 FilmAffinity reviews (ES) | Custom vocabulary (20k) · Embedding → LSTM → Linear in PyTorch · gradient clipping · ReduceLROnPlateau | **Accuracy = 0.76** · Macro F1 = 0.73 |
| **4 · Transformers** | Binary sentiment + semantic stress test | IMDb (2,000 train / 500 test, EN) | `all-MiniLM-L6-v2` sentence embeddings · Logistic Regression vs MLP (PyTorch) | **Accuracy = 0.80** (LR) · 0.76 (MLP) |

## Highlights

**Stage 1 — problem framing matters as much as the model**
- Compared **3 vs 4 sentiment classes** empirically: 3 classes won by +0.13 Macro F1.
- Diagnosed class imbalance and reported **Macro F1 instead of accuracy** (a model predicting always "neutral" reaches 41% accuracy but only 0.19 Macro F1).
- Title-only vs title + full text: the full review adds **+0.18 Macro F1** (LinearSVC); stemming adds a further +0.03.
- An alternative target built from the film's average rating reached **Macro F1 = 0.91**, showing how label definition drives difficulty.
- A BiLSTM trained on the same data scored Macro F1 = 0.58: with ~8.6k reviews, a well-tuned TF-IDF model beats a neural baseline.

**Stage 3 — building an LSTM from scratch**
- Vocabulary, padding, `Dataset` / `DataLoader`, training loop and evaluation implemented by hand in PyTorch.
- Positive class F1 = 0.81, negative class F1 = 0.65 — error analysis with confusion matrix.

**Stage 4 — contextual embeddings**
- Pre-trained 384-d sentence embeddings + a linear classifier reach 80% accuracy with only 2,000 training examples.
- Adversarial phrases designed to break frequency-based models (negations, irony): LR 7/11 vs MLP 6/11 correct.

## Tech stack

Python · pandas · scikit-learn · NLTK / spaCy · Gensim · PyTorch · TensorFlow/Keras · sentence-transformers · Hugging Face Datasets · matplotlib

## About this repository

This is a **showcase** of coursework carried out as a team. Notebooks, reports and datasets are not published.

---

**María Tocado Murillo** · Applied AI & Data Scientist · [LinkedIn](https://www.linkedin.com/in/mar%C3%ADa-tocado-murillo/) · [GitHub](https://github.com/Maria-TM1)

© 2026 María Tocado Murillo and María Fernanda Martínez García. All rights reserved — see [LICENSE](LICENSE).
