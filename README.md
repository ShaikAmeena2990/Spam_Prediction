# Spam Email Detection using Machine Learning and NLP

This repository contains two Jupyter notebooks that explore different machine learning techniques for detecting spam emails. The project involves data preprocessing, text vectorization, and training classification models to accurately predict whether a message is spam or not.

## Files

- `SPAM.ipynb`: Spam classification using TF-IDF vectorization and models like Logistic Regression and XGBoost.
- `SPAM_Prediction.ipynb`: Spam classification using spaCy for text preprocessing and Naive Bayes classification.

## Notebook 1: SPAM.ipynb

Goal: Spam email classification using machine learning models.

Steps:

- Data loading and preprocessing.

- Text vectorization with TfidfVectorizer.

- Dimensionality reduction with PCA.

- Model training: LogisticRegression, XGBoost.

- Evaluation: Accuracy, Classification report.

- Libraries Used: pandas, numpy, sklearn, matplotlib, re, xgboost.

## Notebook 2: SPAM_Prediction.ipynb

Goal: Spam detection with NLP preprocessing using spaCy and text classification with MultinomialNB.

Steps:

- Data loading and label encoding.

- Text preprocessing using spaCy tokenizer.

- Vectorization using CountVectorizer.

- Model training: Naive Bayes (MultinomialNB) inside a Pipeline.

- Evaluation: Classification report.

Libraries Used: pandas, numpy, re, spaCy, sklearn.

## Dataset

The dataset used is a labeled dataset of spam and ham (non-spam) messages, loaded from `spam.csv`. It includes:
- `Category`: Label (ham or spam)
- `Message`: The email/message content

## Project Workflow

### 1. Data Preprocessing
- Cleaning missing values
- Label encoding (`spam` → 1, `ham` → 0)
- Text normalization

### 2. Feature Extraction
- `SPAM.ipynb`: Uses **TF-IDF Vectorizer**.
- `SPAM_Prediction.ipynb`: Uses **spaCy Tokenizer** and **Count Vectorizer**.

### 3. Model Training
- Logistic Regression (sklearn)
- XGBoost Classifier
- Naive Bayes (MultinomialNB via sklearn Pipeline)

### 4. Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)

##  Requirements

Install required Python packages:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib spacy
python -m spacy download en_core_web_sm
