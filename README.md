# 🛡️ Fake Job Listing Detection System

A Machine Learning-based web application that detects whether a job posting is **Real** or **Fraudulent** using Natural Language Processing (NLP) techniques.

This project was developed as a Mini Project for B.Tech Computer Science Engineering (Data Science).

---

## 📌 Project Overview

Online job platforms are increasingly targeted by fraudulent job postings.  
This system uses text analysis and machine learning to automatically classify job descriptions as:

- ✅ Real Job
- 🚨 Fake Job

The model analyzes the job description content and provides a probability-based prediction.

---

## ⚙️ Technologies Used

- Python
- Streamlit (Web UI)
- Scikit-learn
- TF-IDF Vectorization
- Support Vector Machine (LinearSVC)
- NumPy & SciPy

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Lowercasing text
- Removing HTML tags
- Removing special characters
- Whitespace normalization

### 2️⃣ Feature Engineering
- TF-IDF Vectorization
- Unigrams + Bigrams (`ngram_range=(1,2)`)
- Maximum 8000 features
- Minimum document frequency filtering

### 3️⃣ Model Training
- Algorithm: **Support Vector Machine (LinearSVC)**
- Class imbalance handled using `class_weight='balanced'`
- 80-20 Train-Test split with stratification

### 4️⃣ Data Augmentation
To improve detection of modern scam patterns (crypto scams, security deposits, wallet activation fraud), synthetic fraudulent samples were added to the dataset.

### 5️⃣ Threshold Tuning
A custom fraud detection threshold (55%) was applied to improve sensitivity toward fake job postings.

---

## 📊 Model Performance

- **Accuracy:** ~97.7%
- **Fake Job Recall:** 83%
- **Fake Job Precision:** 76%
- Balanced detection with reduced false negatives

The system prioritizes detecting fraudulent postings to minimize missed scams.

---

## 🌐 Web Application Features

- Clean Streamlit-based UI
- Sidebar navigation (Detection + Model Summary)
- Probability-based prediction
- Adjustable fraud sensitivity
- Animated feedback (GIF indicators)
- Confidence score display

---

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
