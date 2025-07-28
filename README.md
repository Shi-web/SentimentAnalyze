# 🤖 Sentiment Analysis Using TF-IDF and a Neural Network

## 📝 Project Overview

This project focuses on **binary sentiment classification** using book reviews. Given the text of a review, the goal is to predict whether the sentiment is **positive** or **negative**. This has been implemented by training a **feedforward neural network** for improved performance.

This work is part of my learning journey through the Break Through Tech AI Program, where I applied machine learning concepts in natural language processing (NLP).

---

## 📚 What the Project Covers

### ✅ 1. Define the ML Problem

* **Problem Type:** Binary classification
* **Input:** Raw text from book reviews
* **Output:** Sentiment label (positive or negative)

### 🧹 2. Data Preprocessing

* Tokenized and cleaned text reviews
* Removed stopwords and punctuation
* Transformed text using `TfidfVectorizer` with:

  * Custom `max_df` and `min_df` thresholds
  * Max features set to 3000

### 🤖 3. Model Development

* **Model Type:** Feedforward Neural Network using `tensorflow.keras`
* **Architecture:**

  * Dense hidden layers with ReLU activation
  * Dropout regularization
  * Final sigmoid layer for binary output
* Trained on the TF-IDF transformed text data
* Used binary crossentropy loss and SGD optimizer

### 💾 4. Model Persistence

* **Saved** both the trained model and the vectorizer using:

  * `model.save()` from TensorFlow
  * `pickle.dump()` for the TF-IDF vectorizer

### 📈 5. Evaluation

* Accuracy, precision, recall, and F1-score calculated on test data
* Plotted the **precision-recall curve** to analyze classifier performance

---

## 🔧 Technologies Used

* **Python**
* **pandas**, **numpy**
* **scikit-learn** (`TfidfVectorizer`, metrics)
* **TensorFlow / Keras** (modeling)
* **matplotlib**, **seaborn** (visualizations)
* **pickle** (persistence)

---

## 🎯 Key Takeaways

* Implemented a **custom neural network** for NLP tasks
* Tuned TF-IDF vectorizer parameters to improve model generalization
* Understood the importance of **precision-recall trade-offs**
* Learned to **persist and reload models and vectorizers** for deployment use cases

---

## 📌 How to Reuse the Model

* Load the saved `TF-IDF vectorizer` from the `.pkl` file
* Load the neural network using `tensorflow.keras.models.load_model()`
* Pass new text data through the vectorizer, then predict with the model

---

## 🤔 Reflection

This project helped me solidify my understanding of:

* NLP text vectorization techniques
* Neural network design and optimization
* Model persistence for real-world ML pipelines
* Evaluation strategies beyond accuracy


