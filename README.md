# 📰 VeriFy — Fake News Detection & Explanation System

VeriFy is an intelligent **Fake News Detection and Explanation system** that combines **deep learning–based text classification** with **LLM-powered explainability**.  
The project aims to not only classify news as *Fake* or *Real*, but also **explain why** a particular decision was made and **collect user feedback** to improve future model training.

---

## 🚀 Project Motivation

With the rapid spread of misinformation across digital platforms, users often struggle to judge the credibility of online news.  
Most existing tools either:
- Only classify news without explanation, or  
- Provide generic AI-generated responses without grounding in model behavior.

**VeriFy bridges this gap** by combining:
- A **trained deep learning model** for classification
- **Explainable AI (XAI)** using Gemini
- **Counterfactual reasoning**
- **Human-in-the-loop feedback collection**

---

## 🧠 How the System Works

### 1️⃣ Text Classification (Core Model)
- News text is preprocessed and converted into sequences using a **Tokenizer**
- The sequences are padded to a fixed length (`maxlen = 1000`)
- A **Word2Vec + LSTM neural network** predicts whether the news is *Fake* or *Real*

### 2️⃣ Explainability (LLM Integration)
- The prediction and original text are passed to **Gemini LLM**
- Gemini generates **human-readable reasoning** explaining *why* the model may have made that prediction
- This improves transparency and user trust

### 3️⃣ Counterfactual Analysis
- The system highlights how **small wording changes** could flip the prediction
- Helps identify **sensational terms, emotional language, or misleading phrases**

### 4️⃣ User Feedback Loop
- Users can confirm whether the prediction was correct
- Feedback is logged into a CSV file
- This data can be reused for **future re-training and dataset expansion**

---

## 🏗️ Model Architecture

- **Embedding Layer**
  - Pre-trained **Word2Vec embeddings** (100 dimensions)
  - Embeddings frozen for stable semantic learning

- **LSTM Layer**
  - Captures long-term dependencies in news articles
  - Effective for modeling narrative flow and context

- **Dense Output Layer**
  - Sigmoid activation for binary classification (Fake / Real)

📌 **Why LSTM + Word2Vec?**
- Handles long news articles better than traditional ML models
- Learns semantic and contextual patterns
- Performs well on sequential text data
- More interpretable than large black-box transformer models for this task

---

## 🧪 Dataset

- Source: Aggregated fake and real news articles
- Preprocessing:
  - Special character removal
  - Tokenization
  - Padding to uniform sequence length
- Labels:
  - `0 → Fake`
  - `1 → Real`

---

## 🖥️ Tech Stack

### Core Technologies
- **Python**
- **TensorFlow / Keras**
- **Gensim (Word2Vec)**
- **Streamlit** (Frontend UI)
- **Google Gemini API** (Explainability)
- **Pandas / NumPy**

### Explainability & Feedback
- Gemini LLM for explanations
- CSV-based feedback logging for retraining

---


---

## 🔗 Model & Tokenizer Download

Due to GitHub size limits, the trained model and tokenizer are hosted on Google Drive.

📥 **Download from here:**  
👉 **[Google Drive Link – Model & Tokenizer](https://drive.google.com/drive/folders/1JwDpVI9xDkLnCoE0yHEjtgSZ6_iOvDpV?usp=drive_link
)**

After downloading:
- Place `my_model.h5` inside `model/`
- Place `tokenizer.pkl` inside `model/`

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/verify-fake-news.git
cd verify-fake-news
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

# Activate on Linux / macOS
```bash
source venv/bin/activate
```

# Activate on Windows
```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Gemini API Key
Create a file named api_key.env in the project root and add:
```bash
GEMINI_API_KEY=your_api_key_here
```

### ▶️ Run the Application

