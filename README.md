# SMS Spam Detection Web App

This project is a **Machine Learning–powered SMS Spam Detection web application** built using **Python, NLP, scikit-learn, and Flask**.
It classifies SMS messages as **Spam** or **Not Spam** using text preprocessing and a **Naive Bayes** model.

---

## 🚀 Features

* Text preprocessing (cleaning, tokenization, stopword removal, stemming)
* TF-IDF feature extraction
* Naive Bayes spam classifier
* Flask web interface for real-time predictions
* Simple and lightweight deployment setup

---

## 🗂 Project Structure

```
Mitchell_Project/
│
├── data/
│   └── spam.csv                # Dataset used for training
│
├── templates/
│   └── index.html              # Frontend HTML template
│
├── main.py                     # Flask app + ML logic
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download NLTK resources (run once)

Open Python and run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## ▶️ Running the Application

Start the Flask server:

```bash
python main.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 🧠 Model Overview

* **Algorithm:** Multinomial Naive Bayes
* **Vectorization:** TF-IDF
* **Text Processing:**

  * Lowercasing
  * Removing punctuation
  * Stopword removal
  * Stemming (Lancaster Stemmer)

---

## 🛠 Tech Stack

* Python
* Flask
* scikit-learn
* NLTK
* Pandas & NumPy
* HTML (Jinja templates)

---

## 📌 Future Improvements

* Replace Naive Bayes with BERT or other transformers
* Add user feedback loop for continuous learning
* Improve UI with modern frontend frameworks
* Deploy to Render / Railway / Fly.io

---

## 👤 Author

**Ebosele Isimhemhe Mitchell**
Machine Learning Engineer
📍 Nigeria
