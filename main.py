# main.py
import re  
import string 
import nltk
""" nltk.download('stopwords')
nltk.download('punkt') """
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words('english'))
stemmer = LancasterStemmer()

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load the dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df.columns = ["label", "Message"]

# Clean the text data
df["CleanMessage"] = df["Message"].apply(cleaning_data)

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["CleanMessage"])
y = df["label"].map({'ham': 0, 'spam': 1})
model = MultinomialNB()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sms_input = data['smsInput']
    
    # Preprocess the input data
    cleaned_input = cleaning_data(sms_input)
    
    # Transform input using the same vectorizer used for training
    input_transformed = vectorizer.transform([cleaned_input])
    
    # Make predictions
    prediction = model.predict(input_transformed)[0]
    
    # Return result
    if prediction == 0:
        result = "Safe text (not spam)"
    else:
        result = "Spam text!!!"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
