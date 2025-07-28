# app.py
import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_mapping = joblib.load('label_mapping.pkl')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

inverse_mapping = {v: k for k, v in label_mapping.items()}
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if char.isascii()])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🧠 Emotion Classifier from Text")

user_input = st.text_area("Enter a sentence to analyze emotion:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        st.success(f"'{user_input}' ➜ {inverse_mapping[pred]}")