# app.py
import streamlit as st
import string
import joblib

# Load model and vectorizer
model = joblib.load('models/emotion_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_mapping = joblib.load('models/label_mapping.pkl')

# Define stopwords manually to avoid NLTK dependency
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 
    'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once'
}

inverse_mapping = {v: k for k, v in label_mapping.items()}
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if char.isascii()])
    tokens = text.split()
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