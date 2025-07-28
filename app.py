import streamlit as st
import string
import joblib
import re
from typing import List, Dict
import numpy as np

# Cache model loading for better performance
@st.cache_resource
def load_models():
    """Load models with caching for better performance"""
    try:
        model = joblib.load('models/emotion_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        label_mapping = joblib.load('models/label_mapping.pkl')
        return model, vectorizer, label_mapping
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

# Enhanced stopwords with contractions and common words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 
    'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now'
}

# Your specific emotions with emojis
EMOTION_EMOJIS = {
    'joy': '😊',
    'fear': '😨',
    'anger': '😠',
    'sadness': '😢',
    'surprise': '😲',
    'love': '❤️'
}

# Color mapping for emotions
EMOTION_COLORS = {
    'joy': '#FFD700',      # Gold
    'fear': '#8A2BE2',     # Blue Violet
    'anger': '#FF4500',    # Red Orange
    'sadness': '#4169E1',  # Royal Blue
    'surprise': '#FF69B4', # Hot Pink
    'love': '#DC143C'      # Crimson
}

def expand_contractions(text: str) -> str:
    """Expand common contractions"""
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "mustn't": "must not",
        "mightn't": "might not",
        "needn't": "need not"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text: str) -> str:
    """Enhanced preprocessing function"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Handle repeated characters (e.g., "sooooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep emoticons
    text = re.sub(r'[^\w\s:;)(]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove non-ASCII characters
    text = ''.join([char for char in text if char.isascii()])
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word and word not in STOP_WORDS and len(word) > 1]
    
    return ' '.join(tokens)

def get_prediction_confidence(model, vectorized_text) -> Dict:
    """Get prediction probabilities if available"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized_text)[0]
            prediction = model.predict(vectorized_text)[0]
            confidence = max(probabilities)
            return {
                'prediction': prediction,
                'confidence': confidence,
                'all_probabilities': probabilities
            }
        else:
            prediction = model.predict(vectorized_text)[0]
            return {
                'prediction': prediction,
                'confidence': None,
                'all_probabilities': None
            }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def display_results(result: Dict, inverse_mapping: Dict, original_text: str):
    """Enhanced result display with confidence scores for 6 emotions"""
    if not result:
        return
    
    prediction = result['prediction']
    emotion = inverse_mapping.get(prediction, 'Unknown')
    emoji = EMOTION_EMOJIS.get(emotion.lower(), '🤔')
    color = EMOTION_COLORS.get(emotion.lower(), '#000000')
    
    # Main result with colored background
    st.markdown(f"""
    <div style='padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color}'>
        <h2 style='color: {color}; text-align: center; margin: 0;'>
            {emoji} Detected Emotion: {emotion.title()} {emoji}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show confidence if available
    if result['confidence'] is not None:
        confidence_percent = result['confidence'] * 100
        
        # Progress bar for confidence
        st.markdown("### Confidence Level")
        progress_color = '#4CAF50' if confidence_percent >= 70 else '#FF9800' if confidence_percent >= 50 else '#F44336'
        st.markdown(f"""
        <div style='background-color: #f0f0f0; border-radius: 10px; padding: 5px; margin: 10px 0;'>
            <div style='background-color: {progress_color}; width: {confidence_percent}%; height: 20px; border-radius: 5px; display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-weight: bold;'>{confidence_percent:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence interpretation
        if confidence_percent >= 70:
            st.success("🎯 High confidence - Very reliable prediction!")
        elif confidence_percent >= 50:
            st.warning("⚖️ Medium confidence - Fairly reliable prediction")
        else:
            st.error("⚠️ Low confidence - Consider adding more context")
    
    # Show all 6 emotion probabilities
    if result['all_probabilities'] is not None:
        st.markdown("### Emotion Breakdown")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        prob_dict = {}
        for idx, prob in enumerate(result['all_probabilities']):
            emotion_name = inverse_mapping.get(idx, f'Class_{idx}')
            prob_dict[emotion_name] = prob * 100
        
        # Sort by probability
        sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion_name, prob) in enumerate(sorted_emotions):
            emoji = EMOTION_EMOJIS.get(emotion_name.lower(), '🤔')
            color = EMOTION_COLORS.get(emotion_name.lower(), '#000000')
            
            # Alternate between columns
            current_col = col1 if i % 2 == 0 else col2
            
            with current_col:
                # Create a mini progress bar for each emotion
                st.markdown(f"""
                <div style='margin: 5px 0; padding: 10px; border-radius: 8px; background-color: {color}10; border-left: 4px solid {color}'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-weight: bold;'>{emoji} {emotion_name.title()}</span>
                        <span style='color: {color}; font-weight: bold;'>{prob:.1f}%</span>
                    </div>
                    <div style='background-color: #f0f0f0; border-radius: 3px; height: 6px; margin-top: 5px;'>
                        <div style='background-color: {color}; width: {prob}%; height: 100%; border-radius: 3px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Load models
model, vectorizer, label_mapping = load_models()

if model is None or vectorizer is None or label_mapping is None:
    st.error("Unable to load models. Please ensure model files exist in the 'models/' directory.")
    st.stop()

inverse_mapping = {v: k for k, v in label_mapping.items()}

# Streamlit UI
st.set_page_config(
    page_title="Emotion Classifier", 
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Advanced Emotion Classifier")
st.markdown("*Analyze emotions from text with confidence scores and detailed insights*")

# Input section
st.subheader("Enter Text to Analyze")
user_input = st.text_area(
    "Type or paste your text here:",
    placeholder="Example: I'm feeling really excited about my new job!",
    height=100
)

# Analysis section
col1, col2 = st.columns([1, 1])

with col1:
    analyze_button = st.button("🔍 Analyze Emotion", type="primary")

with col2:
    clear_button = st.button("🗑️ Clear Text")

if clear_button:
    st.rerun()

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing emotion..."):
            # Preprocess text
            cleaned_text = preprocess_text(user_input)
            
            if not cleaned_text.strip():
                st.warning("⚠️ No meaningful content found after preprocessing. Try adding more descriptive words.")
            else:
                # Show preprocessing result
                with st.expander("🔧 Preprocessing Details"):
                    st.write(f"**Original:** {user_input}")
                    st.write(f"**Processed:** {cleaned_text}")
                
                # Vectorize and predict
                try:
                    vectorized_text = vectorizer.transform([cleaned_text])
                    result = get_prediction_confidence(model, vectorized_text)
                    
                    if result:
                        display_results(result, inverse_mapping, user_input)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {e}")

# Sidebar with tips
with st.sidebar:
    st.header("💡 Tips for Better Results")
    st.markdown("""
    - **Express feelings clearly:** "I feel..." statements work best
    - **Use descriptive words:** Happy, terrified, furious, heartbroken
    - **Provide context:** Why do you feel this way?
    - **Longer texts:** 10+ words give better accuracy
    - **Examples:**
      - Joy: "I'm so excited about my vacation!"
      - Fear: "I'm really nervous about the exam"
      - Anger: "This traffic is making me furious"
      - Sadness: "I miss my old friends so much"
      - Surprise: "Wow, I can't believe this happened!"
      - Love: "I absolutely adore spending time with you"
    """)
    
    st.header("📊 Your 6 Emotions")
    emotions_display = [
        ('😊 Joy', 'Happiness, excitement, contentment'),
        ('😨 Fear', 'Anxiety, worry, nervousness'),
        ('😠 Anger', 'Frustration, rage, irritation'),
        ('😢 Sadness', 'Sorrow, grief, melancholy'),
        ('😲 Surprise', 'Amazement, shock, wonder'),
        ('❤️ Love', 'Affection, care, romance')
    ]
    
    for emotion, description in emotions_display:
        st.markdown(f"**{emotion}** - {description}")