import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -------------------------
# CUSTOM STYLING (DARK UI)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# NLP SETUP
# -------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# -------------------------
# MODEL LOAD (CACHED)
# -------------------------
import pickle

@st.cache_resource
def load_model():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_model()

# -------------------------
# UI LAYOUT
# -------------------------
st.markdown('<div class="big-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.write("Check whether a news article is real or fake using AI")

user_input = st.text_area("✍️ Paste News Content Here", height=200)

if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        if prediction == 1:
            st.markdown(
                f'<div class="result-box" style="background-color:#ff4b4b;">❌ FAKE NEWS<br>Confidence: {proba[1]*100:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box" style="background-color:#2ecc71;">✅ REAL NEWS<br>Confidence: {proba[0]*100:.2f}%</div>',
                unsafe_allow_html=True
            )
            
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0E1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #333;
    z-index: 100;
}
</style>

<div class="footer">
    🚀 Built by <b>Ayushman B. Pratap Singh</b> | Fake News Detection System
</div>
""", unsafe_allow_html=True)
