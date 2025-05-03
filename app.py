import streamlit as st
import joblib
import requests
from streamlit_lottie import st_lottie
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load Lottie animation from URL
def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except:
        return None

# Lottie animations
lottie_header = load_lottie_url("https://lottie.host/9c0c1a4f-d6e6-40bc-b661-95ef4eb33659/BAGQ05Edfp.json")
lottie_success = load_lottie_url("https://lottie.host/14d173a8-49ff-4c73-a6b2-96bb5ecb2b01/oD4YI1U1gO.json")
lottie_processing = load_lottie_url("https://lottie.host/e3ff7e2c-5e9d-4a49-a1c1-988d95c6c63f/RJ0V9oGZss.json")
lottie_bg = load_lottie_url("https://lottie.host/dfbeed4b-610e-4c5d-b032-fd89f6d5ac3b/NOCnzXvphg.json")

# Model and vectorizer
model = joblib.load('ensemble_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App configuration
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")

# App styling
st.markdown("""
    <style>
    body {
        background-color: #f2f6ff;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Features
st.sidebar.title("📘 Features")
st.sidebar.toggle("🌙 Dark Mode (Experimental)")

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.markdown("### 🧾 Recent Predictions")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.markdown(f"{item['timestamp']} — {item['text'][:30]}... ➤ {item['label']}")

st.sidebar.markdown("### 🎉 Fun Fact")
st.sidebar.info("Did you know? The word 'tweet' was added to the Oxford English Dictionary in 2013!")

# Main Header
if lottie_header:
    st_lottie(lottie_header, height=250, key="header")
else:
    st.title("🌍 Disaster Tweet Classifier")

st.markdown("### ✨ Try with an example:")
st.markdown("- Just happened a terrible earthquake")
st.markdown("- What a beautiful sunny day")
st.markdown("- Forest fire near LA, evacuating now")

st.markdown("### 🔍 Enter your tweet:")
tweet = st.text_area("✏️ Tweet Text", height=100)

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        with st.spinner("Analyzing tweet..."):
            if lottie_processing:
                st_lottie(lottie_processing, height=150)

            transformed = vectorizer.transform([tweet])
            prediction = model.predict(transformed)
            proba = model.predict_proba(transformed)[0]  # confidence

            result_label = "🚨 Disaster" if prediction[0] == 1 else "✅ Not Disaster"

            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%H:%M"),
                'text': tweet,
                'label': result_label
            })

            if lottie_success:
                st_lottie(lottie_success, height=200, key="result")

            st.success(f"**Prediction:** {result_label}")
            st.info(f"Confidence: Disaster = {proba[1]:.2f}, Not Disaster = {proba[0]:.2f}")

# Footer info
st.markdown("---")
st.markdown("### 👨‍💻 About This Project")
st.markdown("This app was built using **Streamlit** and an **ensemble machine learning model** combining Logistic Regression, Naive Bayes, and SGD. It uses **TF-IDF vectorization** to process tweet text.")

if lottie_bg:
    st_lottie(lottie_bg, height=100, loop=True, speed=0.1, key="bg")
