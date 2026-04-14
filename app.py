import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😋")

# --- Load Model & Setup ---
@st.cache_resource # This keeps the model in memory so it doesn't reload on every click
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

# Download stopwords for the cleaning function
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z]+', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- UI Layout ---
st.title("🍽️ Product Review Sentiment Classifier")
st.write("Enter a product review below to see if it's Positive or Negative.")

user_input = st.text_area("Review Text:", placeholder="e.g., This coffee was amazing and arrived on time!")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        # 1. Preprocess
        cleaned = clean_text(user_input)
        
        # 2. Vectorize
        vectorized_input = vectorizer.transform([cleaned])
        
        # 3. Predict
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input).max()
        
        # 4. Display Results
        if prediction == 1:
            st.success(f"**Positive Sentiment** (Confidence: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"**Negative Sentiment** (Confidence: {probability:.2%})")
    else:
        st.warning("Please enter some text first!")