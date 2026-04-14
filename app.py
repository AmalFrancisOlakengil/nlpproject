import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Review Analytics Pro", page_icon="📊", layout="wide")

# --- Load Model & Setup ---
@st.cache_resource
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z]+', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Sidebar ---
st.sidebar.title("Settings")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Single Review", "Batch Upload (CSV/Excel/TXT)"])

# --- Main UI ---
st.title("🍽️ Product Review Analytics")

if input_mode == "Single Review":
    user_input = st.text_area("Review Text:", placeholder="Enter a review...")
    if st.button("Analyze"):
        if user_input:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            label = "Positive 😊" if pred == 1 else "Negative 😡"
            st.subheader(f"Result: {label}")
        else:
            st.warning("Please enter text.")

else:
    st.subheader("Upload Batch File")
    uploaded_file = st.file_uploader("Upload CSV, Excel, or TXT", type=['csv', 'xlsx', 'txt'])

    if uploaded_file is not None:
        # Load File
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else: # .txt file
            content = uploaded_file.read().decode("utf-8")
            data = pd.DataFrame({'Text': content.splitlines()})

        st.write("Preview of Uploaded Data:", data.head())

        # Select Column
        column = st.selectbox("Select the column containing the reviews:", data.columns)

        if st.button("Process Batch"):
            with st.spinner('Analyzing...'):
                # 1. Clean & Predict
                data['Cleaned'] = data[column].apply(clean_text)
                vectorized_data = vectorizer.transform(data['Cleaned'])
                data['Sentiment'] = model.predict(vectorized_data)
                data['Label'] = data['Sentiment'].map({1: 'Positive', 0: 'Negative'})

                

                # 2. Visualizations
                col1, col2 = st.columns(2)
                
                counts = data['Label'].value_counts()
                
                with col1:
                    st.write("### Sentiment Distribution")
                    fig, ax = plt.subplots()
                    colors = ['#2ecc71', '#e74c3c'] if counts.index[0] == 'Positive' else ['#e74c3c', '#2ecc71']
                    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                    st.pyplot(fig)

                with col2:
                    st.write("### Data Summary")
                    st.write(f"Total Reviews: {len(data)}")
                    st.write(f"Positive: {counts.get('Positive', 0)}")
                    st.write(f"Negative: {counts.get('Negative', 0)}")
                # --- Inside the 'if st.button("Process Batch"):' block ---

                # ... [Previous code for prediction and pie chart] ...

                st.write("---")
                st.subheader("📊 Top Keyword Frequency")

                def get_top_n_words(corpus, n=10):
                    if corpus.empty:
                        return pd.DataFrame()
                    
                    # Use the vectorizer we loaded from joblib
                    # We don't fit() it; we only transform()
                    matrix = vectorizer.transform(corpus)
                    
                    # Calculate average TF-IDF weight for each word across this batch
                    weights = np.asarray(matrix.mean(axis=0)).ravel()
                    
                    # Create a DataFrame of words and their weights
                    feature_names = vectorizer.get_feature_names_out()
                    word_weights = pd.DataFrame({'Keyword': feature_names, 'Weight': weights})
                    
                    # Sort by weight and take the top N
                    return word_weights.sort_values(by='Weight', ascending=False).head(n)

                # Split data for analysis
                pos_reviews = data[data['Sentiment'] == 1]['Cleaned']
                neg_reviews = data[data['Sentiment'] == 0]['Cleaned']

                col_word1, col_word2 = st.columns(2)

                with col_word1:
                    if not pos_reviews.empty:
                        st.write("🟢 **Top Positive Keywords**")
                        pos_freq = get_top_n_words(pos_reviews)
                        fig, ax = plt.subplots()
                        ax.barh(pos_freq['Keyword'], pos_freq['Weight'], color='#2ecc71')
                        ax.invert_yaxis()  # Highest weight at the top
                        st.pyplot(fig)
                    else:
                        st.write("No positive reviews found.")

                with col_word2:
                    if not neg_reviews.empty:
                        st.write("🔴 **Top Negative Keywords**")
                        neg_freq = get_top_n_words(neg_reviews)
                        fig, ax = plt.subplots()
                        ax.barh(neg_freq['Keyword'], neg_freq['Weight'], color='#e74c3c')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    else:
                        st.write("No negative reviews found.")
                
                # 3. Show Results Table
                st.write("### Full Analysis Result")
                st.dataframe(data[[column, 'Label']])

                # --- Feature: Top Positive/Negative Words ---
                st.write("---")
                st.subheader("💡 Key Sentiment Drivers")

                # Get feature names from the vectorizer
                feature_names = vectorizer.get_feature_names_out()
                coefficients = model.coef_[0]

                # Create a mapping of word -> importance
                word_importance = pd.DataFrame({'word': feature_names, 'importance': coefficients})

                col_a, col_b = st.columns(2)

                with col_a:
                    st.write("Top 'Positive' Words")
                    st.dataframe(word_importance.sort_values(by='importance', ascending=False).head(10))

                with col_b:
                    st.write("Top 'Negative' Words")
                    st.dataframe(word_importance.sort_values(by='importance', ascending=True).head(10))

                # 4. Download Result
                csv = data[[column, 'Label']].to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")