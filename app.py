import streamlit as st
import joblib
import nltk
import re 
import string  
from nltk.stem import WordNetLemmatizer

# SETUP & ASSETS

@st.cache_resource
def load_assets():
    # Download NLTK data inside the cached function
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    try:
        tfidf = joblib.load('nclite_vec.pkl')
        clf = joblib.load('nclite.pkl')
        return tfidf, clf
    except FileNotFoundError:
        return None, None

tfidf, clf = load_assets()
lemmatizer = WordNetLemmatizer()

def cleaner(text):
    # Lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    # Lemmatize each word
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# UI CONFIG

st.set_page_config(page_title="News Classifier Lite", page_icon="📰", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), 
                    url('https://images.unsplash.com/photo-1495020689067-958852a7765e?q=80&w=2670&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    /* ... rest of your CSS ... */
    </style>
""", unsafe_allow_html=True)

# APP UI

st.title("News Classifier Lite")

categories = ['Graphics', 'Medicine', 'Space', 'Politics']
cat_html = "".join([f'<span class="cat-tag">{cat}</span>' for cat in categories])

st.markdown(f"""
    <div style='border: 1px solid #1f1f1f; padding: 15px; border-radius: 12px; background: rgba(255,255,255,0.02); margin-bottom: 25px; display: flex; flex-direction: column; gap: 10px;'>
        <span style='color: #888; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>
            LIMITED TO 4 CATEGORIES:
        </span>
        <div style='display: flex; flex-wrap: wrap; gap: 6px;'>
            {cat_html}
        </div>
    </div>
""", unsafe_allow_html=True)

with st.container():
    news_input = st.text_area("Input Text", placeholder="Paste news snippet here...", height=150, label_visibility="collapsed")
    submit = st.button("Identify Category")

    if submit:
        if news_input.strip():
            if tfidf and clf:
                try:
                    # CLEAN THE INPUT (Crucial Step!)
                    cleaned_text = cleaner(news_input)
                    
                    # NLP Transformation
                    data = [cleaned_text]
                    vect = tfidf.transform(data)
                    prediction = clf.predict(vect)[0]
                    
                    # Mapping
                    mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
                    result = mapping.get(prediction, str(prediction))
                    
                    st.markdown(f"""
                        <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.03); border: 1px solid #1f1f1f; border-radius: 12px;">
                            <p style="color: #888; font-size: 0.7rem; text-transform: uppercase; margin: 0;">Result</p>
                            <h2 style="color: white; margin: 5px 0 0 0;">{result}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
            else:
                st.error("Model files are missing. Check your directory for 'nclite_vec.pkl' and 'nclite.pkl'.")
        else:
            st.warning("Please enter some text.")
