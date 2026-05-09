import streamlit as st
import pickle
import nltk
import re
import string
import os
from nltk.stem import WordNetLemmatizer

# PAGE CONFIG
st.set_page_config(page_title="News Classifier Lite", layout="centered")

# ML SETUP & PREPROCESSING
@st.cache_resource
def load_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    
    try:
        with open('nclite_vec.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('nclite.pkl', 'rb') as f:
            clf = pickle.load(f)
        return lemmatizer, tfidf, clf
    except Exception as e:
        return lemmatizer, None, None

lemmatizer, tfidf, clf = load_resources()

def cleaner(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# LOGIC
categories = ['Graphics', 'Medicine', 'Space', 'Politics']
prediction = None
error = None
original_text = ""

# Streamlit "Form" Logic
with st.sidebar:
    st.write("### Backend Controls")
    if tfidf is None:
        st.error("Model files not found!")

# We use st.form to capture the input
with st.container():
    # Placeholder for the exact HTML/CSS UI
    pass

# --- INJECTING THE DITTO HTML/CSS ---
# We use st.markdown with unsafe_allow_html=True to replicate your design
html_code = f"""
<style>
    :root {{
        --bg: #000000;
        --card-bg: #0a0a0a;
        --border: #1f1f1f;
        --text-main: #ffffff;
        --text-dim: #888888;
        --accent: #ffffff;
    }}

    /* Hide Streamlit elements to keep the "Ditto" look */
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), 
                    url('https://images.unsplash.com/photo-1495020689067-958852a7765e?q=80&w=2670&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-container {{
        background-color: var(--card-bg);
        width: 100%;
        max-width: 480px;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
        margin: auto;
        color: var(--text-main);
        font-family: -apple-system, system-ui, sans-serif;
    }}

    h2 {{ margin: 0 0 20px 0; font-size: 1.5rem; font-weight: 800; }}

    .scope-box {{
        border: 1px solid var(--border);
        padding: 15px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        margin-bottom: 25px;
    }}

    .scope-label {{
        color: var(--text-dim);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        display: block;
        margin-bottom: 10px;
    }}

    .cat-tag {{
        display: inline-block;
        font-size: 0.65rem;
        color: var(--text-dim);
        border: 1px solid var(--border);
        padding: 3px 8px;
        border-radius: 5px;
        margin: 2px;
        text-transform: uppercase;
    }}

    .result-box {{
        margin-top: 30px;
        padding-top: 25px;
        border-top: 1px solid var(--border);
    }}

    .result-label {{ color: var(--text-dim); font-size: 0.7rem; text-transform: uppercase; }}
    .result-value {{ font-size: 1.4rem; font-weight: 600; margin-top: 5px; color: #4ade80; }}

    .github-corner {{
        position: fixed;
        top: 20px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        text-decoration: none;
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 0.85rem;
    }}
</style>

<a href="https://github.com/themehmi/News-Classifier-Lite" target="_blank" class="github-corner">
    <span>Source</span>
</a>
"""

st.markdown(html_code, unsafe_allow_html=True)

# RENDER UI
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h2>News Classifier Lite</h2>', unsafe_allow_html=True)
    
    # Scope Tags
    tag_html = "".join([f'<span class="cat-tag">{cat}</span>' for cat in categories])
    st.markdown(f'''
        <div class="scope-box">
            <span class="scope-label">Trained Categories:</span>
            {tag_html}
        </div>
    ''', unsafe_allow_html=True)

    # Input Section
    # Note: Streamlit's text_area is used for functionality, styled via CSS
    news_input = st.text_area("Paste news snippet here...", height=150, label_visibility="collapsed", key="input_text")
    
    if st.button("Identify Category", use_container_width=True):
        if not news_input.strip():
            st.error("Please enter text.")
        elif tfidf is None:
            st.error("Models not loaded correctly.")
        else:
            try:
                cleaned = cleaner(news_input)
                vect = tfidf.transform([cleaned])
                pred_idx = clf.predict(vect)[0]
                
                mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
                result = mapping.get(pred_idx, str(pred_idx))
                
                # Show Result in your HTML Style
                st.markdown(f'''
                    <div class="result-box">
                        <span class="result-label">AI Classification</span>
                        <div class="result-value">{result}</div>
                    </div>
                ''', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
