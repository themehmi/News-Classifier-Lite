import streamlit as st
import joblib
import re
import string
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# --- PAGE CONFIG ---
st.set_page_config(page_title="News Classifier Lite", layout="centered")

# --- NLTK SETUP (Vercel-Friendly) ---
nltk_path = '/tmp/nltk_data'
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)

if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)

@st.cache_resource
def download_resources():
    try:
        nltk.download('wordnet', download_dir=nltk_path)
        nltk.download('punkt', download_dir=nltk_path)
        nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_path)
        nltk.download('universal_tagset', download_dir=nltk_path)
        nltk.download('omw-1.4', download_dir=nltk_path)
        nltk.download('stopwords', download_dir=nltk_path)
        nltk.download('punkt_tab', download_dir=nltk_path)
    except Exception as e:
        st.error(f"NLTK Download Error: {e}")

download_resources()
lemmatizer = WordNetLemmatizer()

# --- CSS INJECTION (1.2x Scaling & Dark Theme) ---
st.markdown(f"""
    <style>
        header, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; display: none !important; }}
        .block-container {{ padding-top: 2rem !important; }}
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.9)), 
                        url('https://images.unsplash.com/photo-1495020689067-958852a7765e?q=80&w=2670');
            background-size: cover;
        }}
        .main-container {{
            background-color: #0a0a0a;
            max-width: 500px; margin: auto; padding: 30px;
            border-radius: 20px; border: 1px solid #1f1f1f;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
            color: #ffffff; font-family: sans-serif;
        }}
        h2 {{ font-size: 1.8rem !important; font-weight: 800; margin-bottom: 20px; }}
        .cat-tag {{
            display: inline-block; font-size: 0.72rem !important; color: #888;
            border: 1px solid #1f1f1f; padding: 4px 10px;
            border-radius: 5px; margin: 2px; text-transform: uppercase;
        }}
        .result-box {{ margin-top: 30px; padding-top: 25px; border-top: 1px solid #1f1f1f; }}
        .res-label {{ color: #888; font-size: 0.84rem !important; text-transform: uppercase; font-weight: 700; }}
        .res-val {{ font-size: 1.68rem !important; font-weight: 600; margin-top: 5px; }}
        textarea {{ font-size: 1.2rem !important; color: white !important; }}
        button[kind="primary"] {{ 
            width: 100%; background: white !important; color: black !important; 
            font-weight: 700 !important; padding: 15px !important; border-radius: 12px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- ADVANCED LOGIC ---
def advanced_cleaner(text):
    # Regex cleaning
    text = re.sub(r'\S*@\S*\s?', '', text) # Remove emails
    text = re.sub(r'\d+', '', text)          # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # POS Tagging for precision
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens, tagset='universal')
    
    # Keep only high-value words (Nouns, Adjectives, Verbs)
    cleaned = [lemmatizer.lemmatize(w) for w, tag in tagged 
               if len(w) > 2 and tag in ['NOUN', 'ADJ', 'VERB']]
    return " ".join(cleaned)

@st.cache_resource
def load_assets():
    try:
        tfidf = joblib.load('nclite_vec.pkl')
        selector = joblib.load('nclite_selector.pkl')
        model = joblib.load('nclite.pkl')
        return tfidf, selector, model
    except Exception as e:
        st.error(f"Asset Load Error: {e}")
        return None, None, None

tfidf, selector, model = load_assets()

# 9 Categories (Ordered alphabetically as trained)
mapping = {
    0: 'Graphics', 1: 'Crypt', 2: 'Electronics', 
    3: 'Medicine', 4: 'Space', 5: 'Guns', 6: 'Mideast'
}

# --- UI RENDER ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h2>News Classifier Lite</h2>', unsafe_allow_html=True)
    
    # Render tags
    tags_html = "".join([f'<span class="cat-tag">{v}</span>' for v in mapping.values()])
    st.markdown(f'<div style="margin-bottom:25px;">{tags_html}</div>', unsafe_allow_html=True)

    # User Input
    user_input = st.text_area("", placeholder="Paste news snippet here...", height=150, label_visibility="collapsed")
    
    if st.button("Analyze News", type="primary"):
        if not user_input.strip():
            st.warning("Please enter text first.")
        elif model is None:
            st.error("Model assets missing from root directory.")
        else:
            # 1. Preprocess
            clean_text = advanced_cleaner(user_input)
            
            # 2. Transform (Vectorize -> Select)
            vec = tfidf.transform([clean_text])
            vec_selected = selector.transform(vec)
            
            # 3. Predict Probabilities
            probs = model.predict_proba(vec_selected)[0]
            top_indices = probs.argsort()[-2:][::-1]
            
            # 4. Extract Top 2
            p1_label, p1_conf = mapping[top_indices[0]], probs[top_indices[0]]
            p2_label, p2_conf = mapping[top_indices[1]], probs[top_indices[1]]
            
            # 5. Display Results
            st.markdown(f"""
               <div class="result-box">
                   <span class="res-label">Primary Match</span>
                   <div class="res-val" style="color: #4ade80;">{p1_label}</div>
                   <span class="res-label">Secondary Guess ({p2_conf:.1%})</span>
                   <div class="res-val" style="color: #888; font-size: 1.2rem !important;">{p2_label}</div>
                </div>
""", unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)