import streamlit as st
import joblib

# Page Configuration
st.set_page_config(page_title="News Classifier Lite", page_icon="📰", layout="centered")

# Load individual components
# Ensure these files are in the same directory as this script
tfidf = joblib.load('nclite_vec.pkl')
clf = joblib.load('nclite.pkl')

# Custom CSS for the "Black" Aesthetic & Responsive Design
st.markdown(f"""
    <style>
    /* Background Image with Dark Overlay */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), 
                    url('https://images.unsplash.com/photo-1495020689067-958852a7765e?q=80&w=2670&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* GitHub Corner Button */
    .github-corner {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 100;
        text-decoration: none;
        color: white;
        background: rgba(0,0,0,0.6);
        padding: 8px 14px;
        border-radius: 10px;
        border: 1px solid #1f1f1f;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .github-corner:hover {{
        background: white;
        color: black;
    }}

    /* Main Container Styling */
    .main-container {{
        background-color: #0a0a0a;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #1f1f1f;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
        color: white;
    }}

    /* Input Styling */
    .stTextArea textarea {{
        background-color: #000 !important;
        color: white !important;
        border: 1px solid #1f1f1f !important;
        border-radius: 12px !important;
    }}

    /* Button Styling */
    .stButton>button {{
        width: 100%;
        background-color: white !important;
        color: black !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        height: 3rem;
    }}

    /* Category Tags */
    .cat-tag {{
        font-size: 0.7rem;
        color: #888;
        border: 1px solid #1f1f1f;
        padding: 2px 8px;
        border-radius: 5px;
        margin-right: 5px;
        text-transform: uppercase;
    }}
    </style>
    
    <a href="https://github.com/themehmi/News-Classifier-Lite" target="_blank" class="github-corner">
        <svg height="18" viewBox="0 0 16 16" width="18" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        Source
    </a>
    """, unsafe_allow_html=True)

# App UI
st.title("News Classifier Lite")

# Categories Display
categories = ['Graphics', 'Medicine', 'Space', 'Politics']
cat_html = "".join([f'<span class="cat-tag">{cat}</span>' for cat in categories])
st.markdown(f"**Scope:** {cat_html}", unsafe_allow_html=True)

# Prediction Logic
news_input = st.text_area("", placeholder="Start typing news...", height=150)

if st.button("Identify Category"):
    if news_input:
        # Transform and Predict
        text_vectorized = tfidf.transform([news_input])
        prediction_idx = clf.predict(text_vectorized)[0]
        
        # Mapping (Ensure this matches your training)
        mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
        result = mapping.get(prediction_idx, "Unknown")
        
        # Display Result
        st.markdown(f"""
            <div style="margin-top: 25px; padding-top: 25px; border-top: 1px solid #1f1f1f;">
                <span style="color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;">AI Classification</span>
                <div style="font-size: 1.4rem; font-weight: 600; color: white;">{result}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text first.")