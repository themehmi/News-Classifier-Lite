import streamlit as st
import joblib

st.set_page_config(page_title="News Classifier Lite", page_icon="📰", layout="centered")

#LOAD MODELS
# Using @st.cache_resource ensures the model loads only once, preventing lag
@st.cache_resource
def load_assets():
    try:
        tfidf = joblib.load('nclite_vec.pkl')
        clf = joblib.load('nclite.pkl')
        return tfidf, clf
    except FileNotFoundError:
        st.error("Error: '.pkl' files not found. Ensure they are in the same directory.")
        return None, None

tfidf, clf = load_assets()

# RESPONSIVE CSS
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

    header, footer {{visibility: hidden;}}

    .github-corner {{
        position: fixed;
        top: 15px;
        right: 15px;
        z-index: 999;
        text-decoration: none;
        color: white;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .block-container {{
        padding-top: 5rem !important;
        max-width: 500px !important;
    }}

    .stTextArea textarea {{
        background-color: #000 !important;
        color: white !important;
        border: 1px solid #1f1f1f !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }}

    .stButton>button {{
        width: 100%;
        background-color: white !important;
        color: black !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.6rem;
    }}

    .cat-tag {{
        display: inline-block;
        font-size: 0.65rem;
        color: #888;
        border: 1px solid #1f1f1f;
        padding: 3px 8px;
        border-radius: 5px;
        margin: 2px;
        text-transform: uppercase;
    }}

    @media (max-width: 480px) {{
        .github-corner span {{ display: none; }}
    }}
    </style>
    
    <a href="https://github.com/themehmi/News-Classifier-Lite" target="_blank" class="github-corner">
        <svg height="18" viewBox="0 0 16 16" width="18" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        <span>Source</span>
    </a>
    """, unsafe_allow_html=True)

# APP UI
st.title("News Classifier Lite")

# Scope Display
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

# Main Form
with st.container():
    news_input = st.text_area("Input Text", placeholder="Paste news snippet here...", height=150, label_visibility="collapsed")
    submit = st.button("Identify Category")

    if submit:
        if news_input.strip() and tfidf and clf:
            try:
                # NLP Transformation
                data = [news_input]
                vect = tfidf.transform(data)
                prediction = clf.predict(vect)[0]
                
                # Mapping - Update these to match your specific model labels
                mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
                result = mapping.get(prediction, str(prediction))
                
                # Result Card
                st.markdown(f"""
                    <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.03); border: 1px solid #1f1f1f; border-radius: 12px;">
                        <p style="color: #888; font-size: 0.7rem; text-transform: uppercase; margin: 0;">Result</p>
                        <h2 style="color: white; margin: 5px 0 0 0;">{result}</h2>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            if not news_input.strip():
                st.warning("Please enter some text.")
            else:
                st.error("Model files are missing. Check your directory.")
