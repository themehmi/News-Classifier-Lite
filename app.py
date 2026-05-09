from flask import Flask, render_template, request
import pickle
import nltk
import re
import string
import os
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# NLTK CONFIG FOR VERCEL
# Tell NLTK to look for data in the current directory's 'nltk_data' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk_data_path = os.path.join(BASE_DIR, 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Initialize Lemmatizer
# We don't download here because it causes timeouts on Vercel
lemmatizer = WordNetLemmatizer()

def cleaner(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# LOAD MODELS SAFELY 
def load_models():
    try:
        # Construct the full path to your files
        tfidf_path = os.path.join(BASE_DIR, 'nclite_vec.pkl')
        clf_path = os.path.join(BASE_DIR, 'nclite.pkl')

        # Check if files exist before opening to avoid cryptic errors
        if not os.path.exists(tfidf_path) or not os.path.exists(clf_path):
            print(f"CRITICAL ERROR: Files not found at {BASE_DIR}")
            return None, None

        # Load using 'rb' (Read Binary)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)

        print("Models loaded successfully!")
        return tfidf, clf

    except Exception as e:
        print(f"Error during loading: {e}")
        return None, None

tfidf, clf = load_models()

@app.route('/', methods=['GET'])
def home():
    categories = ['Graphics', 'Medicine', 'Space', 'Politics']
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    categories = ['Graphics', 'Medicine', 'Space', 'Politics']
    news_input = request.form.get('news_text', '')
    
    if not news_input.strip():
        return render_template('index.html', categories=categories, error="Please enter text.")

    if tfidf is None or clf is None:
        return render_template('index.html', categories=categories, error="Server Error: Models failed to load.")

    try:
        cleaned_text = cleaner(news_input)
        vect = tfidf.transform([cleaned_text])
        prediction = clf.predict(vect)[0]
        
        mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
        result = mapping.get(prediction, str(prediction))
        
        return render_template('index.html', 
                               categories=categories, 
                               prediction=result, 
                               original_text=news_input)
    except Exception as e:
        return render_template('index.html', categories=categories, error=f"Error: {e}")

# Required for Vercel
if __name__ == '__main__':
    app.run(debug=True)
