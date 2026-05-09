from flask import Flask, render_template, request
import joblib
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# PREPROCESSING SETUP
# Initialize NLTK components
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def cleaner(text):
    """Processes raw text identically to your training logic."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# LOAD MODELS
try:
    tfidf = joblib.load('nclite_vec.pkl')
    clf = joblib.load('nclite.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    tfidf, clf = None, None

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

    try:
        # Preprocess
        cleaned_text = cleaner(news_input)
        # Transform and Predict
        vect = tfidf.transform([cleaned_text])
        prediction = clf.predict(vect)[0]
        
        # Mapping
        mapping = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}
        result = mapping.get(prediction, str(prediction))
        
        return render_template('index.html', 
                               categories=categories, 
                               prediction=result, 
                               original_text=news_input)
    except Exception as e:
        return render_template('index.html', categories=categories, error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
