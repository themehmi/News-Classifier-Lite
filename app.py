from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

# Load the components individually
# Make sure these filenames match what you saved in your notebook
tfidf = joblib.load('nclite.pkl')
clf = joblib.load('nclite_vec.pkl')

# Define the display labels for your categories
DISPLAY_CATEGORIES = ['Graphics', 'Medicine', 'Space', 'Politics']

# Ensure the mapping matches your training label order
# 0: comp.graphics, 1: sci.med, 2: sci.space, 3: talk.politics.guns
CATEGORY_MAP = {0: 'Graphics', 1: 'Medicine', 2: 'Space', 3: 'Politics'}

@app.route('/')
def home():
    return render_template('index.html', categories=DISPLAY_CATEGORIES)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form['news_text']
        
        # Transform the text manually
        # The vectorizer uses the vocabulary it learned during training
        text_vectorized = tfidf.transform([raw_text])
        
        # 3. Predict using the classifier
        prediction_idx = clf.predict(text_vectorized)[0]
        result = CATEGORY_MAP.get(prediction_idx, "Unknown")
        
        return render_template('index.html', 
                               prediction=result, 
                               original_text=raw_text,
                               categories=DISPLAY_CATEGORIES)

if __name__ == '__main__':
    app.run(debug=True)