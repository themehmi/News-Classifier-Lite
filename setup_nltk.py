import nltk
import os

# Create the directory if it doesn't exist
data_path = './nltk_data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Download the specific corpora needed for Lemmatization
nltk.download('wordnet', download_dir=data_path)
nltk.download('omw-1.4', download_dir=data_path)
nltk.download('punkt', download_dir=data_path)