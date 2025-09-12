import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data if not already present
try:
    stopwords.words('english')
    WordNetLemmatizer()
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans text data by removing special characters, lowercasing, and tokenizing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(data_list):
    """Applies text cleaning and TF-IDF vectorization to a list of text data."""
    if not data_list:
        return pd.DataFrame(), None

    df = pd.DataFrame(data_list)

    # Combine relevant text fields for processing
    if 'body' in df.columns and 'title' in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
    elif 'body' in df.columns:
        df['full_text'] = df['body'].fillna('')
    elif 'title' in df.columns:
        df['full_text'] = df['title'].fillna('')
    elif 'selftext' in df.columns:
        df['full_text'] = df['selftext'].fillna('')
    else:
        df['full_text'] = ''

    df['cleaned_text'] = df['full_text'].apply(clean_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features to avoid high dimensionality
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

    # Convert TF-IDF matrix to DataFrame for easier handling
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine original DataFrame with TF-IDF features
    processed_df = pd.concat([df, tfidf_df], axis=1)

    return processed_df, vectorizer

if __name__ == "__main__":
    # Example Usage:
    sample_comments = [
        {'id': 'c1', 'body': 'This is a great comment about machine learning.', 'title': 'ML Post'},
        {'id': 'c2', 'body': 'Another comment, very positive!', 'title': 'Good News'},
        {'id': 'c3', 'body': 'Spammy content here, buy my product!', 'title': 'Spam Alert'}
    ]

    sample_submissions = [
        {'id': 's1', 'title': 'New AI breakthrough', 'selftext': 'Researchers announced a new AI model today.'},
        {'id': 's2', 'title': 'Fake news article', 'selftext': 'This article contains misleading information.'}
    ]

    print("\n--- Preprocessing Sample Comments ---")
    processed_comments_df, comment_vectorizer = preprocess_data(sample_comments)
    if not processed_comments_df.empty:
        print(processed_comments_df[['id', 'cleaned_text']].head())
        print(f"Shape of processed comments: {processed_comments_df.shape}")

    print("\n--- Preprocessing Sample Submissions ---")
    processed_submissions_df, submission_vectorizer = preprocess_data(sample_submissions)
    if not processed_submissions_df.empty:
        print(processed_submissions_df[['id', 'cleaned_text']].head())
        print(f"Shape of processed submissions: {processed_submissions_df.shape}")

    print("\n`data_preprocessing.py` created. This script provides functions for cleaning text and generating TF-IDF features. You may need to install `nltk` and `scikit-learn` if you haven't already.")
    print("Remember to download NLTK data (stopwords, wordnet) if running for the first time.")