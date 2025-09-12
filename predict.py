import pandas as pd
import joblib
from data_preprocessing import clean_text

def load_resources():
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        rf_model = joblib.load('bot_detection_random_forest_model.joblib')
        if_model = joblib.load('bot_detection_isolation_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        print("Model and vectorizer loaded successfully.")
        return rf_model, if_model, vectorizer
    except FileNotFoundError:
        print("Error: Model or vectorizer files not found. Please run main.py first to train the model.")
        return None, None

def predict_bot_or_human(text, model, vectorizer, model_type='random_forest'):
    """Predicts if a given text is from a bot or human."""
    if model is None or vectorizer is None:
        return "Error: Model or vectorizer not loaded."

    cleaned_input = clean_text(text)
    # Transform the single input text using the loaded vectorizer
    # The vectorizer expects an iterable (list of strings)
    input_vectorized = vectorizer.transform([cleaned_input])

    if model_type == 'random_forest':
        prediction = model.predict(input_vectorized)
        prediction_proba = model.predict_proba(input_vectorized)

        if prediction[0] == 1:
            return f"Prediction (Random Forest): Bot (Confidence: {prediction_proba[0][1]:.2f})"
        else:
            return f"Prediction (Random Forest): Human (Confidence: {prediction_proba[0][0]:.2f})"
    elif model_type == 'isolation_forest':
        # Isolation Forest returns -1 for outliers (anomalies/bots) and 1 for inliers (normal/humans)
        prediction = model.predict(input_vectorized)
        if prediction[0] == -1:
            return "Prediction (Isolation Forest): Bot (Anomaly detected)"
        else:
            return "Prediction (Isolation Forest): Human (Normal)"
    else:
        return "Error: Unsupported model type for prediction."

if __name__ == "__main__":
    rf_model, if_model, vectorizer = load_resources()

    if rf_model and if_model and vectorizer:
        print("\n--- Testing Prediction with Random Forest Model ---")

        # Example 1: Likely human-like text
        text1 = "Just saw a really interesting discussion on r/science about quantum physics. Fascinating stuff!"
        print(f"Text: '{text1}'")
        print(predict_bot_or_human(text1, rf_model, vectorizer, model_type='random_forest'))

        # Example 2: Likely bot-like text (based on dummy data patterns)
        text2 = "Click here for free V-bucks! Limited time offer! Don't miss out!"
        print(f"Text: '{text2}'")
        print(predict_bot_or_human(text2, rf_model, vectorizer, model_type='random_forest'))

        print("\n--- Testing Prediction with Isolation Forest Model ---")
        # Isolation Forest is for anomaly detection. It will classify texts that are 'different' from the training data as anomalies.
        # In this dummy example, 'bot-like' texts are anomalies.

        # Example 1: Likely human-like text (should be normal/human by IF)
        print(f"Text: '{text1}'")
        print(predict_bot_or_human(text1, if_model, vectorizer, model_type='isolation_forest'))

        # Example 2: Likely bot-like text (should be anomaly/bot by IF)
        print(f"Text: '{text2}'")
        print(predict_bot_or_human(text2, if_model, vectorizer, model_type='isolation_forest'))

        # Example 3: Another human-like text
        text3 = "My favorite hobby is hiking in the mountains during the fall. The colors are beautiful."
        print(f"Text: '{text3}'")
        print(predict_bot_or_human(text3, if_model, vectorizer, model_type='isolation_forest'))

        # Example 4: Another bot-like text
        text4 = "Earn easy money online! No experience needed. Visit our site now!"
        print(f"Text: '{text4}'")
        print(predict_bot_or_human(text4, if_model, vectorizer, model_type='isolation_forest'))

    print("\n`predict.py` updated. Run `main.py` first to generate the models and vectorizer files.")