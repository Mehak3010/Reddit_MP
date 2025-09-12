import streamlit as st
import joblib
from data_preprocessing import clean_text
from predict import predict_bot_or_human # Import the prediction function

# Load models and vectorizer
@st.cache_resource
def load_resources():
    try:
        rf_model = joblib.load('bot_detection_random_forest_model.joblib')
        if_model = joblib.load('bot_detection_isolation_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return rf_model, if_model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Please run main.py first to train the model.")
        return None, None, None

rf_model, if_model, vectorizer = load_resources()

st.title("Reddit Bot Detection")
st.write("Enter text to predict if it's from a bot or a human.")

if rf_model and if_model and vectorizer:
    user_input = st.text_area("Enter text here:", "")

    model_choice = st.selectbox(
        "Choose Model for Prediction:",
        ('Random Forest (Supervised)', 'Isolation Forest (Unsupervised)')
    )

    if st.button("Predict"):
        if user_input:
            if model_choice == 'Random Forest (Supervised)':
                prediction_result = predict_bot_or_human(user_input, rf_model, vectorizer, model_type='random_forest')
            else: # Isolation Forest
                prediction_result = predict_bot_or_human(user_input, if_model, vectorizer, model_type='isolation_forest')
            st.write(prediction_result)
        else:
            st.warning("Please enter some text for prediction.")
else:
    st.info("Please run `main.py` to train the models and generate the necessary files before using this UI.")