import streamlit as st
import joblib
from data_preprocessing import clean_text, preprocess_data
from predict import predict_bot_or_human # Import the prediction function
from reddit_data_collector import get_user_data, get_subreddit_posts # Import data collection functions

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
st.write("Enter a Reddit username or subreddit to analyze for bot activity.")

# Input for Reddit username
username = st.text_input("Enter Reddit Username (optional):", "")

# Input for Subreddit
subreddit = st.text_input("Enter Subreddit (optional):", "")

if st.button("Fetch Data and Predict"):
    if not username and not subreddit:
        st.warning("Please enter either a Reddit username or a subreddit to analyze.")
    else:
        st.info("Fetching data from Reddit...")
        user_comments = []
        user_submissions = []
        subreddit_posts = []

        if username:
            st.write(f"Fetching data for user: {username}")
            user_comments, user_submissions = get_user_data(username)
            if not user_comments and not user_submissions:
                st.warning(f"Could not fetch data for user '{username}'. Please check the username and your Reddit API credentials.")
            else:
                st.write(f"Fetched {len(user_comments)} comments and {len(user_submissions)} submissions for user '{username}'.")

        if subreddit:
            st.write(f"Fetching data for subreddit: {subreddit}")
            subreddit_posts = get_subreddit_posts(subreddit)
            if not subreddit_posts:
                st.warning(f"Could not fetch posts for subreddit '{subreddit}'. Please check the subreddit name and your Reddit API credentials.")
            else:
                st.write(f"Fetched {len(subreddit_posts)} posts from subreddit '{subreddit}'.")

        if user_comments or user_submissions or subreddit_posts:
            st.success("Data fetched successfully!")
            # Combine all text data for preprocessing
            all_text_data = []
            for comment in user_comments:
                all_text_data.append(comment['body'])
            for submission in user_submissions:
                all_text_data.append(submission['title'] + " " + submission['selftext'])
            for post in subreddit_posts:
                all_text_data.append(post['title'] + " " + post['selftext'])

            if all_text_data:
                st.info("Preprocessing data...")
                processed_data = preprocess_data(all_text_data)
                st.write("Processed data samples:", processed_data[:5]) # Display first 5 processed samples
                # Now you would typically use the processed_data for prediction
                # For now, we'll just indicate that data was processed.
                st.success("Data preprocessed. Ready for prediction (prediction logic to be integrated).")
            else:
                st.warning("No textual data found to preprocess.")
        else:
            st.error("No data was fetched. Please try again with different inputs or check your credentials.")

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