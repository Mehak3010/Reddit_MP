import os
import pandas as pd
from reddit_data_collector import get_user_data, get_subreddit_posts
from data_preprocessing import preprocess_data
from model_training import train_model, save_model

def main():
    print("Starting Reddit Bot Detection AI Model Training Process...")

    # --- Step 1: Data Collection ---
    print("\n--- Step 1: Data Collection ---")
    # NOTE: Replace with actual users/subreddits for data collection
    # For demonstration, we'll use dummy data or a very small sample.
    # In a real scenario, you'd collect a large dataset of known bots and legitimate users.

    # Example: Collect data for a few users (replace with actual usernames)
    # Example: Collect data for a few users (replace with actual usernames)
    # user_data_samples = []
    # users_to_collect = ['spez', 'AutoModerator'] # Example users
    # for user in users_to_collect:
    #     data = get_user_data(user, limit=50) # Limit for quick testing
    #     if data:
    #         user_data_samples.append(data)

    # Example: Collect posts from a subreddit (replace with actual subreddit)
    # subreddit_posts_samples = get_subreddit_posts('all', limit=50) # Limit for quick testing

    # For initial testing, we'll create some dummy data that mimics the structure
    # In a real application, you would load your collected data here.
    print("Using dummy data for demonstration. Please collect real data for actual training.")
    dummy_comments = [
        {'id': 'c1', 'body': 'This is a normal comment.', 'title': 'Normal Post', 'author': 'user1'},
        {'id': 'c2', 'body': 'Another regular comment here.', 'title': 'Regular Post', 'author': 'user2'},
        {'id': 'c3', 'body': 'Spammy content, buy my product now!', 'title': 'Spam Ad', 'author': 'bot1'},
        {'id': 'c4', 'body': 'I am a human, definitely not a bot.', 'title': 'Human Talk', 'author': 'user3'},
        {'id': 'c5', 'body': 'Free V-bucks! Click here!', 'title': 'Scam Alert', 'author': 'bot2'},
    ]
    dummy_submissions = [
        {'id': 's1', 'title': 'Interesting article on AI', 'selftext': 'Discussing recent advancements.', 'author': 'user1'},
        {'id': 's2', 'title': 'How to get rich quick', 'selftext': 'Follow these simple steps!', 'author': 'bot1'},
        {'id': 's3', 'title': 'My cat is cute', 'selftext': 'Just sharing a pic of my cat.', 'author': 'user2'},
    ]

    # Combine all text data for preprocessing
    all_text_data = []
    for comment in dummy_comments:
        all_text_data.append({'text': comment['body'], 'type': 'comment', 'author': comment['author']})
    for submission in dummy_submissions:
        all_text_data.append({'text': submission['title'] + ' ' + submission['selftext'], 'type': 'submission', 'author': submission['author']})

    # Assign dummy labels for demonstration (0: human, 1: bot)
    # In a real scenario, you'd need a labeled dataset.
    labels = []
    for item in all_text_data:
        if 'bot' in item['author']:
            labels.append(1) # Label as bot
        else:
            labels.append(0) # Label as human

    raw_df = pd.DataFrame(all_text_data)
    raw_df['label'] = labels

    # --- Step 2: Data Preprocessing ---
    print("\n--- Step 2: Data Preprocessing ---")
    # The preprocess_data function expects a list of dictionaries, where each dict is a comment/submission
    # We need to adapt the dummy_comments and dummy_submissions to fit this structure for preprocessing
    # For simplicity, let's just use the 'text' column from raw_df for TF-IDF
    processed_df, vectorizer = preprocess_data(raw_df.to_dict('records'))

    if processed_df.empty:
        print("No data to preprocess. Exiting.")
        return

    # Extract features (X) and labels (y)
    # X will be the TF-IDF features, y will be the 'label' column
    # Ensure that 'label' is not part of the features for training
    feature_columns = [col for col in processed_df.columns if col.startswith('feature_') or col in vectorizer.get_feature_names_out()]
    X = processed_df[feature_columns]
    y = processed_df['label']

    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")

    # --- Step 3: Model Training ---
    print("\n--- Step 3: Model Training ---")
    # Train a Random Forest model
    print("\n--- Training Random Forest Model ---")
    rf_model = train_model(X, y, model_type='random_forest')
    save_model(rf_model, 'bot_detection_random_forest_model.joblib')

    # Train an Isolation Forest model
    print("\n--- Training Isolation Forest Model ---")
    # Isolation Forest is an unsupervised anomaly detection algorithm.
    # It does not use 'y' (labels) for training, but rather identifies outliers in 'X'.
    # For bot detection, you might train it on known 'human' data and then use it to flag deviations.
    if_model = train_model(X, y, model_type='isolation_forest')
    save_model(if_model, 'bot_detection_isolation_forest_model.joblib')

    # For the purpose of this project, we'll primarily use the Random Forest model for prediction
    # as it's a supervised classification model more directly suited for 'bot' vs 'human' labeling.
    model = rf_model

    # Save the trained model and vectorizer
    # The primary model for prediction will be the Random Forest model
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Models and TF-IDF vectorizer saved.")

    print("\nReddit Bot Detection AI Model Training Process Completed.")
    print("You can now use 'bot_detection_random_forest_model.joblib' (for supervised classification) or 'bot_detection_isolation_forest_model.joblib' (for anomaly detection) and 'tfidf_vectorizer.joblib' for prediction.")

if __name__ == "__main__":
    main()