# Reddit Bot and Fake Account Detection AI Model

This project aims to build an AI model to detect bots and fake accounts on Reddit using the PRAW library.

## Project Structure

- `requirements.txt`: Lists all necessary Python dependencies.
- `reddit_data_collector.py`: Script to collect data from Reddit using PRAW.
- `data_preprocessing.py`: Contains functions for cleaning and preprocessing text data, including TF-IDF vectorization.
- `model_training.py`: Script for training and evaluating machine learning models (Random Forest, Logistic Regression, Isolation Forest).
- `main.py`: Orchestrates the data collection, preprocessing, and model training pipeline.
- `predict.py`: Demonstrates how to load the trained model and make predictions on new text.

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Reddit_MP.git
    cd Reddit_MP
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **NLTK Data Download:**
    The `data_preprocessing.py` script uses NLTK for stopwords and lemmatization. You might need to download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5.  **Configure Reddit API Credentials:**

    *   Create a Reddit API application [here](https://www.reddit.com/prefs/apps).
    *   Choose `script` as the application type.
    *   Note down your `client_id` (under your app's name) and `client_secret` (the string next to `secret`).
    *   Set the following environment variables with your credentials:

        **Windows (Command Prompt):**
        ```cmd
        set REDDIT_CLIENT_ID=YOUR_CLIENT_ID
        set REDDIT_CLIENT_SECRET=YOUR_CLIENT_SECRET
        set REDDIT_USER_AGENT="BotDetectionScript by /u/YourRedditUsername"
        ```

        **Windows (PowerShell):**
        ```powershell
        $env:REDDIT_CLIENT_ID="YOUR_CLIENT_ID"
        $env:REDDIT_CLIENT_SECRET="YOUR_CLIENT_SECRET"
        $env:REDDIT_USER_AGENT="BotDetectionScript by /u/YourRedditUsername"
        ```

        **macOS/Linux (Bash/Zsh):**
        ```bash
        export REDDIT_CLIENT_ID="YOUR_CLIENT_ID"
        export REDDIT_CLIENT_SECRET="YOUR_CLIENT_SECRET"
        export REDDIT_USER_AGENT="BotDetectionScript by /u/YourRedditUsername"
        ```

## How to Run

1.  **Train the Models:**
    Run the `main.py` script to collect (dummy) data, preprocess it, and train the bot detection models. This will save `bot_detection_random_forest_model.joblib`, `bot_detection_isolation_forest_model.joblib`, and `tfidf_vectorizer.joblib`.
    ```bash
    python main.py
    ```

2.  **Run the Streamlit UI:**
    After training the models, you can launch the interactive UI using Streamlit.
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your web browser with the application.

3.  **Make Predictions (CLI):**
    Alternatively, you can still use `predict.py` to test the model with new text inputs via the command line.
    ```bash
    python predict.py
    ```

## Model Details

The `model_training.py` script currently supports `RandomForestClassifier`, `LogisticRegression`, and `IsolationForest`. The `main.py` trains both `RandomForestClassifier` (for supervised classification) and `IsolationForest` (for unsupervised anomaly detection). `predict.py` demonstrates how to use both.

## Future Improvements

-   Implement more sophisticated data collection strategies.
-   Explore advanced NLP techniques (e.g., word embeddings, deep learning models).
-   Incorporate more features (e.g., user metadata, network analysis).
-   Build a web interface for real-time detection.