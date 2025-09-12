import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model(X, y, model_type='random_forest'):
    """Trains a machine learning model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == 'isolation_forest':
        model = IsolationForest(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'random_forest', 'logistic_regression', or 'isolation_forest'.")

    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

def save_model(model, filename):
    """Saves the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Loads a trained model from a file."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

if __name__ == "__main__":
    # Example Usage (requires dummy data for demonstration):
    # In a real scenario, X would be TF-IDF features and y would be labels (bot/human)
    print("This script provides functions for training and saving machine learning models.")
    print("To use it, you'll need preprocessed data (features X and labels y).")
    print("Example: X = pd.DataFrame(some_tfidf_features), y = pd.Series(some_labels)")

    # Dummy data for demonstration purposes
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
    y_dummy_series = pd.Series(y_dummy)

    print("\n--- Training Random Forest Model with Dummy Data ---")
    rf_model = train_model(X_dummy_df, y_dummy_series, model_type='random_forest')
    save_model(rf_model, 'random_forest_model.joblib')

    print("--- Training Logistic Regression Model with Dummy Data ---")
    lr_model = train_model(X_dummy_df, y_dummy_series, model_type='logistic_regression')
    save_model(lr_model, 'logistic_regression_model.joblib')

    print("\n--- Training Isolation Forest Model with Dummy Data ---")
    # Isolation Forest is typically used for unsupervised anomaly detection, so it doesn't use y_dummy_series for training
    # For demonstration, we'll still pass it, but it's ignored by IsolationForest's fit method for unsupervised learning.
    # In a real scenario, you'd train it on your 'normal' data and then use it to detect anomalies.
    if_model = train_model(X_dummy_df, y_dummy_series, model_type='isolation_forest')
    save_model(if_model, 'isolation_forest_model.joblib')

    # Load and test a model
    loaded_rf_model = load_model('random_forest_model.joblib')
    print(f"Loaded model type: {type(loaded_rf_model)}")