import json
import numpy as np
import pandas as pd
import joblib

# Load the saved models (OneHotEncoder and MinMaxScaler)
one_hot_encoder = joblib.load('one_hot_encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Normalize the input data
def normalize_input(data):
    df = pd.DataFrame([data])
    features = df[[
        'colorDepth',
        'deviceMemory',
        'hardwareConcurrency',
        'language',
        'platform',
        'screenResolution',
        'timezone',
        'touchSupport'
    ]]

    # One-hot encode categorical features
    categorical_cols = ['language', 'platform', 'screenResolution', 'timezone']
    encoded_categorical = one_hot_encoder.transform(features[categorical_cols])

    # Scale numerical features
    numerical_cols = ['colorDepth', 'deviceMemory', 'hardwareConcurrency', 'touchSupport']
    scaled_numerical = scaler.transform(features[numerical_cols])

    # Combine scaled numerical features and encoded categorical features
    normalized_data = pd.DataFrame(scaled_numerical, columns=numerical_cols)
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    normalized_data = pd.concat([normalized_data, encoded_categorical_df], axis=1)

    return normalized_data

# Calculate the risk score based on individual feature scores
def calculate_risk_score(normalized_data):
    # Define weights for each feature (currently equal weights)
    num_features = len(normalized_data.columns)
    feature_weights = np.ones(num_features)

    # Calculate individual feature scores (clip values between 0 and 1)
    feature_scores = normalized_data.iloc[0].to_dict()  # Get first row as a dictionary
    feature_scores_clipped = {key: np.clip(value, 0, 1) for key, value in feature_scores.items()}  # Clip each score

    # Calculate the overall risk score by averaging the clipped feature scores
    total_score = np.dot(list(feature_scores_clipped.values()), feature_weights) / num_features

    return feature_scores_clipped, total_score

def load_user_model(user_id):
    """Load the per-user IsolationForest model if it exists."""
    try:
        model_path = f'models/user_model_{user_id}.joblib'
        user_model = joblib.load(model_path)
        return user_model
    except FileNotFoundError:
        print(f"No model found for user {user_id}")
        return None

def calculate_user_risk_score(user_model, normalized_data):
    """Calculate the anomaly score for the specific user based on their model."""
    if user_model is not None:
        # Isolation Forest returns -1 for anomaly and 1 for normal data
        # We transform it to a risk score: 1 for anomaly and 0 for normal
        is_anomaly = user_model.predict(normalized_data)[0]
        user_score = 1.0 if is_anomaly == -1 else 0.0
        return user_score
    else:
        return None  # No user-specific model, no score

# Lambda handler
def lambda_handler(event, context):
    try:
        print("Incoming event:", event)

        # Directly use the incoming event as input_data if it’s a dict
        if isinstance(event, dict):
            input_data = event
        else:
            raise ValueError("Invalid input: event must be a dictionary.")

        # Normalize the input data
        normalized_data = normalize_input(input_data)

        # Calculate the individual scores and risk score
        feature_scores_clipped, risk_score = calculate_risk_score(normalized_data)
        
        # Add user-specific score if available
        user_id = input_data.get('user_id')
        user_model = load_user_model(user_id)
        user_risk_score = calculate_user_risk_score(user_model, normalized_data)


        # Return the result as JSON (without extra escape characters)
        return {
            'statusCode': 200,
            'user_id': input_data.get('user_id', 'unknown'),
            'feature_scores': feature_scores_clipped,
            'risk_score': risk_score,
            'risk_score_user_model': user_risk_score
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

# For local testing
if __name__ == '__main__':
    test_event = {
        "user_id": "user_1",  # Use a string for user_id for consistency
        "colorDepth": 24,
        "deviceMemory": 4,
        "hardwareConcurrency": 4,
        "language": "en",
        "platform": "Windows",
        "screenResolution": "1920x1080",
        "timezone": "UTC",
        "touchSupport": 1
    }
    print(lambda_handler(test_event, None))
