import numpy as np
import pandas as pd
import joblib
import json

# Load the saved models (OneHotEncoder and MinMaxScaler)
one_hot_encoder = joblib.load('data/one_hot_encoder.joblib')
scaler = joblib.load('data/scaler.joblib')

def load_user_model(user_id):
    """Load the per-user IsolationForest model if it exists."""
    try:
        model_path = f'data/user_fingerprint_model_{user_id}.joblib'
        user_model = joblib.load(model_path)
        return user_model
    except FileNotFoundError:
        print(f"No model found for user {user_id}")
        return None

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
    feature_scores_clipped = {key: float(np.clip(value, 0, 1)) for key, value in feature_scores.items()}  # Clip each score and convert to float

    # Calculate the overall risk score by averaging the clipped feature scores
    total_score = np.dot(list(feature_scores_clipped.values()), feature_weights) / num_features

    return feature_scores_clipped, float(total_score)  # Convert the score to float

def calculate_user_risk_score(user_model, normalized_data):
    """Calculate the anomaly score for the specific user based on their model."""
    if user_model is not None:
        # Ensure normalized_data has correct shape and columns
        if isinstance(normalized_data, pd.DataFrame):
            normalized_data = normalized_data.reindex(columns=user_model.feature_names_in_, fill_value=0)
        else:
            normalized_data = pd.DataFrame([normalized_data], columns=user_model.feature_names_in_)
        
        # Reshape if needed to ensure (1, num_features)
        if normalized_data.shape[0] == 1 and len(normalized_data.shape) == 2:
            is_anomaly = user_model.predict(normalized_data)[0]
            user_score = 1.0 if is_anomaly == -1 else 0.0
            return user_score
    return None  # Return None if user_model is missing

def calculate_ip_risk_score(ip_model, ip_data):
    """Calculate the anomaly score for IP using the IP-specific model."""
    if ip_model is not None:
        if isinstance(ip_data, pd.DataFrame):
            ip_data = ip_data.reindex(columns=ip_model.feature_names_in_, fill_value=0)
        else:
            ip_data = pd.DataFrame([ip_data], columns=ip_model.feature_names_in_)
        
        # Reshape to ensure (1, num_features)
        if ip_data.shape[0] == 1 and len(ip_data.shape) == 2:
            is_anomaly = ip_model.predict(ip_data)[0]
            ip_risk_score = 1.0 if is_anomaly == -1 else 0.0
            return ip_risk_score
    return 1.0  # Set to 1 if IP model is missing or IP is not seen (anomalous)

def lambda_handler(event, context):
    try:
        print("Incoming event:", event)

        if isinstance(event, dict):
            input_data = event
        else:
            raise ValueError("Invalid input: event must be a dictionary.")

        # Normalize the input data for the fingerprint features
        normalized_data = normalize_input(input_data)

        # Calculate the individual scores and risk score
        feature_scores_clipped, fingerprint_risk_score = calculate_risk_score(normalized_data)
        print(feature_scores_clipped)
        print(fingerprint_risk_score)

        # Add user-specific score if available
        user_id = input_data.get('user_id')

        # Calculate the IP risk score based on the user's IP model
        ip_model_path = f'data/user_ip_model_{user_id}.joblib'
        try:
            ip_model = joblib.load(ip_model_path)
            ip_encoder_path = f'data/ip_encoder_{user_id}.joblib'
            ip_encoder = joblib.load(ip_encoder_path)

            # Prepare the IP data for scoring (handle separately)
            ip_data = {'ip_address': input_data.get('ip_address')}
            print("before normalize ip")
            
            # Normalize the IP address (but it may not need extensive processing, it's up to your model)
            normalized_ip_data = pd.DataFrame([ip_data])
            
            # IP model prediction for anomaly detection
            ip_risk_score = calculate_ip_risk_score(ip_model, normalized_ip_data)
            print("IP risk score:", ip_risk_score)
        except FileNotFoundError:
            ip_risk_score = 1.0  # IP model not found or IP not seen, consider it anomalous

        # Calculate overall weighted score (weighted fingerprint score and IP score)
        weighted_fingerprint_score = fingerprint_risk_score * 7
        weighted_ip_score = ip_risk_score * 3
        total_weight = 10  # 7 + 3
        overall_score = (weighted_fingerprint_score + weighted_ip_score) / total_weight

        # Return the result as JSON with standard Python types (float)
        return {
            'statusCode': 200,
            'user_id': input_data.get('user_id', 'unknown'),
            'feature_scores': {key: float(value) for key, value in feature_scores_clipped.items()},  # Convert to float
            'risk_score_fingerprint': fingerprint_risk_score,
            'ip_risk_score': ip_risk_score,
            'overall_score': overall_score
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

# Test locally with the test_event data
if __name__ == '__main__':
    test_event = {
        "user_id": "user_1",
        "colorDepth": 24,
        "deviceMemory": 4,
        "hardwareConcurrency": 4,
        "language": "en-GB",
        "platform": "MacIntel",
        "screenResolution": "3440x1440",
        "timezone": "Europe/London",
        "touchSupport": 1,
        "ip_address": "192.168.0.91"
    }
    print(lambda_handler(test_event, None))
