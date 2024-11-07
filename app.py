import numpy as np
import pandas as pd
import joblib
import json
import ipaddress

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

def ip_to_int(ip_address):
    """Convert IP address to an integer."""
    try:
        return int(ipaddress.ip_address(ip_address))
    except ValueError:
        return None

# Normalize the input data
def normalize_input(data):
    df = pd.DataFrame([data])
    
    # Convert IP address to integer
    df['ip_address_int'] = df['ip_address'].apply(ip_to_int)
    
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
    num_features = len(normalized_data.columns)
    feature_weights = np.ones(num_features)

    feature_scores = normalized_data.iloc[0].to_dict()
    feature_scores_clipped = {key: float(np.clip(value, 0, 1)) for key, value in feature_scores.items()}
    total_score = np.dot(list(feature_scores_clipped.values()), feature_weights) / num_features

    return feature_scores_clipped, float(total_score)

def calculate_user_risk_score(user_model, normalized_data):
    if user_model is not None:
        normalized_data = normalized_data.reindex(columns=user_model.feature_names_in_, fill_value=0)
        is_anomaly = user_model.predict(normalized_data)[0]
        return 1.0 if is_anomaly == -1 else 0.0
    return None

def calculate_ip_risk_score(ip_model, ip_data):
    """Calculate the anomaly score for the IP address using the IP-specific model."""
    if ip_model is not None:
        # Use the integer-converted IP data for prediction
        is_anomaly = ip_model.predict(ip_data)[0]
        return 1.0 if is_anomaly == -1 else 0.0
    return 1.0  # Consider IP anomalous if no model is available


def lambda_handler(event, context):
    try:
        print("Incoming event:", event)
        input_data = event if isinstance(event, dict) else {}
        
        # Normalize fingerprint data
        normalized_data = normalize_input(input_data)
        feature_scores_clipped, fingerprint_risk_score = calculate_risk_score(normalized_data)
        
        # Load the user's model and compute user-specific risk score
        user_id = input_data.get('user_id')
        user_model = load_user_model(user_id)
        user_risk_score = calculate_user_risk_score(user_model, normalized_data)

        # Handle IP risk scoring
        ip_model_path = f'data/user_ip_model_{user_id}.joblib'
        ip_risk_score = 1.0  # Default to 1 (anomalous) if no model found
        try:
            ip_model = joblib.load(ip_model_path)
            ip_data = pd.DataFrame([[ip_to_int(input_data.get('ip_address'))]], columns=['ip_address_int'])
            ip_risk_score = calculate_ip_risk_score(ip_model, ip_data)
        except FileNotFoundError:
            print(f"No IP model found for user {user_id}, IP considered anomalous.")

        # Calculate the overall risk score with weighted scores
        weighted_fingerprint_score = fingerprint_risk_score * 7
        weighted_ip_score = ip_risk_score * 3
        overall_score = (weighted_fingerprint_score + weighted_ip_score) / 10

        # Prepare response
        return {
            'statusCode': 200,
            'user_id': user_id,
            'feature_scores': {key: float(value) for key, value in feature_scores_clipped.items()},
            'risk_score_fingerprint': fingerprint_risk_score,
            'ip_risk_score': ip_risk_score,
            'overall_score': overall_score
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }


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
        "ip_address": "192.168.0.145"
    }
    print(lambda_handler(test_event, None))
