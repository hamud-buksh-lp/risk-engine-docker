import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
import ipaddress

# Function to convert IP to integer
def ip_to_int(ip_address):
    try:
        return int(ipaddress.ip_address(ip_address))
    except ValueError:
        # Handle any non-standard IPs like 'unknown_ip'
        return None  # Or a specific integer if you want a fallback

# Function to normalize features and train per-user models
def process_and_train_model(file_path, output_file):
    # Load data
    df = pd.read_csv(file_path)

    # Handle missing IP addresses
    df['ip_address'].fillna('unknown_ip', inplace=True)

    # Convert IP addresses to integer format
    df['ip_address_int'] = df['ip_address'].apply(ip_to_int)

    # Select relevant features
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
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = one_hot_encoder.fit_transform(features[categorical_cols])

    # Create DataFrame for the encoded categorical features
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

    # Scale numerical features
    numerical_cols = ['colorDepth', 'deviceMemory', 'hardwareConcurrency', 'touchSupport']
    scaler = MinMaxScaler()
    scaled_numerical = scaler.fit_transform(features[numerical_cols])

    # Create DataFrame for scaled numerical features
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols)

    # Concatenate the scaled numerical, encoded categorical, and IP address features
    normalized_df = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

    # Add user_id and the integer IP address to the normalized DataFrame
    normalized_df['user_id'] = df['user_id']
    normalized_df['ip_address_int'] = df['ip_address_int']

    # Save the normalized DataFrame to a CSV file
    normalized_df.to_csv(output_file, index=False)

    # Save the encoders for later use
    joblib.dump(one_hot_encoder, 'data/one_hot_encoder.joblib')
    joblib.dump(scaler, 'data/scaler.joblib')

    # Train a per-user anomaly detection model
    user_ids = normalized_df['user_id'].unique()

    for user in user_ids:
        user_data = normalized_df[normalized_df['user_id'] == user].drop(['user_id', 'ip_address_int'], axis=1)

        # Use Isolation Forest for anomaly detection on primary features
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(user_data)

        # Save the model for this user
        joblib.dump(model, f'data/user_fingerprint_model_{user}.joblib')

        # Train a separate Isolation Forest model specifically for the IP address feature
        user_ip_data = normalized_df[normalized_df['user_id'] == user][['ip_address_int']].dropna()

        # Only fit model if there are IP addresses to train on
        if not user_ip_data.empty:
            ip_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            ip_model.fit(user_ip_data)

            # Save the IP model for this user
            joblib.dump(ip_model, f'data/user_ip_model_{user}.joblib')

    return normalized_df

# File paths
input_file_path = 'data/fingerprint_data.csv'  # Replace with your input file path
output_file_path = 'data/normalized_data.csv'  # Output file path

# Process the data and train models
processed_data = process_and_train_model(input_file_path, output_file_path)

# Display the processed data
print(processed_data.head())
