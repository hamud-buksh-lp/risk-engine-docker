import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib

# Function to normalize features and train per-user models
def process_and_train_model(file_path, output_file):
    # Load data
    df = pd.read_csv(file_path)

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

    # Concatenate the scaled numerical and encoded categorical features
    normalized_df = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

    # Add user_id to the normalized DataFrame
    normalized_df['user_id'] = df['user_id']

    # Save the normalized DataFrame to a CSV file
    normalized_df.to_csv(output_file, index=False)

    # Save the encoders for later use
    joblib.dump(one_hot_encoder, 'one_hot_encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Train a per-user anomaly detection model
    user_ids = normalized_df['user_id'].unique()

    for user in user_ids:
        user_data = normalized_df[normalized_df['user_id'] == user].drop('user_id', axis=1)

        # Use Isolation Forest for anomaly detection
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(user_data)

        # Save the model for this user
        joblib.dump(model, f'user_model_{user}.joblib')

    return normalized_df

# File paths
input_file_path = 'fingerprint_data.csv'  # Replace with your input file path
output_file_path = 'normalized_data.csv'  # Output file path

# Process the data and train models
processed_data = process_and_train_model(input_file_path, output_file_path)

# Display the processed data
print(processed_data.head())
