import numpy as np
import pandas as pd
import hashlib
import random

# Create sample fingerprint data for users
def generate_sample_fingerprint(user_id, anomaly=False):
    """Generates browser fingerprint data with an optional anomaly flag."""
    sample_data = {
        "user_id": user_id,
        "canvasFingerprint": f"data:image/png;base64,{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=50)) if anomaly else hashlib.sha256(f'canvas_{user_id}'.encode()).hexdigest()}",
        "colorDepth": random.choice([16, 24, 32]) if anomaly else 24,
        "deviceMemory": random.choice([2, 4, 8, 16]) if anomaly else 8,
        "hardwareConcurrency": random.choice([2, 4, 8, 12, 16]) if anomaly else 8,
        "language": random.choice(["en-GB", "en-US", "fr-FR"]) if anomaly else "en-GB",
        "platform": random.choice(["MacIntel", "Win32", "Linux x86_64"]) if anomaly else "MacIntel",
        "screenResolution": random.choice(["1920x1080", "2560x1440", "3440x1440"]) if anomaly else "3440x1440",
        "timezone": random.choice(["Europe/London", "America/New_York", "Asia/Tokyo"]) if anomaly else "Europe/London",
        "touchSupport": random.choice([0, 1]) if anomaly else 0,
        "webglFingerprint": f"webgl_{random.randint(1, 1000)}" if anomaly else f"webgl_{user_id}"
    }
    return sample_data

def generate_fingerprint_data(num_records=5000, num_users=5, anomaly_percentage=0.1):
    """Generates a dataset of browser fingerprint data for multiple users with anomalies."""
    data = []
    user_ids = [f'user_{i}' for i in range(num_users)]
    
    for _ in range(num_records):
        user_id = random.choice(user_ids)
        anomaly = random.random() < anomaly_percentage
        fingerprint_data = generate_sample_fingerprint(user_id, anomaly=anomaly)
        data.append(fingerprint_data)
    
    df = pd.DataFrame(data)
    return df



# Function to generate IP address within specified range
def generate_ip_address(anomaly=False):
    """Generates an IP address, with an option for anomaly outside specified range."""
    if anomaly:
        # Generate an IP outside of the typical range
        octet = random.randint(21, 255)  # Anomalous IP, outside of .1 to .20 range
        return f"192.168.0.{octet}"
    else:
        # Normal IP within 192.168.0.1 to 192.168.0.20
        octet = random.randint(1, 20)
        return f"192.168.0.{octet}"

# Create sample fingerprint data for users with IP
def generate_sample_fingerprint(user_id, anomaly=False):
    """Generates browser fingerprint data with an optional anomaly flag."""
    sample_data = {
        "user_id": user_id,
        "canvasFingerprint": f"data:image/png;base64,{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=50)) if anomaly else hashlib.sha256(f'canvas_{user_id}'.encode()).hexdigest()}",
        "colorDepth": random.choice([16, 24, 32]) if anomaly else 24,
        "deviceMemory": random.choice([2, 4, 8, 16]) if anomaly else 8,
        "hardwareConcurrency": random.choice([2, 4, 8, 12, 16]) if anomaly else 8,
        "language": random.choice(["en-GB", "en-US", "fr-FR"]) if anomaly else "en-GB",
        "platform": random.choice(["MacIntel", "Win32", "Linux x86_64"]) if anomaly else "MacIntel",
        "screenResolution": random.choice(["1920x1080", "2560x1440", "3440x1440"]) if anomaly else "3440x1440",
        "timezone": random.choice(["Europe/London", "America/New_York", "Asia/Tokyo"]) if anomaly else "Europe/London",
        "touchSupport": random.choice([0, 1]) if anomaly else 0,
        "webglFingerprint": f"webgl_{random.randint(1, 1000)}" if anomaly else f"webgl_{user_id}",
        "ip_address": generate_ip_address(anomaly=anomaly)
    }
    return sample_data

def generate_fingerprint_data(num_records=5000, num_users=5, anomaly_percentage=0.1):
    """Generates a dataset of browser fingerprint data for multiple users with anomalies."""
    data = []
    user_ids = [f'user_{i}' for i in range(num_users)]
    
    for _ in range(num_records):
        user_id = random.choice(user_ids)
        anomaly = random.random() < anomaly_percentage
        fingerprint_data = generate_sample_fingerprint(user_id, anomaly=anomaly)
        data.append(fingerprint_data)
    
    df = pd.DataFrame(data)
    return df



# Main code to generate and save the dataset
if __name__ == "__main__":
    num_records = 5000
    num_users = 5
    anomaly_percentage = 0.1

    # Generate the data
    df = generate_fingerprint_data(num_records=num_records, num_users=num_users, anomaly_percentage=anomaly_percentage)

    # Save the data to a CSV file
    df.to_csv("fingerprint_data.csv", index=False)
    print(f"Data generated and saved to fingerprint_data.csv")

    # Generate the data
    df = generate_fingerprint_data(num_records=num_records, num_users=num_users, anomaly_percentage=anomaly_percentage)
    # Save the data to a CSV file
    df.to_csv("IP_data.csv", index=False)
    print(f"Data generated and saved to IP_data.csv")
