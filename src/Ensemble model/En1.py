# Isolation Forest, One-Class SVM, LSTM Autoencoder

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)  # Fill missing values
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def split_sequences(data, n_steps):
    sequences = []
    for i in range(len(data) - n_steps + 1):
        sequences.append(data[i:i + n_steps])
    return np.array(sequences)

# Isolation Forest model
def train_isolation_forest(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    scores = model.decision_function(data)
    return scores

# One-Class SVM model
def train_one_class_svm(data):
    model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    model.fit(data)
    scores = model.decision_function(data)
    return scores

# LSTM Autoencoder model
def train_lstm_autoencoder(data, n_steps, epochs=20, batch_size=32):
    n_features = data.shape[1]
    sequences = split_sequences(data, n_steps)

    X_train = sequences[:, :-1]
    X_test = sequences[:, 1:]

    model = Sequential([
        LSTM(128, activation='relu', input_shape=(n_steps - 1, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(n_steps - 1),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test), verbose=1)

    reconstruction = model.predict(X_train)
    errors = np.mean(np.square(X_train - reconstruction), axis=(1, 2))
    return errors

# Combine results
def combine_anomaly_scores(scores_if, scores_svm, scores_lstm, weights=[1, 1, 1]):
    """
    Combine anomaly scores from different methods with alignment.
    """
    # Ensure all scores are numpy arrays
    scores_if = np.array(scores_if)
    scores_svm = np.array(scores_svm)
    scores_lstm = np.array(scores_lstm)
    
    # Find the minimum length of scores
    min_length = min(len(scores_if), len(scores_svm), len(scores_lstm))
    
    # Trim all scores to the same length
    scores_if = scores_if[:min_length]
    scores_svm = scores_svm[:min_length]
    scores_lstm = scores_lstm[:min_length]
    
    # Combine scores using weights
    combined_scores = (weights[0] * scores_if) + (weights[1] * scores_svm) + (weights[2] * scores_lstm)
    return combined_scores
def analyze_anomalies(data, scores, threshold):
    """
    Analyze anomalies and provide detailed explanations
    """
    anomalies = np.where(scores > threshold)[0]
    anomaly_scores = scores[anomalies]
    
    # Create a detailed analysis
    analysis = []
    for idx, score in zip(anomalies, anomaly_scores):
        severity = "High" if score > np.percentile(scores, 98) else "Medium"
        
        # Determine local context
        start_idx = max(0, idx - 5)
        end_idx = min(len(data), idx + 5)
        local_mean = np.mean(data[start_idx:end_idx])
        local_std = np.std(data[start_idx:end_idx])
        
        # Analyze why it's an anomaly
        value = data[idx]
        if value > local_mean + 2*local_std:
            reason = "Unusually high value compared to surrounding data"
        elif value < local_mean - 2*local_std:
            reason = "Unusually low value compared to surrounding data"
        else:
            reason = "Irregular pattern detected in the sequence"
            
        analysis.append({
            'index': idx,
            'value': value,
            'severity': severity,
            'score': score,
            'reason': reason
        })
    
    return pd.DataFrame(analysis)

def enhanced_visualization(data, scores, threshold, analysis_df):
    """
    Create multiple visualizations for better understanding
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Time Series with Anomalies
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Normal Data', color='blue', alpha=0.5)
    
    # Color-code anomalies by severity
    high_severity = analysis_df[analysis_df['severity'] == 'High']
    medium_severity = analysis_df[analysis_df['severity'] == 'Medium']
    
    plt.scatter(high_severity['index'], data[high_severity['index'].values],
                color='red', label='High Severity Anomalies', s=100)
    plt.scatter(medium_severity['index'], data[medium_severity['index'].values],
                color='orange', label='Medium Severity Anomalies', s=100)
    
    plt.title('Time Series Data with Detected Anomalies', fontsize=12)
    plt.legend()
    
    # Plot 2: Anomaly Scores Distribution
    plt.subplot(3, 1, 2)
    sns.histplot(scores, bins=50, color='blue', alpha=0.5)
    plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.title('Distribution of Anomaly Scores', fontsize=12)
    plt.legend()
    
    # Plot 3: Local Context View
    plt.subplot(3, 1, 3)
    for idx in analysis_df['index'][:3]:  # Show first 3 anomalies for clarity
        start_idx = max(0, idx - 10)
        end_idx = min(len(data), idx + 10)
        
        plt.plot(range(start_idx, end_idx), data[start_idx:end_idx], 
                label=f'Context around anomaly at {idx}')
        plt.scatter(idx, data[idx], color='red', s=100)
    
    plt.title('Local Context Around Selected Anomalies', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_anomaly_report(analysis_df):
    """
    Print a human-readable report of the anomalies
    """
    print("\n=== ANOMALY DETECTION REPORT ===")
    print(f"Total Anomalies Found: {len(analysis_df)}")
    print("\nDetailed Breakdown:")
    print("-" * 80)
    
    for severity in ['High', 'Medium']:
        severity_group = analysis_df[analysis_df['severity'] == severity]
        print(f"\n{severity} Severity Anomalies ({len(severity_group)} found):")
        
        for _, row in severity_group.iterrows():
            print(f"\nTime Index: {row['index']}")
            print(f"Value: {row['value']:.3f}")
            print(f"Reason: {row['reason']}")
            print("-" * 40)

def main(file_path):
    # Load and preprocess data
    raw_data = load_data(file_path)
    scaled_data, scaler = preprocess_data(raw_data)

    # Train models
    scores_if = train_isolation_forest(scaled_data)
    scores_svm = train_one_class_svm(scaled_data)
    scores_lstm = train_lstm_autoencoder(scaled_data, n_steps=10)

    # Normalize and combine scores
    scores_if = (scores_if - scores_if.min()) / (scores_if.max() - scores_if.min())
    scores_svm = (scores_svm - scores_svm.min()) / (scores_svm.max() - scores_svm.min())
    scores_lstm = (scores_lstm - scores_lstm.min()) / (scores_lstm.max() - scores_lstm.min())
    combined_scores = combine_anomaly_scores(scores_if, scores_svm, scores_lstm)

    # Analyze anomalies
    threshold = np.percentile(combined_scores, 95)
    analysis_df = analyze_anomalies(scaled_data[:, 0], combined_scores, threshold)
    
    # Generate visualizations and report
    enhanced_visualization(scaled_data[:, 0], combined_scores, threshold, analysis_df)
    print_anomaly_report(analysis_df)

# Execute main pipeline
if __name__ == "__main__":
    file_path = 'data/raw/Drone_CoD.csv'
    main(file_path)