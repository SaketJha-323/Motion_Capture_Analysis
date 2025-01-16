# Isolation Forest, One-Class SVM, LSTM Autoencoder

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# Visualization
def visualize_anomalies(data, scores, threshold, title='Anomaly Detection'):
    anomalies = np.where(scores > threshold)[0]
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data')
    plt.scatter(anomalies, data[anomalies], color='red', label='Anomalies')
    plt.title(title)
    plt.legend()
    plt.show()

# Main pipeline
def main(file_path):
    # Load and preprocess data
    raw_data = load_data(file_path)
    scaled_data, scaler = preprocess_data(raw_data)

    # Train models
    scores_if = train_isolation_forest(scaled_data)
    scores_svm = train_one_class_svm(scaled_data)
    scores_lstm = train_lstm_autoencoder(scaled_data, n_steps=10)

    # Normalize scores for combining
    scores_if = (scores_if - scores_if.min()) / (scores_if.max() - scores_if.min())
    scores_svm = (scores_svm - scores_svm.min()) / (scores_svm.max() - scores_svm.min())
    scores_lstm = (scores_lstm - scores_lstm.min()) / (scores_lstm.max() - scores_lstm.min())

    # Combine scores
    combined_scores = combine_anomaly_scores(scores_if, scores_svm, scores_lstm)

    # Set threshold and visualize
    threshold = np.percentile(combined_scores, 95)  # Top 5% as anomalies
    visualize_anomalies(scaled_data[:, 0], combined_scores, threshold)

# Execute main pipeline
file_path = 'data/raw/Drone_CoD.csv'
main(file_path)
