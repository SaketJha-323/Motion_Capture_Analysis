import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping

# Load the calibrated drone data
file_path = 'data/raw/Drone_CoD.csv'  # Your file path
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly: 'Frame', 'Time', 'X', 'Y', 'Z'
drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# Convert data to numpy array for easier manipulation
frames = drone_data['Frame'].values
times = drone_data['Time'].values
x = drone_data['X'].values / 100  # Convert from mm to meters
y = drone_data['Y'].values / 100  # Convert from mm to meters
z = drone_data['Z'].values / 100  # Convert from mm to meters

# Function to calculate velocity
def calculate_velocity(coord, times):
    return np.diff(coord) / np.diff(times)

# Function to calculate acceleration
def calculate_acceleration(velocity, times):
    return np.diff(velocity) / np.diff(times[1:])

# Calculate velocities
vx = calculate_velocity(x, times)
vy = calculate_velocity(y, times)
vz = calculate_velocity(z, times)

# Calculate accelerations
ax = calculate_acceleration(vx, times)
ay = calculate_acceleration(vy, times)
az = calculate_acceleration(vz, times)

# Ensure all arrays are the same size by trimming
min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
ax, ay, az = ax[:min_length], ay[:min_length], az[:min_length]

# Combine velocities and accelerations into a single array for anomaly detection
motion_data = np.vstack((vx, vy, vz, ax, ay, az)).T

# Create a placeholder for the ground truth array
num_samples = motion_data.shape[0]
y_true = np.random.choice([0, 1], size=(num_samples,), p=[0.9, 0.1])  # Example ground truth

# Step 1: Scale the data
scaler = StandardScaler()
motion_data_scaled = scaler.fit_transform(motion_data)

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(motion_data_scaled, y_true, test_size=0.2, random_state=42)

# Step 3: Fit Isolation Forest Model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predict anomalies
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Map -1 to 1 (anomaly), 1 to 0 (normal)

# Step 4: Fit One-Class SVM Model
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
svm.fit(X_train)

# Predict anomalies
y_pred_svm = svm.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)  # Map -1 to 1 (anomaly), 1 to 0 (normal)

# Step 5: Fit Autoencoder Model
input_dim = motion_data_scaled.shape[1]  # Number of features
encoding_dim = 3  # Adjust as needed

# Define the Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Create the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2)

# Step 6: Make predictions with the Autoencoder
predictions = autoencoder.predict(X_test)

# Compute reconstruction error
reconstruction_error = np.mean(np.power(X_test - predictions, 2), axis=1)

# Set a threshold for anomalies (e.g., 95th percentile)
threshold = np.percentile(reconstruction_error, 95)
y_pred_autoencoder = [1 if e > threshold else 0 for e in reconstruction_error]

# Step 7: Evaluate the models
print("Isolation Forest Metrics:")
print("Precision:", precision_score(y_test, y_pred_iso))
print("Recall:", recall_score(y_test, y_pred_iso))
print("F1 Score:", f1_score(y_test, y_pred_iso))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_iso))

print("\nOne-Class SVM Metrics:")
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_svm))

print("\nAutoencoder Metrics:")
print("Precision:", precision_score(y_test, y_pred_autoencoder))
print("Recall:", recall_score(y_test, y_pred_autoencoder))
print("F1 Score:", f1_score(y_test, y_pred_autoencoder))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_autoencoder))
