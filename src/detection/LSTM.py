import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = 'data/raw/Drone_CoD.csv'
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
def calculate_velocity(pos, times):
    return np.diff(pos) / np.diff(times)

# Function to calculate acceleration
def calculate_acceleration(vel, times):
    return np.diff(vel) / np.diff(times[1:])

# Calculate velocities and accelerations
vx, vy, vz = calculate_velocity(x, times), calculate_velocity(y, times), calculate_velocity(z, times)
ax, ay, az = calculate_acceleration(vx, times), calculate_acceleration(vy, times), calculate_acceleration(vz, times)

# Ensure consistent sizes
min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
motion_data = np.vstack((vx[:min_length], vy[:min_length], vz[:min_length], ax[:min_length], ay[:min_length], az[:min_length])).T

# Normalize the data
scaler = MinMaxScaler()
motion_data_scaled = scaler.fit_transform(motion_data)

# Create sequences for LSTM
sequence_length = 50
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequences = create_sequences(motion_data_scaled, sequence_length)

# Split into training and test sets
train_size = int(0.8 * len(sequences))
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, sequences.shape[2]), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(sequences.shape[2])
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_data, train_data, epochs=20, batch_size=32, validation_split=0.1)

# Predict and calculate reconstruction error
reconstructed = model.predict(test_data)
reconstruction_error = np.mean((test_data - reconstructed) ** 2, axis=(1, 2))

# Define anomaly threshold
threshold = np.percentile(reconstruction_error, 95)

# Identify anomalies
anomalies = reconstruction_error > threshold

# Visualization of reconstruction error
plt.figure(figsize=(12, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error and Anomaly Threshold')
plt.legend()
plt.show()

# Text-based output
anomaly_indices = np.where(anomalies)[0]
print("Anomalies detected at indices:", anomaly_indices)

# Visualize anomalies in 3D space
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(motion_data[:, 0], motion_data[:, 1], motion_data[:, 2], c='blue', label='Normal Data')
ax.scatter(motion_data[anomaly_indices, 0], motion_data[anomaly_indices, 1], motion_data[anomaly_indices, 2], c='red', label='Anomalies')
ax.set_title('Anomalies in Motion Data (Velocity in 3D Space)')
ax.set_xlabel('Vx')
ax.set_ylabel('Vy')
ax.set_zlabel('Vz')
ax.legend()
plt.show()
