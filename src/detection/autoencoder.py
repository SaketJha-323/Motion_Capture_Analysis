import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
def calculate_velocity(x, times):
    return np.diff(x) / np.diff(times)

# Function to calculate acceleration
def calculate_acceleration(vx, times):
    return np.diff(vx) / np.diff(times[1:])

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

# Normalize the data for Autoencoder
scaler = StandardScaler()
motion_data_scaled = scaler.fit_transform(motion_data)

# Split the data into training and testing sets
X_train, X_test = train_test_split(motion_data_scaled, test_size=0.2, random_state=42)

# Build the Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 8  # Increased bottleneck size

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Dropout layer added
encoded = BatchNormalization()(encoded)  # Adding Batch Normalization
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the Autoencoder
history = autoencoder.fit(X_train, X_train, 
                          epochs=100, 
                          batch_size=32, 
                          validation_data=(X_test, X_test),
                          callbacks=[early_stopping],
                          verbose=1)

# Use the Autoencoder to reconstruct the data
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error
train_reconstruction_error = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
test_reconstruction_error = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Define a threshold for anomaly detection
# Calculate mean and standard deviation of training reconstruction errors
mean_train_error = np.mean(train_reconstruction_error)
std_train_error = np.std(train_reconstruction_error)
threshold = mean_train_error + 2 * std_train_error  # Dynamic threshold

# Identify anomalies in the test data
anomalies = test_reconstruction_error > threshold
anomaly_indices = np.where(anomalies)[0]

# Visualization of reconstruction errors
plt.figure(figsize=(10, 6))

# Plot the reconstruction error
plt.hist(train_reconstruction_error, bins=50, alpha=0.75, label='Training Data')
plt.hist(test_reconstruction_error, bins=50, alpha=0.75, label='Test Data')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error Histogram")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plotting anomalies in position data
plt.figure(figsize=(12, 8))

# Plot X Position with anomalies highlighted
plt.subplot(3, 1, 1)
plt.plot(x[:-2], label='X Position')
plt.scatter(anomaly_indices, x[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (X)')

# Plot Y Position with anomalies highlighted
plt.subplot(3, 1, 2)
plt.plot(y[:-2], label='Y Position')
plt.scatter(anomaly_indices, y[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Y)')

# Plot Z Position with anomalies highlighted
plt.subplot(3, 1, 3)
plt.plot(z[:-2], label='Z Position')
plt.scatter(anomaly_indices, z[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Z)')

plt.tight_layout()
plt.show()

# Print average velocity and acceleration
print(f"\nAverage Velocity: Vx = {np.mean(vx):.2f} m/s, Vy = {np.mean(vy):.2f} m/s, Vz = {np.mean(vz):.2f} m/s")
print(f"Average Acceleration: Ax = {np.mean(ax):.2f} m/s^2, Ay = {np.mean(ay):.2f} m/s^2, Az = {np.mean(az):.2f} m/s^2")
