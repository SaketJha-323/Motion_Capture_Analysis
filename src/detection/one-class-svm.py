import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Normalize the data for One-Class SVM
scaler = StandardScaler()
motion_data_scaled = scaler.fit_transform(motion_data)

# Apply One-Class SVM for anomaly detection
one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # nu controls the proportion of outliers
anomaly_scores = one_class_svm.fit_predict(motion_data_scaled)

# Find indices of anomalies
anomaly_indices = np.where(anomaly_scores == -1)[0]

# Explanation of anomalies
def explain_anomalies(anomaly_indices, vx, vy, vz, ax, ay, az):
    anomalies = []
    for idx in anomaly_indices:
        anomaly_desc = f"Anomaly at index {idx}: "
        if abs(vx[idx]) > np.mean(vx) + 2 * np.std(vx):
            anomaly_desc += f"Vx ({vx[idx]:.2f}) is abnormally high; "
        if abs(vy[idx]) > np.mean(vy) + 2 * np.std(vy):
            anomaly_desc += f"Vy ({vy[idx]:.2f}) is abnormally high; "
        if abs(vz[idx]) > np.mean(vz) + 2 * np.std(vz):
            anomaly_desc += f"Vz ({vz[idx]:.2f}) is abnormally high; "
        if abs(ax[idx]) > np.mean(ax) + 2 * np.std(ax):
            anomaly_desc += f"Ax ({ax[idx]:.2f}) is abnormally high; "
        if abs(ay[idx]) > np.mean(ay) + 2 * np.std(ay):
            anomaly_desc += f"Ay ({ay[idx]:.2f}) is abnormally high; "
        if abs(az[idx]) > np.mean(az) + 2 * np.std(az):
            anomaly_desc += f"Az ({az[idx]:.2f}) is abnormally high; "
        anomalies.append(anomaly_desc.strip())
    return anomalies

# Generate anomaly explanations
anomaly_explanations = explain_anomalies(anomaly_indices, vx, vy, vz, ax, ay, az)

# Visualize the results
plt.figure(figsize=(12, 8))

# Plot position data with anomalies highlighted
plt.subplot(3, 1, 1)
plt.plot(x[:-2], label='X Position')
plt.scatter(anomaly_indices, x[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (X)')

plt.subplot(3, 1, 2)
plt.plot(y[:-2], label='Y Position')
plt.scatter(anomaly_indices, y[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Y)')

plt.subplot(3, 1, 3)
plt.plot(z[:-2], label='Z Position')
plt.scatter(anomaly_indices, z[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Z)')

plt.tight_layout()
plt.show()

# Print detected anomalies and explanations
print("Anomalies detected at indices:", anomaly_indices)
for explanation in anomaly_explanations:
    print(explanation)

# Print average velocity and acceleration
print(f"\nAverage Velocity: Vx = {np.mean(vx):.2f} m/s, Vy = {np.mean(vy):.2f} m/s, Vz = {np.mean(vz):.2f} m/s")
print(f"Average Acceleration: Ax = {np.mean(ax):.2f} m/s^2, Ay = {np.mean(ay):.2f} m/s^2, Az = {np.mean(az):.2f} m/s^2")
