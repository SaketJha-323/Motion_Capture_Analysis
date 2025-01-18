import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Normalize the data for Isolation Forest
scaler = StandardScaler()
motion_data_scaled = scaler.fit_transform(motion_data)

# Apply Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)

# Find indices of anomalies
anomaly_indices = np.where(anomaly_scores == -1)[0]

# Visualize the results
dimensions = ['X', 'Y', 'Z']
velocities = [vx, vy, vz]
accelerations = [ax, ay, az]
colors = ['blue', 'green', 'purple']

for i, (dim, vel, acc, color) in enumerate(zip(dimensions, velocities, accelerations, colors)):
    plt.figure(figsize=(15, 12))
    
    # Plot velocity
    plt.subplot(3, 1, 1)
    plt.plot(vel, label=f'{dim} Velocity', color=color)
    plt.scatter(anomaly_indices, vel[anomaly_indices], color='red', label='Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'{dim}-Axis Velocity with Anomalies')
    plt.legend()
    plt.grid(True)
    
    # Plot acceleration
    plt.subplot(3, 1, 2)
    plt.plot(acc, label=f'{dim} Acceleration', color=color)
    plt.scatter(anomaly_indices[:-1], acc[anomaly_indices[:-1]], color='red', label='Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{dim}-Axis Acceleration with Anomalies')
    plt.legend()
    plt.grid(True)
    
    # Superimposed normalized plot
    plt.subplot(3, 1, 3)
    vel_norm = (vel - np.mean(vel)) / np.std(vel)
    acc_norm = (acc - np.mean(acc)) / np.std(acc)
    
    plt.plot(vel_norm, label=f'Normalized {dim} Velocity', color=color, alpha=0.7)
    plt.plot(acc_norm, label=f'Normalized {dim} Acceleration', color='darkgray', alpha=0.7)
    plt.scatter(anomaly_indices, vel_norm[anomaly_indices], color='red', marker='o', label='Velocity Anomalies')
    plt.scatter(anomaly_indices[:-1], acc_norm[anomaly_indices[:-1]], color='darkred', marker='s', label='Acceleration Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    plt.title(f'Superimposed Normalized {dim}-Axis Velocity and Acceleration')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Create a combined superimposed plot for all dimensions
plt.figure(figsize=(15, 10))

# Normalize all velocities and accelerations
vx_norm = (vx - np.mean(vx)) / np.std(vx)
vy_norm = (vy - np.mean(vy)) / np.std(vy)
vz_norm = (vz - np.mean(vz)) / np.std(vz)
ax_norm = (ax - np.mean(ax)) / np.std(ax)
ay_norm = (ay - np.mean(ay)) / np.std(ay)
az_norm = (az - np.mean(az)) / np.std(az)

# Plot velocities
plt.subplot(2, 1, 1)
plt.plot(vx_norm, label='X Velocity', color='blue', alpha=0.7)
plt.plot(vy_norm, label='Y Velocity', color='green', alpha=0.7)
plt.plot(vz_norm, label='Z Velocity', color='purple', alpha=0.7)
plt.scatter(anomaly_indices, vx_norm[anomaly_indices], color='red', marker='o', label='Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Velocity')
plt.title('Normalized Velocities Comparison (All Axes)')
plt.legend()
plt.grid(True)

# Plot accelerations
plt.subplot(2, 1, 2)
plt.plot(ax_norm, label='X Acceleration', color='blue', alpha=0.7)
plt.plot(ay_norm, label='Y Acceleration', color='green', alpha=0.7)
plt.plot(az_norm, label='Z Acceleration', color='purple', alpha=0.7)
plt.scatter(anomaly_indices[:-1], ax_norm[anomaly_indices[:-1]], color='red', marker='o', label='Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Acceleration')
plt.title('Normalized Accelerations Comparison (All Axes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print statistics for all dimensions
for dim, vel, acc in zip(dimensions, velocities, accelerations):
    print(f"\n{dim}-Axis Statistics:")
    print(f"Average Velocity: {np.mean(vel):.2f} m/s")
    print(f"Average Acceleration: {np.mean(acc):.2f} m/s²")
    print(f"Velocity Standard Deviation: {np.std(vel):.2f} m/s")
    print(f"Acceleration Standard Deviation: {np.std(acc):.2f} m/s²")

# Print detected anomalies
print("Anomalies detected at indices:", anomaly_indices)
