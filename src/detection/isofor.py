import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.3, random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)

# Find indices of anomalies
anomaly_indices = np.where(anomaly_scores == -1)[0]

# Visualize the results
dimensions = ['X', 'Y', 'Z']
velocities = [vx, vy, vz]
accelerations = [ax, ay, az]
colors = ['blue', 'green', 'purple']

# Combine velocities for all axes into a single array
velocity_data = np.vstack((vx, vy, vz)).T  # Shape: (n_samples, 3)

# Apply KMeans clustering on the combined velocity data
kmeans_vel = KMeans(n_clusters=1, random_state=42)
vel_clusters = kmeans_vel.fit_predict(velocity_data)

# Visualize the results in a single plot
plt.figure(figsize=(18, 6))

# Plot combined velocity clusters
plt.scatter(range(len(velocity_data)), velocity_data[:, 0], c=vel_clusters, cmap='viridis', alpha=0.7, label='X Velocity')
plt.scatter(range(len(velocity_data)), velocity_data[:, 1], c=vel_clusters, cmap='plasma', alpha=0.7, label='Y Velocity')
plt.scatter(range(len(velocity_data)), velocity_data[:, 2], c=vel_clusters, cmap='inferno', alpha=0.7, label='Z Velocity')

# Mark anomalies
plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 0], color='red', label='Anomalies (X)', edgecolor='black')
plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 1], color='red', label='Anomalies (Y)', edgecolor='black')
plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 2], color='red', label='Anomalies (Z)', edgecolor='black')

plt.title('Combined Velocity Clusters (X, Y, Z)')
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combine accelerations for all axes into a single array
acceleration_data = np.vstack((ax, ay, az)).T  # Shape: (n_samples, 3)

# Apply KMeans clustering on the combined acceleration data
kmeans_acc = KMeans(n_clusters=1, random_state=42)
acc_clusters = kmeans_acc.fit_predict(acceleration_data)

# Visualize the results in a single plot
plt.figure(figsize=(18, 6))

# Plot combined acceleration clusters
plt.scatter(range(len(acceleration_data)), acceleration_data[:, 0], c=acc_clusters, cmap='viridis', alpha=0.7, label='X Acceleration')
plt.scatter(range(len(acceleration_data)), acceleration_data[:, 1], c=acc_clusters, cmap='plasma', alpha=0.7, label='Y Acceleration')
plt.scatter(range(len(acceleration_data)), acceleration_data[:, 2], c=acc_clusters, cmap='inferno', alpha=0.7, label='Z Acceleration')

# Mark anomalies
plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 0], color='red', label='Anomalies (X)', edgecolor='black')
plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 1], color='red', label='Anomalies (Y)', edgecolor='black')
plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 2], color='red', label='Anomalies (Z)', edgecolor='black')

plt.title('Combined Acceleration Clusters (X, Y, Z)')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Visualize the results
plt.figure(figsize=(12, 8))

# X, Y and Z as a functino of frames
plt.figure(figsize=(12, 8))
plt.plot(x[:-2], label='X Position')
plt.scatter(anomaly_indices, x[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (X)')

plt.figure(figsize=(12, 8))
plt.plot(y[:-2], label='Y Position')
plt.scatter(anomaly_indices, y[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Y)')

plt.figure(figsize=(12, 8))
plt.plot(z[:-2], label='Z Position')
plt.scatter(anomaly_indices, z[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Position with Anomalies (Z)')

plt.tight_layout()
plt.show()





# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3D Visualization of Anomalies in Motion Data
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normal points
normal_indices = np.where(anomaly_scores == 1)[0]
ax.scatter(motion_data[normal_indices, 0], motion_data[normal_indices, 1], motion_data[normal_indices, 2],
           c='blue', label='Normal', alpha=0.6)

# Anomaly points
ax.scatter(motion_data[anomaly_indices, 0], motion_data[anomaly_indices, 1], motion_data[anomaly_indices, 2],
           c='red', label='Anomalies', alpha=0.9)

# Labels and legend
ax.set_title('3D Visualization of Anomalies', fontsize=14)
ax.set_xlabel('X (m/s)')
ax.set_ylabel('Y (m/s)')
ax.set_zlabel('Z (m/s)')
ax.legend()

plt.show()





# Create a figure for plotting X Velocity and X Acceleration
plt.figure(figsize=(15, 8))

# Plot X velocity
plt.subplot(2, 1, 1)
plt.plot(vx, label='X Velocity', color='blue')
plt.scatter(anomaly_indices, vx[anomaly_indices], color='red', label='Anomalies (X)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.title('X Velocity with Anomalies')
plt.grid(True)

# Plot X acceleration
plt.subplot(2, 1, 2)
plt.plot(ax, label='X Acceleration', color='green')
plt.scatter(anomaly_indices[:-1], ax[anomaly_indices[:-1]], color='red', label='Anomalies (X)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.title('X Acceleration with Anomalies')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()





# Create a figure for plotting velocities
plt.figure(figsize=(15, 8))

# Plot velocities for each dimension (X, Y, Z) in separate subplots
plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
plt.plot(vx, label='X Velocity', color='blue')
plt.scatter(anomaly_indices, vx[anomaly_indices], color='red', label='Anomalies (X)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.title('X Velocity with Anomalies')
plt.grid(True)

plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
plt.plot(vy, label='Y Velocity', color='green')
plt.scatter(anomaly_indices, vy[anomaly_indices], color='red', label='Anomalies (Y)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.title('Y Velocity with Anomalies')
plt.grid(True)

plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
plt.plot(vz, label='Z Velocity', color='orange')
plt.scatter(anomaly_indices, vz[anomaly_indices], color='red', label='Anomalies (Z)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.title('Z Velocity with Anomalies')
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Create a figure for plotting accelerations
plt.figure(figsize=(15, 8))

# Plot accelerations for each dimension (X, Y, Z) in separate subplots
plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
plt.plot(ax, label='X Acceleration', color='blue')
plt.scatter(anomaly_indices[:-1], ax[anomaly_indices[:-1]], color='red', label='Anomalies (X)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.title('X Acceleration with Anomalies')
plt.grid(True)

plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
plt.plot(ay, label='Y Acceleration', color='green')
plt.scatter(anomaly_indices[:-1], ay[anomaly_indices[:-1]], color='red', label='Anomalies (Y)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.title('Y Acceleration with Anomalies')
plt.grid(True)

plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
plt.plot(az, label='Z Acceleration', color='orange')
plt.scatter(anomaly_indices[:-1], az[anomaly_indices[:-1]], color='red', label='Anomalies (Z)', edgecolor='black')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.title('Z Acceleration with Anomalies')
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()



for i, (dim, vel, acc, color) in enumerate(zip(dimensions, velocities, accelerations, colors)):
    plt.figure(figsize=(15, 12))
    
    # Plot velocity
    plt.plot(vel, label=f'{dim} Velocity', color=color)
    plt.scatter(anomaly_indices, vel[anomaly_indices], color='red', label='Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'{dim}-Axis Velocity with Anomalies')
    plt.legend()
    plt.grid(True)
    
    # Plot acceleration
    plt.subplot(2, 1, 2)
    plt.plot(acc, label=f'{dim} Acceleration', color=color)
    plt.scatter(anomaly_indices[:-1], acc[anomaly_indices[:-1]], color='red', label='Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{dim}-Axis Acceleration with Anomalies')
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