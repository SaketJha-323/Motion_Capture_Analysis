import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = 'data/processed/updated_drone_data2.csv'
drone_data = pd.read_csv(file_path)

# Convert position into meters
drone_data['X'] /= 1000
drone_data['Y'] /= 1000
drone_data['Z'] /= 1000

# Calculate time differences
time_diff = np.gradient(drone_data['Time'])

# Calculate velocity from position
drone_data['vx'] = np.gradient(drone_data['X'], drone_data['Time'])
drone_data['vy'] = np.gradient(drone_data['Y'], drone_data['Time'])
drone_data['vz'] = np.gradient(drone_data['Z'], drone_data['Time'])

# Calculate acceleration from velocity
drone_data['ax'] = np.gradient(drone_data['vx'], drone_data['Time'])
drone_data['ay'] = np.gradient(drone_data['vy'], drone_data['Time'])
drone_data['az'] = np.gradient(drone_data['vz'], drone_data['Time'])

# Combine motion data for anomaly detection
motion_data = drone_data[['vx', 'vy', 'vz', 'ax', 'ay', 'az']]

# Apply Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.1, random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data)

# Identify anomaly indices
anomaly_indices = np.where(anomaly_scores == -1)[0]


#for X Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vx'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vx'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for X acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['ax'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['ax'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X-Axis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()


#for Y Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vy'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vy'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Y-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for X acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['ay'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['ay'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Y-AXis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=1)
plt.tight_layout()
plt.show()


#for Z Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vz'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vz'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Z-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for X acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['az'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['az'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Z-Axis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()
