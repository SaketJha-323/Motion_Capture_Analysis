import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the dataset (replace 'your_file.csv' with your file path)
data = pd.read_csv('data/processed/updated_drone_data2.csv')

# Convert positions to meters
data['X'] /= 1000
data['Y'] /= 1000
data['Z'] /= 1000

# Calculate time differences
time_diff = np.gradient(data['Time'])

# Calculate velocity from position
data['vx_calc'] = np.gradient(data['X'], data['Time'])
data['vy_calc'] = np.gradient(data['Y'], data['Time'])
data['vz_calc'] = np.gradient(data['Z'], data['Time'])

# Calculate acceleration from velocity
data['ax_calc'] = np.gradient(data['vx_calc'], data['Time'])
data['ay_calc'] = np.gradient(data['vy_calc'], data['Time'])
data['az_calc'] = np.gradient(data['vz_calc'], data['Time'])

# Combine velocity and acceleration features for anomaly detection
features = data[['vx_calc', 'vy_calc', 'vz_calc', 'ax_calc', 'ay_calc', 'az_calc']]

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # 1% contamination
data['anomaly'] = iso_forest.fit_predict(features)

# Separate normal and anomalous data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Create subplots for X, Y, Z axes
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axes_labels = ['X', 'Y', 'Z']
colors = ['r', 'g', 'b']

for i, axis in enumerate(axes_labels):
    # Velocity subplot
    axes[i, 0].plot(data['Time'], data[f'v{axis.lower()}_calc'], color=colors[i], label=f'v{axis.lower()} (m/s)')
    axes[i, 0].set_ylabel('Velocity (m/s)')
    axes[i, 0].set_title(f'{axis}-Axis Velocity (Calculated) vs Time')
    axes[i, 0].legend()
    axes[i, 0].grid()

    # Acceleration subplot
    axes[i, 1].plot(data['Time'], data[f'a{axis.lower()}_calc'], color=colors[i], label=f'a{axis.lower()} (m/s²)')
    axes[i, 1].set_ylabel('Acceleration (m/s²)')
    axes[i, 1].set_title(f'{axis}-Axis Acceleration (Calculated) vs Time')
    axes[i, 1].legend()
    axes[i, 1].grid()

# Set x-axis label
for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()  
