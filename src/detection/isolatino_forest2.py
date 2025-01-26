import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Read the CSV file
df = pd.read_csv('data/processed/updated_drone_data2.csv')

# Prepare features for anomaly detection
features = ['vx', 'vy', 'vz', 'ax', 'ay', 'az']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)

# Create a figure with 2x3 subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Drone Velocity and Acceleration Components with Anomalies', fontsize=16)

# Plot velocity components
velocity_features = ['vx', 'vy', 'vz']
colors = ['blue', 'green', 'red']

for i, (feature, color) in enumerate(zip(velocity_features, colors)):
    # Top row for velocity
    axs[0, i].scatter(df['Time'], df[feature], 
                      c=np.where(anomaly_labels == -1, 'red', color), 
                      alpha=0.5)
    axs[0, i].set_title(f'Velocity {feature.upper()}')
    axs[0, i].set_xlabel('Time')
    axs[0, i].set_ylabel('Velocity')
    axs[0, i].grid(True)

# Plot acceleration components
acceleration_features = ['ax', 'ay', 'az']

for i, (feature, color) in enumerate(zip(acceleration_features, colors)):
    # Bottom row for acceleration
    axs[1, i].scatter(df['Time'], df[feature], 
                      c=np.where(anomaly_labels == -1, 'red', color), 
                      alpha=0.5)
    axs[1, i].set_title(f'Acceleration {feature.upper()}')
    axs[1, i].set_xlabel('Time')
    axs[1, i].set_ylabel('Acceleration')
    axs[1, i].grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('drone_data_anomalies.png')
plt.close()

# Print anomaly statistics
total_points = len(df)
anomaly_count = np.sum(anomaly_labels == -1)
anomaly_percentage = (anomaly_count / total_points) * 100

print(f"Total data points: {total_points}")
print(f"Number of anomalies detected: {anomaly_count}")
print(f"Percentage of anomalies: {anomaly_percentage:.2f}%")
print("Visualization saved as drone_data_anomalies.png")