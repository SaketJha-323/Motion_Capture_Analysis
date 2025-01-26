import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = 'data/processed/updated_drone_data2.csv'
drone_data = pd.read_csv(file_path)

# # Ensure the columns are named correctly
# drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

#convert position into meters
drone_data['X'] /= 1000
drone_data['Y'] /= 1000
drone_data['Z'] /= 1000

#calculate time differences
time_diff = np.gradient(drone_data['Time'])

#calculate velocity from position
drone_data['vx'] = np.gradient(drone_data['X'], drone_data['Time'])
drone_data['vy'] = np.gradient(drone_data['Y'], drone_data['Time'])
drone_data['vz'] = np.gradient(drone_data['Z'], drone_data['Time'])

#calculate acceleration from velocity
drone_data['ax'] = np.gradient(drone_data['vx'], drone_data['Time'])
drone_data['ay'] = np.gradient(drone_data['vy'], drone_data['Time'])
drone_data['az'] = np.gradient(drone_data['vz'], drone_data['Time'])

# # Convert data to numpy array for easier manipulation
# frames = drone_data['Frame'].values
# times = drone_data['Time'].values
# x = drone_data['X'].values / 1000  # Convert from mm to meters
# y = drone_data['Y'].values / 1000
# z = drone_data['Z'].values / 1000

# # Function to calculate a moving average
# def moving_average(data, window_size):
#     return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# # Apply moving average with a window size of 10
# window_size = 10
# x_avg = moving_average(x, window_size)
# y_avg = moving_average(y, window_size)
# z_avg = moving_average(z, window_size)
# times_avg = moving_average(times, window_size)  # Average times for alignment

# # Functions to calculate velocity and acceleration
# def calculate_velocity(position):
#     return np.diff(position) / dt

# def calculate_acceleration(velocity):
#     return np.diff(velocity) / dt

# # Calculate velocities and accelerations using the averaged data
# vx = calculate_velocity(x_avg, times_avg)
# vy = calculate_velocity(y_avg, times_avg)
# vz = calculate_velocity(z_avg, times_avg)

# ax = calculate_acceleration(vx, times_avg)
# ay = calculate_acceleration(vy, times_avg)
# az = calculate_acceleration(vz, times_avg)


# # Calculate velocities and accelerations using the averaged data
# vx = calculate_velocity(x)
# vy = calculate_velocity(y)
# vz = calculate_velocity(z)

# ax = calculate_acceleration(vx)
# ay = calculate_acceleration(vy)
# az = calculate_acceleration(vz)

# dt = 0.00333
# vx = np.diff(x) / dt
# vy = np.diff(y) / dt
# vz = np.diff(z) / dt

# ax = np.diff(vx) / dt
# ay = np.diff(vy) / dt
# az = np.diff(vz) / dt


# Trim all arrays to the same size
# min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
# vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
# ax, ay, az = ax[:min_length], ay[:min_length], az[:min_length]

# # Trim the times array to match the length of the velocity data
# times_trimmed = times_avg[1:len(vx) + 1]  # Ensure times aligns with velocity data length

# Trim the times array to match the length of the velocity data
# times_trimmed = times[1:len(vx) + 1]  # Ensure times aligns with velocity data length

# Combine motion data for anomaly detection
# motion_data = np.vstack((vx, vy, vz, ax, ay, az)).T
motion_data = drone_data[['vx','vy','vz','ax','ay','az']].T

# # Normalize data using StandardScaler
# scaler = StandardScaler()
# motion_data_scaled = scaler.fit_transform(motion_data)

# Apply Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.1, random_state=42)
# anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)
anomaly_scores = isolation_forest.fit_predict(motion_data)

# Identify anomaly indices
anomaly_indices = np.where(anomaly_scores == -1)[0]

# # Visualize velocities and accelerations with anomalies
# def plot_with_anomalies(data, anomalies, labels, title, ylabel, times):
#     plt.figure(figsize=(15, 8))
#     for i, (label, color) in enumerate(labels):
#         plt.plot(times, data[:, i], label=f'{label}', color=color, alpha=0.7)
#         plt.scatter(times[anomalies], data[anomalies, i], color='red', label=f'Anomalies ({label})', edgecolor='black')
#     plt.title(title)
#     plt.xlabel('Time (s)')
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Plot velocities
# velocity_data = np.vstack((vx, vy, vz)).T
# plot_with_anomalies(velocity_data, anomaly_indices, 
#                     [('X Velocity', 'blue'), ('Y Velocity', 'green'), ('Z Velocity', 'purple')], 
#                     'Velocities with Anomalies', 'Velocity (m/s)', times_trimmed)

# # Plot accelerations
# acceleration_data = np.vstack((ax, ay, az)).T
# plot_with_anomalies(acceleration_data, anomaly_indices[:-1], 
#                     [('X Acceleration', 'blue'), ('Y Acceleration', 'green'), ('Z Acceleration', 'purple')], 
#                     'Accelerations with Anomalies', 'Acceleration (m/s²)', times_trimmed)


# # Set the Seaborn style for better aesthetics
# sns.set_theme(style="whitegrid", palette="muted")

# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['vx'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(anomaly_indices, drone_data['vx'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X Velocity with Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


# Subplot for X acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['ax'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(anomaly_indices[:-1], drone_data['ax'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X Acceleration with Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


# # Subplot for Y velocity
# plt.subplot(3, 1, 2)
# plt.plot(vy, label='Y Velocity', color='green', alpha=0.8)
# plt.scatter(anomaly_indices, vy[anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
# plt.title('Y Velocity with Anomalies')
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # Subplot for Z velocity
# plt.subplot(3, 1, 3)
# plt.plot(vz, label='Z Velocity', color='purple', alpha=0.8)
# plt.scatter(anomaly_indices, vz[anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
# plt.title('Z Velocity with Anomalies')
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()

# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X acceleration
plt.subplot(3, 1, 1)
plt.plot(ax, label='X Acceleration', color='blue', alpha=0.8)
plt.scatter(anomaly_indices[:-1], ax[anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X Acceleration with Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for Y acceleration
plt.subplot(3, 1, 2)
plt.plot(ay, label='Y Acceleration', color='green', alpha=0.8)
plt.scatter(anomaly_indices[:-1], ay[anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Y Acceleration with Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for Z acceleration
plt.subplot(3, 1, 3)
plt.plot(az, label='Z Acceleration', color='purple', alpha=0.8)
plt.scatter(anomaly_indices[:-1], az[anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Z Acceleration with Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.9)
plt.tight_layout()
plt.show()


# Rename acceleration arrays to avoid conflict with plotting axis
ax_accel, ay_accel, az_accel = ax, ay, az

# 3D Plot of the trajectory with anomalies
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory with anomalies highlighted
ax.plot(x, y, z, c='blue', label='Trajectory', alpha=0.6)

# Add labels and legend
ax.set_title('3D Trajectory with Anomaly Detection', fontsize=16)
ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_zlabel('Z Position (m)', fontsize=12)
ax.legend()

plt.show()


# 3D Visualization of Anomalies
fig = plt.figure(figsize=(10, 8))
ax_3d = fig.add_subplot(111, projection='3d')
normal_indices = np.where(anomaly_scores == 1)[0]

# Plot normal points
ax_3d.scatter(motion_data[normal_indices, 0], motion_data[normal_indices, 1], motion_data[normal_indices, 2], 
              c='blue', label='Normal', alpha=0.6)

# Plot anomaly points
ax_3d.scatter(motion_data[anomaly_indices, 0], motion_data[anomaly_indices, 1], motion_data[anomaly_indices, 2], 
              c='red', label='Anomalies', alpha=0.9)

ax_3d.set_title('3D Visualization of Anomalies', fontsize=14)
ax_3d.set_xlabel('X (m/s)')
ax_3d.set_ylabel('Y (m/s)')
ax_3d.set_zlabel('Z (m/s)')
ax_3d.legend()
plt.show()

# Print statistics for each axis
dimensions = ['X', 'Y', 'Z']
for dim, vel, acc in zip(dimensions, [vx, vy, vz], [ax, ay, az]):
    print(f"\n{dim}-Axis Statistics:")
    print(f"  Average Velocity: {np.mean(vel):.2f} m/s")
    print(f"  Average Acceleration: {np.mean(acc):.2f} m/s²")
    print(f"  Velocity Std Dev: {np.std(vel):.2f} m/s")
    print(f"  Acceleration Std Dev: {np.std(acc):.2f} m/s²")

# Print detected anomalies
print("\nAnomalies detected at indices:", anomaly_indices)







# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # Load the data
# file_path = 'data/raw/Drone_CoD.csv'
# drone_data = pd.read_csv(file_path)

# # Ensure columns are named correctly
# drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# # Convert data to numpy array for easier manipulation
# times = drone_data['Time'].values
# x = drone_data['X'].values / 100  # Convert from mm to meters
# y = drone_data['Y'].values / 100  # Convert from mm to meters
# z = drone_data['Z'].values / 100  # Convert from mm to meters

# # Function to calculate velocity
# def calculate_velocity(x, times):
#     return np.diff(x) / np.diff(times)

# # Function to calculate acceleration
# def calculate_acceleration(vx, times):
#     return np.diff(vx) / np.diff(times[1:])

# # Calculate velocities and accelerations
# vx = calculate_velocity(x, times)
# vy = calculate_velocity(y, times)
# vz = calculate_velocity(z, times)
# ax = calculate_acceleration(vx, times)
# ay = calculate_acceleration(vy, times)
# az = calculate_acceleration(vz, times)

# # Ensure all arrays are the same size by trimming
# min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
# vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
# ax, ay, az = ax[:min_length], ay[:min_length], az[:min_length]

# # Combine velocities and accelerations into a single array for anomaly detection
# motion_data = np.vstack((vx, vy, vz, ax, ay, az)).T

# # Normalize the data for Isolation Forest
# scaler = StandardScaler()
# motion_data_scaled = scaler.fit_transform(motion_data)

# # Fit Isolation Forest with a range of contamination values and examine anomaly scores
# contamination_values = np.linspace(0.01, 0.5, 50)  # Range of contamination values to test
# anomaly_scores_by_contamination = []

# # Loop through contamination values to see how they affect the model
# for contamination in contamination_values:
#     isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=contamination, random_state=42)
#     anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)
#     anomaly_scores_by_contamination.append(anomaly_scores)

# # Plot the effect of contamination on anomaly classification
# plt.figure(figsize=(12, 6))
# for idx, contamination in enumerate(contamination_values):
#     plt.plot(motion_data_scaled[:, 0], anomaly_scores_by_contamination[idx], label=f'Contamination: {contamination:.2f}')
#     print(contamination_values)

# plt.xlabel('Data Points')
# plt.ylabel('Anomaly Score')
# plt.title('Anomaly Scores for Different Contamination Levels')
# plt.legend(loc='upper right')
# plt.show()

# # You can also plot the overall anomaly score distribution to visually inspect
# isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.3, random_state=42)
# anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)
# plt.hist(anomaly_scores, bins=50, color='blue', edgecolor='black')
# plt.title('Distribution of Anomaly Scores')
# plt.xlabel('Anomaly Score')
# plt.ylabel('Frequency')
# plt.show()







# ## performance
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import matplotlib.pyplot as plt

# # Load the data
# file_path = 'data/raw/Drone_CoD.csv'
# drone_data = pd.read_csv(file_path)

# # Ensure columns are named correctly
# drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# # Convert data to numpy array for easier manipulation
# times = drone_data['Time'].values
# x = drone_data['X'].values / 100  # Convert from mm to meters
# y = drone_data['Y'].values / 100  # Convert from mm to meters
# z = drone_data['Z'].values / 100  # Convert from mm to meters

# # Function to calculate velocity
# def calculate_velocity(x, times):
#     return np.diff(x) / np.diff(times)

# # Function to calculate acceleration
# def calculate_acceleration(vx, times):
#     return np.diff(vx) / np.diff(times[1:])

# # Calculate velocities and accelerations
# vx = calculate_velocity(x, times)
# vy = calculate_velocity(y, times)
# vz = calculate_velocity(z, times)
# ax = calculate_acceleration(vx, times)
# ay = calculate_acceleration(vy, times)
# az = calculate_acceleration(vz, times)

# # Ensure all arrays are the same size by trimming
# min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
# vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
# ax, ay, az = ax[:min_length], ay[:min_length], az[:min_length]

# # Combine velocities and accelerations into a single array for anomaly detection
# motion_data = np.vstack((vx, vy, vz, ax, ay, az)).T

# # Normalize the data for Isolation Forest
# scaler = StandardScaler()
# motion_data_scaled = scaler.fit_transform(motion_data)

# # Ground truth labels for evaluation (assuming you have them; 1 for normal, -1 for anomaly)
# # Example: manually labeling anomalies, where 1 is normal and -1 is an anomaly
# # Ideally, this is a predefined array of labeled data.
# # Example ground truth (random for the sake of example; replace with real labels)
# # This needs to be replaced with actual ground truth from your dataset (manually labeled or known anomalies).
# ground_truth_labels = np.zeros(len(motion_data_scaled))
# ground_truth_labels[100:150] = -1  # Assume anomalies are present between 100-150 for this example.

# # Fit Isolation Forest with chosen contamination
# contamination = 0.01
# isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=contamination, random_state=42)
# predicted_labels = isolation_forest.fit_predict(motion_data_scaled)

# # Convert predicted labels to 1 (normal) and -1 (anomaly)
# predicted_labels = np.where(predicted_labels == -1, 1, 0)  # 1 for anomalies, 0 for normal points

# # Convert ground truth labels for evaluation
# ground_truth_labels = np.where(ground_truth_labels == -1, 1, 0)  # 1 for anomalies, 0 for normal points

# # Calculate precision, recall, F1-score, and ROC-AUC
# precision = precision_score(ground_truth_labels, predicted_labels)
# recall = recall_score(ground_truth_labels, predicted_labels)
# f1 = f1_score(ground_truth_labels, predicted_labels)
# roc_auc = roc_auc_score(ground_truth_labels, predicted_labels)

# # Visual inspection and manual labeling for ground truth
# # After plotting, manually label anomalies
# ground_truth_labels = np.zeros(len(motion_data_scaled))

# # Manually labeling a portion of anomalies (this is an example)
# # Suppose anomalies are in the index range 100 to 150 for this example
# ground_truth_labels[100:150] = 1  # Label as anomalies (1 for anomaly)

# # Now, you can use these labeled data to calculate the evaluation metrics.
# # Fit Isolation Forest
# isolation_forest = IsolationForest(n_estimators=300, contamination=0.3, random_state=42)
# predicted_labels = isolation_forest.fit_predict(motion_data_scaled)

# # Convert predicted labels from -1 (anomaly) and 1 (normal) to 1 (anomaly) and 0 (normal)
# predicted_labels = np.where(predicted_labels == -1, 1, 0)  # 1 for anomalies, 0 for normal

# # Calculate evaluation metrics (precision, recall, etc.)
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# precision = precision_score(ground_truth_labels, predicted_labels)
# recall = recall_score(ground_truth_labels, predicted_labels)
# f1 = f1_score(ground_truth_labels, predicted_labels)
# roc_auc = roc_auc_score(ground_truth_labels, predicted_labels)

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")
# print(f"ROC-AUC: {roc_auc:.4f}")


# # Confusion Matrix
# conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Optionally, plot the ROC curve
# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(ground_truth_labels, predicted_labels)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label='ROC curve')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()