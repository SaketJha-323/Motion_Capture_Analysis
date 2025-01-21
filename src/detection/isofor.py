# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load the calibrated drone data
# file_path = 'data/raw/Drone_CoD.csv'
# drone_data = pd.read_csv(file_path)

# # Ensure the columns are named correctly: 'Frame', 'Time', 'X', 'Y', 'Z'
# drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# # Convert data to numpy array for easier manipulation
# frames = drone_data['Frame'].values
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

# # Calculate velocities
# vx = calculate_velocity(x, times)
# vy = calculate_velocity(y, times)
# vz = calculate_velocity(z, times)

# # Calculate accelerations
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

# # Apply Isolation Forest for anomaly detection
# isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.3, random_state=42)
# anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)

# # Find indices of anomalies
# anomaly_indices = np.where(anomaly_scores == -1)[0]

# # Visualize the results
# dimensions = ['X', 'Y', 'Z']
# velocities = [vx, vy, vz]
# accelerations = [ax, ay, az]
# colors = ['blue', 'green', 'purple']

# # Combine velocities for all axes into a single array
# velocity_data = np.vstack((vx, vy, vz)).T  # Shape: (n_samples, 3)

# # Apply KMeans clustering on the combined velocity data
# kmeans_vel = KMeans(n_clusters=1, random_state=42)
# vel_clusters = kmeans_vel.fit_predict(velocity_data)

# # Visualize the results in a single plot
# plt.figure(figsize=(18, 6))

# # Plot combined velocity clusters
# plt.scatter(range(len(velocity_data)), velocity_data[:, 0], c=vel_clusters, cmap='viridis', alpha=0.7, label='X Velocity')
# plt.scatter(range(len(velocity_data)), velocity_data[:, 1], c=vel_clusters, cmap='plasma', alpha=0.7, label='Y Velocity')
# plt.scatter(range(len(velocity_data)), velocity_data[:, 2], c=vel_clusters, cmap='inferno', alpha=0.7, label='Z Velocity')

# # Mark anomalies
# plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 0], color='red', label='Anomalies (X)', edgecolor='black')
# plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 1], color='red', label='Anomalies (Y)', edgecolor='black')
# plt.scatter(anomaly_indices, velocity_data[anomaly_indices, 2], color='red', label='Anomalies (Z)', edgecolor='black')

# plt.title('Combined Velocity Clusters (X, Y, Z)')
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Combine accelerations for all axes into a single array
# acceleration_data = np.vstack((ax, ay, az)).T  # Shape: (n_samples, 3)

# # Apply KMeans clustering on the combined acceleration data
# kmeans_acc = KMeans(n_clusters=1, random_state=42)
# acc_clusters = kmeans_acc.fit_predict(acceleration_data)

# # Visualize the results in a single plot
# plt.figure(figsize=(18, 6))

# # Plot combined acceleration clusters
# plt.scatter(range(len(acceleration_data)), acceleration_data[:, 0], c=acc_clusters, cmap='viridis', alpha=0.7, label='X Acceleration')
# plt.scatter(range(len(acceleration_data)), acceleration_data[:, 1], c=acc_clusters, cmap='plasma', alpha=0.7, label='Y Acceleration')
# plt.scatter(range(len(acceleration_data)), acceleration_data[:, 2], c=acc_clusters, cmap='inferno', alpha=0.7, label='Z Acceleration')

# # Mark anomalies
# plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 0], color='red', label='Anomalies (X)', edgecolor='black')
# plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 1], color='red', label='Anomalies (Y)', edgecolor='black')
# plt.scatter(anomaly_indices[:-1], acceleration_data[anomaly_indices[:-1], 2], color='red', label='Anomalies (Z)', edgecolor='black')

# plt.title('Combined Acceleration Clusters (X, Y, Z)')
# plt.xlabel('Time Steps')
# plt.ylabel('Acceleration (m/s²)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Visualize the results
# plt.figure(figsize=(12, 8))

# # X, Y and Z as a functino of frames
# plt.figure(figsize=(12, 8))
# plt.plot(x[:-2], label='X Position')
# plt.scatter(anomaly_indices, x[anomaly_indices], color='red', label='Anomalies')
# plt.legend()
# plt.title('Position with Anomalies (X)')

# plt.figure(figsize=(12, 8))
# plt.plot(y[:-2], label='Y Position')
# plt.scatter(anomaly_indices, y[anomaly_indices], color='red', label='Anomalies')
# plt.legend()
# plt.title('Position with Anomalies (Y)')

# plt.figure(figsize=(12, 8))
# plt.plot(z[:-2], label='Z Position')
# plt.scatter(anomaly_indices, z[anomaly_indices], color='red', label='Anomalies')
# plt.legend()
# plt.title('Position with Anomalies (Z)')

# plt.tight_layout()
# plt.show()





# # Adjust layout
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # 3D Visualization of Anomalies in Motion Data
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Normal points
# normal_indices = np.where(anomaly_scores == 1)[0]
# ax.scatter(motion_data[normal_indices, 0], motion_data[normal_indices, 1], motion_data[normal_indices, 2],
#            c='blue', label='Normal', alpha=0.6)

# # Anomaly points
# ax.scatter(motion_data[anomaly_indices, 0], motion_data[anomaly_indices, 1], motion_data[anomaly_indices, 2],
#            c='red', label='Anomalies', alpha=0.9)

# # Labels and legend
# ax.set_title('3D Visualization of Anomalies', fontsize=14)
# ax.set_xlabel('X (m/s)')
# ax.set_ylabel('Y (m/s)')
# ax.set_zlabel('Z (m/s)')
# ax.legend()

# plt.show()





# # Create a figure for plotting X Velocity and X Acceleration
# plt.figure(figsize=(15, 8))

# # Plot X velocity
# plt.subplot(2, 1, 1)
# plt.plot(vx, label='X Velocity', color='blue')
# plt.scatter(anomaly_indices, vx[anomaly_indices], color='red', label='Anomalies (X)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.title('X Velocity with Anomalies')
# plt.grid(True)

# # Plot X acceleration
# plt.subplot(2, 1, 2)
# plt.plot(ax, label='X Acceleration', color='green')
# plt.scatter(anomaly_indices[:-1], ax[anomaly_indices[:-1]], color='red', label='Anomalies (X)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Acceleration (m/s²)')
# plt.title('X Acceleration with Anomalies')
# plt.grid(True)

# # Show the plot
# plt.tight_layout()
# plt.show()





# # Create a figure for plotting velocities
# plt.figure(figsize=(15, 8))

# # Plot velocities for each dimension (X, Y, Z) in separate subplots
# plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
# plt.plot(vx, label='X Velocity', color='blue')
# plt.scatter(anomaly_indices, vx[anomaly_indices], color='red', label='Anomalies (X)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.title('X Velocity with Anomalies')
# plt.grid(True)

# plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
# plt.plot(vy, label='Y Velocity', color='green')
# plt.scatter(anomaly_indices, vy[anomaly_indices], color='red', label='Anomalies (Y)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.title('Y Velocity with Anomalies')
# plt.grid(True)

# plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
# plt.plot(vz, label='Z Velocity', color='orange')
# plt.scatter(anomaly_indices, vz[anomaly_indices], color='red', label='Anomalies (Z)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity (m/s)')
# plt.title('Z Velocity with Anomalies')
# plt.grid(True)

# # Adjust layout
# plt.tight_layout()
# plt.show()

# # Create a figure for plotting accelerations
# plt.figure(figsize=(15, 8))

# # Plot accelerations for each dimension (X, Y, Z) in separate subplots
# plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
# plt.plot(ax, label='X Acceleration', color='blue')
# plt.scatter(anomaly_indices[:-1], ax[anomaly_indices[:-1]], color='red', label='Anomalies (X)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Acceleration (m/s²)')
# plt.title('X Acceleration with Anomalies')
# plt.grid(True)

# plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
# plt.plot(ay, label='Y Acceleration', color='green')
# plt.scatter(anomaly_indices[:-1], ay[anomaly_indices[:-1]], color='red', label='Anomalies (Y)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Acceleration (m/s²)')
# plt.title('Y Acceleration with Anomalies')
# plt.grid(True)

# plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
# plt.plot(az, label='Z Acceleration', color='orange')
# plt.scatter(anomaly_indices[:-1], az[anomaly_indices[:-1]], color='red', label='Anomalies (Z)', edgecolor='black')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('Acceleration (m/s²)')
# plt.title('Z Acceleration with Anomalies')
# plt.grid(True)

# # Adjust layout
# plt.tight_layout()
# plt.show()



# for i, (dim, vel, acc, color) in enumerate(zip(dimensions, velocities, accelerations, colors)):
#     plt.figure(figsize=(15, 12))
    
#     # Plot velocity
#     plt.plot(vel, label=f'{dim} Velocity', color=color)
#     plt.scatter(anomaly_indices, vel[anomaly_indices], color='red', label='Anomalies')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Velocity (m/s)')
#     plt.title(f'{dim}-Axis Velocity with Anomalies')
#     plt.legend()
#     plt.grid(True)
    
#     # Plot acceleration
#     plt.subplot(2, 1, 2)
#     plt.plot(acc, label=f'{dim} Acceleration', color=color)
#     plt.scatter(anomaly_indices[:-1], acc[anomaly_indices[:-1]], color='red', label='Anomalies')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Acceleration (m/s²)')
#     plt.title(f'{dim}-Axis Acceleration with Anomalies')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# # Create a combined superimposed plot for all dimensions
# plt.figure(figsize=(15, 10))

# # Normalize all velocities and accelerations
# vx_norm = (vx - np.mean(vx)) / np.std(vx)
# vy_norm = (vy - np.mean(vy)) / np.std(vy)
# vz_norm = (vz - np.mean(vz)) / np.std(vz)
# ax_norm = (ax - np.mean(ax)) / np.std(ax)
# ay_norm = (ay - np.mean(ay)) / np.std(ay)
# az_norm = (az - np.mean(az)) / np.std(az)

# # Plot velocities
# plt.subplot(2, 1, 1)
# plt.plot(vx_norm, label='X Velocity', color='blue', alpha=0.7)
# plt.plot(vy_norm, label='Y Velocity', color='green', alpha=0.7)
# plt.plot(vz_norm, label='Z Velocity', color='purple', alpha=0.7)
# plt.scatter(anomaly_indices, vx_norm[anomaly_indices], color='red', marker='o', label='Anomalies')
# plt.xlabel('Time Steps')
# plt.ylabel('Normalized Velocity')
# plt.title('Normalized Velocities Comparison (All Axes)')
# plt.legend()
# plt.grid(True)

# # Plot accelerations
# plt.subplot(2, 1, 2)
# plt.plot(ax_norm, label='X Acceleration', color='blue', alpha=0.7)
# plt.plot(ay_norm, label='Y Acceleration', color='green', alpha=0.7)
# plt.plot(az_norm, label='Z Acceleration', color='purple', alpha=0.7)
# plt.scatter(anomaly_indices[:-1], ax_norm[anomaly_indices[:-1]], color='red', marker='o', label='Anomalies')
# plt.xlabel('Time Steps')
# plt.ylabel('Normalized Acceleration')
# plt.title('Normalized Accelerations Comparison (All Axes)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Print statistics for all dimensions
# for dim, vel, acc in zip(dimensions, velocities, accelerations):
#     print(f"\n{dim}-Axis Statistics:")
#     print(f"Average Velocity: {np.mean(vel):.2f} m/s")
#     print(f"Average Acceleration: {np.mean(acc):.2f} m/s²")
#     print(f"Velocity Standard Deviation: {np.std(vel):.2f} m/s")
#     print(f"Acceleration Standard Deviation: {np.std(acc):.2f} m/s²")

# # Print detected anomalies
# print("Anomalies detected at indices:", anomaly_indices)







import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/raw/Drone_CoD.csv'
drone_data = pd.read_csv(file_path)

# Ensure columns are named correctly
drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# Convert data to numpy array for easier manipulation
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

# Calculate velocities and accelerations
vx = calculate_velocity(x, times)
vy = calculate_velocity(y, times)
vz = calculate_velocity(z, times)
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

# Fit Isolation Forest with a range of contamination values and examine anomaly scores
contamination_values = np.linspace(0.01, 0.5, 50)  # Range of contamination values to test
anomaly_scores_by_contamination = []

# Loop through contamination values to see how they affect the model
for contamination in contamination_values:
    isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=contamination, random_state=42)
    anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)
    anomaly_scores_by_contamination.append(anomaly_scores)

# Plot the effect of contamination on anomaly classification
plt.figure(figsize=(12, 6))
for idx, contamination in enumerate(contamination_values):
    plt.plot(motion_data_scaled[:, 0], anomaly_scores_by_contamination[idx], label=f'Contamination: {contamination:.2f}')
    print(contamination_values)

plt.xlabel('Data Points')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Scores for Different Contamination Levels')
plt.legend(loc='upper right')
plt.show()

# You can also plot the overall anomaly score distribution to visually inspect
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.3, random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)
plt.hist(anomaly_scores, bins=50, color='blue', edgecolor='black')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()







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
