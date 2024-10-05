import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the calibrated drone data
file_path = 'data/raw/Drone_CoD.csv'
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly: 'Frame', 'Time', 'X', 'Y', 'Z'
drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# Convert data to numpy array for easier manipulation
data = np.array(drone_data)

# Extract columns
frames = data[:, 0]
times = data[:, 1]
x = data[:, 2] / 10  # Convert mm to cm
y = data[:, 3] / 10  # Convert mm to cm
z = data[:, 4] / 10  # Convert mm to cm

# Calculate velocities (vx, vy, vz) between consecutive points
def calculate_velocity(x, y, z, times):
    vx = np.diff(x) / np.diff(times)
    vy = np.diff(y) / np.diff(times)
    vz = np.diff(z) / np.diff(times)
    return vx, vy, vz

# Calculate accelerations (ax, ay, az) between consecutive points
def calculate_acceleration(vx, vy, vz, times):
    ax = np.diff(vx) / np.diff(times[1:])
    ay = np.diff(vy) / np.diff(times[1:])
    az = np.diff(vz) / np.diff(times[1:])
    return ax, ay, az

# Calculate velocities
vx, vy, vz = calculate_velocity(x, y, z, times)

# Calculate accelerations
ax, ay, az = calculate_acceleration(vx, vy, vz, times)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(x)
plt.title("X Position")

plt.subplot(3, 2, 2)
plt.plot(y)
plt.title("Y Position")

plt.subplot(3, 2, 3)
plt.plot(vx)
plt.title("Velocity X")

plt.subplot(3, 2, 4)
plt.plot(vy)
plt.title("Velocity Y")

plt.subplot(3, 2, 5)
plt.plot(ax)
plt.title("Acceleration X")

plt.subplot(3, 2, 6)
plt.plot(ay)
plt.title("Acceleration Y")

plt.tight_layout()
plt.show()