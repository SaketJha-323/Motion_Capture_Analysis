import pandas as pd
import numpy as np

# Load the calibrated drone data
file_path = 'data/raw/Drone_CoD.csv'
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly: 'Frame', 'Time', 'X', 'Y', 'Z'
# Sample column names based on your provided data
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

#Print results
print("Position (cm):")
for i in range(len(x)):
    print(f"Frame {frames[i]}: X = {x[i]:.2f} cm, Y = {y[i]:.2f} cm, Z = {z[i]:.2f} cm")

print("\nVelocity (cm/s):")
for i in range(len(vx)):
    print(f"Vx = {vx[i]:.2f} cm/s, Vy = {vy[i]:.2f} cm/s, Vz = {vz[i]:.2f} cm/s")

print("\nAcceleration (cm/s^2):")
for i in range(len(ax)):
    print(f"Ax = {ax[i]:.2f} cm/s^2, Ay = {ay[i]:.2f} cm/s^2, Az = {az[i]:.2f} cm/s^2")

#Calculate average velocity
avg_vx = np.mean(vx)
avg_vy = np.mean(vy)
avg_vz = np.mean(vz)

# Calculate average acceleration
avg_ax = np.mean(ax)
avg_ay = np.mean(ay)
avg_az = np.mean(az)

# Print the averages
print(f"Average Velocity: Vx = {avg_vx:.2f} cm/s, Vy = {avg_vy:.2f} cm/s, Vz = {avg_vz:.2f} cm/s")
print(f"Average Acceleration: Ax = {avg_ax:.2f} cm/s^2, Ay = {avg_ay:.2f} cm/s^2, Az = {avg_az:.2f} cm/s^2")