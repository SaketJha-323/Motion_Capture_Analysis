import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# Load the data (assuming 'calibrated_drone_data.csv' exists)
file_path = 'data/raw/Drone_CoD.csv'
drone_data = pd.read_csv(file_path)


print(drone_data.head())

# Room dimensions (inches)
room_length = 255
room_width = 200
room_height = 90

# Convert room dimensions to meters (1 inch = 0.0254 meters)
room_length_meters = room_length * 0.0254
room_width_meters = room_width * 0.0254
room_height_meters = room_height * 0.0254

# Calibrate drone data (assuming data is in inches)
drone_data['X'] = drone_data['X'] - drone_data['X'].mean() + room_length / 2
drone_data['Y'] = drone_data['Y'] - drone_data['Y'].mean() + room_width / 2
drone_data['Z'] = drone_data['Z'] - drone_data['Z'].min() + room_height / 2

# Convert drone data to meters
drone_data['X'] = drone_data['X'] * 0.0254
drone_data['Y'] = drone_data['Y'] * 0.0254
drone_data['Z'] = drone_data['Z'] * 0.0254

# Create a 3D plot for the drone's movement
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the drone movement in 3D with adjusted colors
ax.plot(drone_data['X'], drone_data['Y'], drone_data['Z'], label='Drone Path (Calibrated)', color='g')

# Set limits slightly larger than room dimensions for better visualization
ax.set_xlim([-room_length_meters/2 + 25, room_length_meters/2 - 25])
ax.set_ylim([-room_width_meters/2 + 25, room_width_meters/2 - 25])
ax.set_zlim([-room_height_meters/2 + 25, room_height_meters/2 - 25])

# Set the viewing angle
ax.view_init(elev=30, azim=120)

# Labels and title
ax.set_xlabel('X Coordinate (Meters)')
ax.set_ylabel('Y Coordinate (Meters)')
ax.set_zlabel('Z Coordinate (Meters)')
ax.set_title('3D Representation of Calibrated Drone Movement (Starting Point: Center)')

# Display the plot
plt.legend()
plt.show()