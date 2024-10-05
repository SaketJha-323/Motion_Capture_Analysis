import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load the data
file_path = 'data/processed/Drone_Data_meters.csv'
drone_movement_data = pd.read_csv(file_path)

# Room dimensions (in inches)
room_height = 90
room_breadth = 200
room_length = 255

# Calculate the center of the room in X and Y
center_x = room_length / 2
center_y = room_breadth / 2

# Translate the path to center it in the X and Y axes
drone_movement_data['X'] = drone_movement_data['X'] - (drone_movement_data['X'].max() + drone_movement_data['X'].min()) / 2 + center_x
drone_movement_data['Y'] = drone_movement_data['Y'] - (drone_movement_data['Y'].max() + drone_movement_data['Y'].min()) / 2 + center_y

# Scale the path to fit within the room
x_range = drone_movement_data['X'].max() - drone_movement_data['X'].min()
y_range = drone_movement_data['Y'].max() - drone_movement_data['Y'].min()
z_range = drone_movement_data['Z'].max() - drone_movement_data['Z'].min()

scaling_factor = min(room_length / x_range, room_breadth / y_range, room_height / z_range)

drone_movement_data['X'] = (drone_movement_data['X'] - center_x) * scaling_factor + center_x
drone_movement_data['Y'] = (drone_movement_data['Y'] - center_y) * scaling_factor + center_y
drone_movement_data['Z'] = (drone_movement_data['Z'] - drone_movement_data['Z'].min()) * scaling_factor

# Ensure Z starts from 0 after scaling
drone_movement_data['Z'] = drone_movement_data['Z'] - drone_movement_data['Z'].min()

# Create a 3D plot for the drone's movement
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the drone movement in 3D
ax.plot(drone_movement_data['X'], drone_movement_data['Y'], drone_movement_data['Z'], label='Drone Path', color='b')

# Set the limits to match the room dimensions
ax.set_xlim([0, room_length])
ax.set_ylim([0, room_breadth])
ax.set_zlim([0, room_height])

# Set the viewing angle
ax.view_init(elev=30, azim=120)

# Labels and title
ax.set_xlabel('Length (X) [inches]')
ax.set_ylabel('Breadth (Y) [inches]')
ax.set_zlabel('Height (Z) [inches]')
ax.set_title('3D Representation of Drone Movement within the Room')

print(drone_movement_data.columns)

# Display the plot
plt.show()