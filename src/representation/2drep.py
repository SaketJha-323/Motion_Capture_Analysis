import matplotlib.pyplot as plt
import pandas as pd

file_path = 'data/raw/Drone_CoD.csv'
drone_data = pd.read_csv(file_path)


# Attempt to convert columns to numeric, forcing errors to NaN, then drop those rows
drone_movement_data = pd.to_numeric(drone_data.iloc[6:, 1:4].stack(), errors='coerce').unstack().dropna()

# Assign the column names correctly
drone_movement_data.columns = ['X', 'Y', 'Z']

print(drone_movement_data)
# Create a 2D plot (X vs Y, X vs Z, and Y vs Z)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# X vs Y
axs[0].plot(drone_movement_data['X'], drone_movement_data['Y'], color='r')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].set_title('X vs Y')

# X vs Z
axs[1].plot(drone_movement_data['X'], drone_movement_data['Z'], color='g')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Z Coordinate')
axs[1].set_title('X vs Z')

# Y vs Z
axs[2].plot(drone_movement_data['Y'], drone_movement_data['Z'], color='b')
axs[2].set_xlabel('Y Coordinate')
axs[2].set_ylabel('Z Coordinate')
axs[2].set_title('Y vs Z')

# Display the 2D plots
plt.show()
