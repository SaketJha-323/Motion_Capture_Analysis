import pandas as pd

# Load the drone data from the new CSV file
file_path = 'data/processed/updated_drone_data2.csv'
drone_data = pd.read_csv(file_path)

# Print column names to verify
print("Column names in the DataFrame:")
print(drone_data.columns)

# Assuming the first row contains the headers and the relevant data starts from the second row
# Extract relevant columns
drone_movement_data = drone_data[['X', 'Y', 'Z']]

# Convert to numeric and drop any rows with NaNs
drone_movement_data = drone_movement_data.apply(pd.to_numeric, errors='coerce').dropna()

# Calculate the differences between consecutive rows for X, Y, Z coordinates
space_resolution = drone_movement_data.diff().dropna()

# Calculate average space resolution in meters
avg_space_resolution = space_resolution.abs().mean()

# Calculate time resolution based on the 'Time' column
time_resolution = drone_data['Time'].diff().dropna().mean()

print(f"Average Space Resolution (meters):\n{avg_space_resolution}")
print(f"Time Resolution (seconds): {time_resolution}")
