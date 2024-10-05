import pandas as pd

# Load the drone data from the new CSV file
file_path = 'data/processed/Drone_Data_meters.csv'
drone_data = pd.read_csv(file_path)

# Print column names to verify
print("Column names in the DataFrame:")
print(drone_data.columns)

# Assuming the first row contains the headers and the relevant data starts from the second row
# Extract relevant columns
drone_movement_data = drone_data[['X (m)', 'Y (m)', 'Z (m)']]

# Provided average space resolution in meters
avg_space_resolution = pd.Series({
    'X': 0.001992,
    'Y': 0.005473,
    'Z': 0.020244
})

# Room dimensions in meters
room_dimensions = {
    'X (m)': 255 * 0.0254,  # Convert inches to meters (1 inch = 0.0254 meters)
    'Y (m)': 200 * 0.0254,  # Convert inches to meters
    'Z (m)': 90 * 0.0254    # Convert inches to meters
}

# Calculate scaling factors based on room dimensions and average space resolution
scaling_factors = pd.Series({
    'X (m)': room_dimensions['X (m)'] / avg_space_resolution['X'],
    'Y (m)': room_dimensions['Y (m)'] / avg_space_resolution['Y'],
    'Z (m)': room_dimensions['Z (m)'] / avg_space_resolution['Z']
})

print(scaling_factors)

# Apply scaling factors to calibrate the drone data
calibrated_drone_data = drone_movement_data * scaling_factors
print(calibrated_drone_data)

# Save the calibrated data to a new CSV file
calibrated_file_path = 'data/processed/Drone_Data_meters_S&T_Calibrated.csv'
calibrated_drone_data.to_csv(calibrated_file_path, index=False)

# Display the first few rows of the calibrated data and the path of the saved file
calibrated_drone_data.head(), calibrated_file_path
