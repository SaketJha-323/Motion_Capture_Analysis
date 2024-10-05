import pandas as pd

# Load the drone data
file_path = 'Drone_CoD.csv'
drone_data = pd.read_csv(file_path)

# Inspect the raw data to check its structure
print("Raw Drone Data:")
print(drone_data.head())

# Assuming the relevant data starts from the 7th row and the 2nd to 4th columns
drone_movement_data = drone_data.iloc[0:, 2:5]

# Convert to numeric and drop any rows with NaNs
drone_movement_data = pd.to_numeric(drone_movement_data.stack(), errors='coerce').unstack().dropna()


# Inspect the cleaned data
print("Cleaned Drone Movement Data:")
print(drone_movement_data.head())