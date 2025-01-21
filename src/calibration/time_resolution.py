import pandas as pd

# Load the dataset
data = pd.read_csv('data/processed/updated_drone_data.csv')

# Calculate the time differences between consecutive rows
time_diff = data['Time'].diff().dropna()

# Calculate the minimum time difference (time resolution)
time_resolution = time_diff.min()

# Calculate the maximum time difference (to check for any irregularities)
max_time_diff = time_diff.max()

print(f"Time Resolution (Smallest Time Difference): {time_resolution} seconds")
print(f"Maximum Time Difference: {max_time_diff} seconds")
