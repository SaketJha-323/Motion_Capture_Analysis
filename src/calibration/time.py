import pandas as pd

# Load the dataset
data = pd.read_csv('data/raw/Drone_CoD.csv')

# Subtract the first time value from all time values
data['Time'] = data['Time'] - data['Time'].iloc[0]

# Show the updated dataframe
print(data.head())

# Optionally, save the updated data to a new CSV file
data.to_csv('data/processed/updated_drone_data.csv', index=False)
