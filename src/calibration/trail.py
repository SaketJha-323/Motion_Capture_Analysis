import pandas as pd
import matplotlib.pyplot as plt

# Assuming the first row is a header
df = pd.read_csv('Drone_CoD.csv')

print(df)

# Extract the 3D position data
x_data = df['X']
y_data = df['Y']
z_data = df['Z']

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_data, y_data, z_data)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Drone 3D Trajectory')
plt.show()