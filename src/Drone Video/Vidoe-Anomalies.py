import cv2
import numpy as np
from scipy.spatial import distance
from scipy.signal import savgol_filter
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Adjust for your video source or camera index

# Initialize parameters
previous_position = None
previous_time = None
trajectory_data = []

def calculate_velocity_acceleration(new_position, previous_position, dt):
    # Calculate velocity (distance per unit time)
    velocity = np.array((new_position - previous_position) / dt)
    
    # Calculate acceleration if enough data points are available
    if len(trajectory_data) > 1:
        prev_velocity = trajectory_data[-1]['velocity']
        acceleration = (velocity - prev_velocity) / dt
    else:
        acceleration = np.array([0, 0, 0])  # No acceleration for the first frame
    
    return velocity, acceleration

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect the drone in frame (Here, we assume some object detection mechanism is used)
    # For this example, we’re simply tracking a specific color range
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Assuming your drone is of a specific color (modify this range as necessary)
    mask = cv2.inRange(hsv_frame, (40, 70, 70), (80, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If the drone is detected, calculate its position in 3D space
    if contours:
        # Assume largest contour is the drone
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Dummy Z coordinate (replace with actual depth data from depth camera)
        z = 1000  # Replace with depth information in mm
        
        # Convert (x, y, z) to millimeters based on camera calibration data (this part depends on your camera setup)
        x_mm = x * conversion_factor_x
        y_mm = y * conversion_factor_y
        position = np.array([x_mm, y_mm, z])
        
        # Calculate time delta
        current_time = time.time()
        if previous_time:
            dt = current_time - previous_time
        else:
            dt = 0
        previous_time = current_time
        
        # Calculate velocity and acceleration
        if previous_position is not None:
            velocity, acceleration = calculate_velocity_acceleration(position, previous_position, dt)
        else:
            velocity, acceleration = np.array([0, 0, 0]), np.array([0, 0, 0])
        
        # Append data for trajectory analysis
        trajectory_data.append({
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'timestamp': current_time
        })
        
        previous_position = position
        
        # Display tracking information
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Position (mm): {position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Drone Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Trajectory Analysis: smoothness, accuracy, precision
positions = np.array([data['position'] for data in trajectory_data])

# Smoothing trajectory data for analysis (e.g., Savitzky-Golay filter)
smoothed_positions = savgol_filter(positions, window_length=5, polyorder=2, axis=0)

# Analyze deviations or anomalies in smoothness, accuracy
deviation = np.linalg.norm(positions - smoothed_positions, axis=1)
anomalies = deviation > threshold  # Define a threshold for deviations

print("Trajectory Analysis Complete.")
print("Detected Anomalies:", np.where(anomalies)[0])  # Indices of anomaly frames
