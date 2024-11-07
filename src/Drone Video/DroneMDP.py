import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the video file
video_path = 'data/video/Drone2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Initialize the tracker
# For OpenCV 4.5 and newer, use the cv2.legacy module if needed
try:
    tracker = cv2.TrackerMIL_create()  # Ensure this method is available in your version
except AttributeError:
    print("TrackerMIL_create() method is not available. Try using an alternative tracker.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Define the initial bounding box
# If cv2.selectROI() is not working, you can manually define bbox
bbox = (100, 100, 200, 200)  # Example: (x, y, width, height)
# Uncomment the following line if you want to use GUI ROI selection
# bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

# Variables to store data
frame_numbers = []
x_positions = []
y_positions = []

# Process video frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)
    
    if success:
        # Extract the center of the bounding box as the object's position
        x_center = int(bbox[0] + bbox[2] / 2)
        y_center = int(bbox[1] + bbox[3] / 2)
        frame_numbers.append(frame_count)
        x_positions.append(x_center)
        y_positions.append(y_center)
    
    frame_count += 1

cap.release()

# Plotting the extracted data
plt.figure(figsize=(12, 6))

# Plot X and Y positions
plt.subplot(2, 1, 1)
plt.plot(frame_numbers, x_positions, label='X Position', color='blue')
plt.xlabel('Frame Number')
plt.ylabel('X Position (pixels)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(frame_numbers, y_positions, label='Y Position', color='red')
plt.xlabel('Frame Number')
plt.ylabel('Y Position (pixels)')
plt.legend()

plt.tight_layout()
plt.show()
