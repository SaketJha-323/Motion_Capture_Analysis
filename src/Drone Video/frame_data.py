import cv2
import numpy as np
from PIL import Image
import os

# Parameters
VIDEO_PATH = "data/video/Drone1.mp4"  # Path to the video
FRAME_OUTPUT_DIR = "data/frames_data/sample_frames1"      # Directory to store extracted frames
GIF_OUTPUT_PATH = "anomalies.gif"  # Output GIF path
FRAMES_AROUND_ANOMALY = 20       # Number of frames before/after anomaly
ANOMALY_THRESHOLD = 50           # Placeholder threshold for anomaly detection

# Ensure output directory exists
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

def extract_frames(video_path, output_dir):
    """Extract frames from the video and save them as images."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames.")

    return frame_idx

def detect_anomalies(frames_dir, total_frames):
    """Detect anomalies in the frames using a placeholder algorithm."""
    anomalies = []

    for i in range(total_frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Placeholder: Calculate the mean intensity as the "feature"
        mean_intensity = np.mean(frame)

        # If mean intensity is below the threshold, mark it as an anomaly
        if mean_intensity < ANOMALY_THRESHOLD:
            anomalies.append(i)

    print(f"Detected {len(anomalies)} anomalies.")
    return anomalies

def create_gif_for_anomalies(frames_dir, anomalies, gif_path, frames_around=20):
    """Create a GIF containing frames around each anomaly."""
    anomaly_frames = []
    for anomaly in anomalies:
        for offset in range(-frames_around, frames_around + 1):
            frame_idx = anomaly + offset
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")

            if os.path.exists(frame_path):
                frame = Image.open(frame_path)
                anomaly_frames.append(frame)

    if anomaly_frames:
        anomaly_frames[0].save(
            gif_path,
            save_all=True,
            append_images=anomaly_frames[1:],
            duration=100,  # Frame duration in milliseconds
            loop=0
        )
        print(f"GIF saved at {gif_path}")
    else:
        print("No frames to create a GIF.")

def main():
    # Step 1: Extract frames from the video
    total_frames = extract_frames(VIDEO_PATH, FRAME_OUTPUT_DIR)

    # Step 2: Detect anomalies in the frames
    anomalies = detect_anomalies(FRAME_OUTPUT_DIR, total_frames)

    # Step 3: Create a GIF around anomalies
    create_gif_for_anomalies(FRAME_OUTPUT_DIR, anomalies, GIF_OUTPUT_PATH, FRAMES_AROUND_ANOMALY)

if __name__ == "__main__":
    main()
