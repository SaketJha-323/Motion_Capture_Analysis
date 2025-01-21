import cv2

# Parameters
VIDEO_PATH = "data/video/Drone1.mp4"  # Path to the video

def get_video_info(video_path):
    """Get video information, including FPS, total frames, and duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()
    return fps, total_frames, duration


def main():
    # Get video information
    video_info = get_video_info(VIDEO_PATH)
    if video_info:
        fps, total_frames, duration = video_info
        print(f"Video FPS: {fps:.2f}")
        print(f"Total frames in the video: {total_frames}")
        print(f"Video duration: {duration:.2f} seconds")
    else:
        print("Failed to retrieve video information.")


if __name__ == "__main__":
    main()
