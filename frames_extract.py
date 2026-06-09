import cv2
import os
def extract_frames(video_path, output_folder, frame_step=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 'frame_step' frame
        if frame_count % frame_step == 0:
            output_path = os.path.join(
                output_folder, f"rock_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(
        f"Extraction complete! Saved {saved_count} frames to '{output_folder}'.")

if __name__ == "__main__":
    # Ensure your video file is named 'rock_video.mp4'
    extract_frames('rock_video.mp4', 'rock_frames', frame_step=7)

