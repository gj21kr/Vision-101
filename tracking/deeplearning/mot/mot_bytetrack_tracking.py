import numpy as np
import cv2
import os
import urllib.request
from ultralytics import YOLO

# --- Utility Functions ---

def download_video_if_needed(filename, url):
    """Downloads the video from the given URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading '{filename}'...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

def main():
    video_filename = 'vtest.avi'
    video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'
    download_video_if_needed(video_filename, video_url)

    cap = cv2.VideoCapture(video_filename)
    # Use the latest official YOLOv8 model
    model = YOLO('yolov8n.pt')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_bytetrack_official.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    colors = np.random.rand(100, 3) * 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detection and Tracking in one step ---
        # Use `persist=True` to keep track of objects between frames.
        # The `tracker='bytetrack.yaml'` argument specifies the tracker configuration.
        results = model.track(frame, persist=True, classes=[0], verbose=False, tracker='bytetrack.yaml')

        # Check if tracks are detected
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # --- Visualization ---
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

    print("Processing finished. ByteTrack output saved to 'output_bytetrack_official.mp4'.")
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
