import cv2
import os
import urllib.request
from ultralytics import YOLO
import torch

# --- PyTracking Imports ---
# Note: This assumes you have installed pytracking and its models correctly.
from pytracking.evaluation.tracker import Tracker
from pytracking.parameter.atom import default_parameter_loader

def download_video_if_needed(filename, url):
    """Downloads the video from the given URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading '{filename}'...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

def main():
    # --- 1. Initial Setup ---
    video_filename = 'vtest.avi'
    video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'
    download_video_if_needed(video_filename, video_url)

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filename}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # --- 2. Use YOLOv8 to find the initial object to track ---
    print("Loading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')
    print("Detecting a person in the first frame...")
    results = yolo_model(frame, classes=[0], verbose=False)

    if len(results[0].boxes) == 0:
        print("Error: No person found in the first frame.")
        cap.release()
        return

    # Use the first detected person as the target
    box = results[0].boxes[0]
    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
    init_roi = (x1, y1, x2 - x1, y2 - y1)
    print(f"YOLO detected person at: {init_roi}")

    # --- 3. Initialize the ATOM Tracker ---
    print("Initializing ATOM tracker...")
    params = default_parameter_loader.get_default_parameters()
    params.tracker_name = 'atom'
    params.model_name = 'default'
    atom_tracker = Tracker(params)
    
    # The tracker needs the first frame and the initial bounding box
    atom_tracker.initialize(frame, init_roi)

    # --- 4. Video Saving Setup ---
    frame_height, frame_width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_atom.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print("Output will be saved to 'output_atom.mp4'")

    # --- 5. Main Tracking Loop ---
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        print(f"Tracking frame {frame_num}...")

        # Track the object
        tracked_bbox, _ = atom_tracker.track(frame)

        # --- Visualization ---
        x, y, w, h = [int(v) for v in tracked_bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "ATOM", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

    print("Processing finished.")
    cap.release()
    out.release()

if __name__ == "__main__":
    # Ensure you are running this inside the pytracking virtual environment
    # and from the root directory of the cloned pytracking repository.
    main()
