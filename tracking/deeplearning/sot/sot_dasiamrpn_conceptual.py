import numpy as np
import cv2
import os
import urllib.request
from ultralytics import YOLO

def download_video_if_needed(filename, url):
    """Downloads the video from the given URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading '{filename}'...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

def get_color_feature(frame, bbox):
    """Extracts a simple color histogram feature for a single bounding box."""
    x1, y1, x2, y2 = bbox.astype(int)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # A simple histogram can act as a feature vector
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def main():
    # --- 1. Initial Setup ---
    video_filename = 'vtest.avi'
    video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'
    download_video_if_needed(video_filename, video_url)

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_filename}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
    
    frame_height, frame_width, _ = frame.shape

    # --- 2. Initialize Target with YOLOv8 ---
    print("Loading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt')
    print("Detecting a person in the first frame...")
    results = yolo_model(frame, classes=[0], verbose=False)

    if len(results[0].boxes) == 0:
        print("Error: No person found in the first frame.")
        cap.release()
        return

    # This is our "template" for the tracker
    init_box = results[0].boxes[0].xyxy[0].cpu().numpy()
    target_feature = get_color_feature(frame, init_box)
    current_bbox = init_box
    print(f"YOLO detected person. This is our target template: {current_bbox.astype(int)}")

    # --- 3. Video Saving Setup ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_sot_dasiamrpn_conceptual.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print("Output will be saved to 'output_sot_dasiamrpn_conceptual.mp4'")

    # --- 4. Main Tracking Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Define a search region around the last known position
        cx, cy = (current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2
        w, h = current_bbox[2] - current_bbox[0], current_bbox[3] - current_bbox[1]
        search_w, search_h = w * 2.5, h * 2.5
        search_x1 = max(0, int(cx - search_w / 2))
        search_y1 = max(0, int(cy - search_h / 2))
        search_x2 = min(frame_width, int(cx + search_w / 2))
        search_y2 = min(frame_height, int(cy + search_h / 2))
        search_region = frame[search_y1:search_y2, search_x1:search_x2]

        # 2. Detect candidate objects in the search region
        if search_region.size > 0:
            results = yolo_model(search_region, classes=[0], verbose=False)
            
            best_match_score = float('inf')
            best_candidate_box = None

            if len(results[0].boxes) > 0:
                # 3. Find the best matching candidate by comparing features
                for box in results[0].boxes:
                    candidate_box_local = box.xyxy[0].cpu().numpy()
                    candidate_box_global = candidate_box_local + np.array([search_x1, search_y1, search_x1, search_y1])
                    
                    candidate_feature = get_color_feature(frame, candidate_box_global)
                    if candidate_feature is None:
                        continue

                    # Compare features (lower score is better)
                    score = cv2.compareHist(target_feature, candidate_feature, cv2.HISTCMP_BHATTACHARYYA)

                    if score < best_match_score:
                        best_match_score = score
                        best_candidate_box = candidate_box_global
            
            # 4. Update the tracker state if a good match is found
            if best_candidate_box is not None and best_match_score < 0.7:
                current_bbox = best_candidate_box

        # --- 5. Visualization ---
        x1, y1, x2, y2 = current_bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "DaSiamRPN (Conceptual)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (search_x1, search_y1), (search_x2, search_y2), (255, 0, 0), 1) # Show search region

        out.write(frame)

    print("Processing finished.")
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
