import cv2
import numpy as np
from collections import deque

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.bbox = bbox
        self.trace = deque(maxlen=20)
        self.id = np.random.randint(0, 10000)

    def update(self, bbox):
        self.bbox = bbox
        self.trace.append(bbox)

class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        updated_trackers = []
        for det in detections:
            tracker = KalmanBoxTracker(det)
            updated_trackers.append(tracker)
        self.trackers = updated_trackers
        return [trk.bbox for trk in self.trackers]

def detect_objects(frame):
    # 임시: 임계값 기반 객체 검출 (실제는 detector 사용)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            detections.append([x, y, x+w, y+h])
    return detections

def main():
    cap = cv2.VideoCapture('vtest.avi')
    tracker = Sort()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame)
        tracked = tracker.update(detections)
        for bbox in tracked:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow('SORT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
