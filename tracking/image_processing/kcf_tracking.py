import cv2
import os
import urllib.request
from ultralytics import YOLO

def download_video_if_needed(filename, url):
    """지정된 URL에서 비디오 파일을 다운로드합니다 (파일이 없는 경우)."""
    if not os.path.exists(filename):
        print(f"'{filename}'을 다운로드합니다...")
        urllib.request.urlretrieve(url, filename)
        print("다운로드 완료.")

def main():
    # --- 1. 트래커 초기화 ---
    tracker_name = 'KCF'
    print(f"실행할 트래커: {tracker_name}")
    tracker = cv2.legacy.TrackerKCF_create()

    # --- 2. 비디오 및 ROI 설정 ---
    video_filename = 'vtest.avi'
    video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'
    download_video_if_needed(video_filename, video_url)

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"에러: 비디오 파일을 열 수 없습니다: {video_filename}")
        return

    ret, frame = cap.read()
    if not ret:
        print("에러: 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    # --- YOLOv8로 추적 대상(ROI) 설정 ---
    print("YOLOv8 모델을 로드합니다...")
    model = YOLO('yolov8n.pt')
    print("첫 프레임에서 사람을 감지합니다...")
    results = model(frame, classes=[0], verbose=False) # class 0 is 'person'

    if len(results[0].boxes) == 0:
        print("에러: 첫 프레임에서 사람을 찾지 못했습니다.")
        cap.release()
        return

    box = results[0].boxes[0]
    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
    w, h = x2 - x1, y2 - y1
    roi = (x1, y1, w, h)
    print(f"YOLO가 사람을 감지했습니다. 초기 ROI: {roi}")

    # 트래커 초기화
    ok = tracker.init(frame, roi)
    if not ok:
        print("에러: 트래커 초기화에 실패했습니다.")
        return

    # --- 3. 비디오 저장 설정 ---
    frame_height, frame_width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = f'output_{tracker_name.lower()}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"결과가 '{output_filename}' 파일로 저장됩니다.")

    # --- 4. 메인 추적 루프 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝입니다.")
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame) # 트래커 업데이트
        fps_val = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 바운딩 박스 그리기
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 화면에 정보 표시 (트래커 종류, FPS)
        cv2.putText(frame, tracker_name + " Tracker", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps_val)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        out.write(frame)

    # --- 5. 종료 및 자원 해제 ---
    print("처리가 완료되었습니다.")
    out.release()
    cap.release()

if __name__ == "__main__":
    main()