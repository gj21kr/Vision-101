import numpy as np
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
    # --- 1. 초기 설정 ---
    video_filename = 'vtest.avi'
    video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'
    download_video_if_needed(video_filename, video_url)

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filename}")
        return

    # 첫 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # --- 2. YOLOv8로 추적 대상(ROI) 설정 및 모델 생성 ---
    print("YOLOv8 모델을 로드합니다...")
    model = YOLO('yolov8n.pt')
    print("첫 프레임에서 사람을 감지합니다...")
    results = model(frame, classes=[0], verbose=False) # class 0 is 'person'

    if len(results[0].boxes) == 0:
        print("에러: 첫 프레임에서 사람을 찾지 못했습니다.")
        cap.release()
        return

    # 첫 번째로 감지된 사람의 바운딩 박스를 추적 윈도우로 사용
    box = results[0].boxes[0]
    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    track_window = (x, y, w, h)
    print(f"YOLO가 사람을 감지했습니다. 추적 윈도우: x={x}, y={y}, w={w}, h={h}")

    # ROI(Region of Interest) 설정
    roi = frame[y:y+h, x:x+w]
    
    # ROI를 HSV 색상 공간으로 변환
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 추적을 위한 객체 모델(Hue 채널 히스토그램) 생성
    # 조명이 어둡거나(value가 낮음) 채도가 낮은(saturation이 낮음) 영역은 무시
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # CamShift 종료 조건 설정 (10번 반복하거나 1pt 이상 이동하면 종료)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # --- 3. 비디오 저장 설정 ---
    frame_height, frame_width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = 'output_camshift.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to '{output_filename}'")

    # --- 4. 메인 추적 루프 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # CamShift 알고리즘 적용하여 새로운 위치 찾기
        ret_val, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 결과 시각화 (회전된 사각형 그리기)
        pts = cv2.boxPoints(ret_val)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        out.write(frame)

    # --- 5. 종료 및 자원 해제 ---
    print("Processing finished.")
    out.release()
    cap.release()

if __name__ == "__main__":
    main()