import numpy as np
import cv2
import os
import urllib.request

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

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # --- 2. 칼만 필터 초기화 ---
    # 상태 변수: [x, y, vx, vy] (위치, 속도) - 4개
    # 측정 변수: [x, y] (측정된 위치) - 2개
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 프로세스 노이즈: 모델의 불확실성. 작을수록 모델을 신뢰.
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    # 측정 노이즈: 측정값의 불확실성. 클수록 측정값을 덜 신뢰.
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.5

    # 초기 위치 찾기 (하드코딩된 ROI의 중심)
    x, y, w, h = 280, 50, 60, 120 # vtest.avi의 걷는 사람
    cx = x + w // 2
    cy = y + h // 2
    kalman.statePost = np.array([cx, cy, 0, 0], np.float32)
    print(f"Initial state set to center of ROI: ({cx}, {cy})")

    # 측정에 사용할 색상 범위 (흰색)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # --- 3. 비디오 저장 설정 ---
    frame_height, frame_width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = 'output_kalman.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to '{output_filename}'")

    # --- 4. 메인 추적 루프 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 1. 예측 (Prediction)
        prediction = kalman.predict()
        pred_pt = (int(prediction[0]), int(prediction[1]))

        # 2. 측정 (Measurement)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found = False
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50: # 노이즈 제거
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 3. 보정 (Correction)
                    kalman.correct(np.array([cx, cy], np.float32))
                    found = True

        # 4. 결과 시각화
        state_pt = (int(kalman.statePost[0]), int(kalman.statePost[1]))
        cv2.circle(frame, pred_pt, 10, (255, 0, 0), 2)      # 예측: 파란색 원
        cv2.circle(frame, state_pt, 10, (0, 255, 0), 2)     # 보정: 초록색 원
        if found:
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2) # 측정: 빨간색 십자

        out.write(frame)

    # --- 5. 종료 및 자원 해제 ---
    print("Processing finished.")
    out.release()
    cap.release()

if __name__ == "__main__":
    main()