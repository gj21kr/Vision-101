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

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    frame_height, frame_width, _ = frame.shape

    # --- 2. YOLOv8로 추적 대상(ROI) 설정 및 모델 생성 ---
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
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    print(f"YOLO가 사람을 감지했습니다: x={x}, y={y}, w={w}, h={h}")

    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 객체의 색상 모델 (Hue 채널 히스토그램)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # --- 3. 파티클 필터 초기화 ---
    n_particles = 300
    # 초기 파티클들을 ROI 중심 주변에 생성
    particles = np.array([
        np.random.randint(x, x + w, n_particles),
        np.random.randint(y, y + h, n_particles)
    ]).T  # shape: (n_particles, 2)

    # 초기 가중치는 모두 동일
    weights = np.ones(n_particles) / n_particles

    # --- 4. 비디오 저장 설정 ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = 'output_particle.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to '{output_filename}'")

    # --- 5. 메인 추적 루프 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 1. 예측 (Prediction): 파티클들을 약간씩 무작위로 이동
        # 표준편차에 따라 움직임의 크기 조절
        np.add(particles, np.random.normal(0, 10, particles.shape), out=particles, casting="unsafe")
        # 파티클이 프레임 밖으로 나가지 않도록 제한
        particles[:, 0] = np.clip(particles[:, 0], 0, frame_width - 1)
        particles[:, 1] = np.clip(particles[:, 1], 0, frame_height - 1)

        # 2. 가중치 계산 (Update)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i, (px, py) in enumerate(particles.astype(int)):
            # 각 파티클 위치에서 작은 패치 추출
            ph, pw = h // 2, w // 2
            patch = hsv_frame[max(0, py-ph):py+ph, max(0, px-pw):px+pw]
            
            if patch.size == 0:
                weights[i] = 0
                continue

            # 패치의 히스토그램 계산 및 타겟과 비교
            patch_hist = cv2.calcHist([patch], [0], None, [180], [0, 180])
            cv2.normalize(patch_hist, patch_hist, 0, 255, cv2.NORM_MINMAX)
            
            # Bhattacharyya 거리를 이용해 유사도 계산 (거리가 작을수록 유사)
            distance = cv2.compareHist(roi_hist, patch_hist, cv2.HISTCMP_BHATTACHARYYA)
            # 거리를 가중치로 변환 (유사할수록 높은 가중치)
            weights[i] = np.exp(-10 * (distance ** 2))

        # 가중치 정규화 (총합이 1이 되도록)
        if weights.sum() > 0:
            weights /= weights.sum()
        else: # 모든 가중치가 0이면 균등하게 재설정
            weights = np.ones(n_particles) / n_particles

        # 3. 상태 추정: 가중 평균으로 객체 위치 계산
        mean_pos = np.sum(particles * weights[:, np.newaxis], axis=0).astype(int)

        # 4. 재샘플링 (Resampling)
        # 가중치에 비례하여 다음 세대의 파티클들을 선택
        indices = np.random.choice(np.arange(n_particles), n_particles, p=weights)
        particles = particles[indices]

        # 5. 결과 시각화
        # 모든 파티클 그리기 (파란색 점)
        for p in particles.astype(int):
            cv2.circle(frame, tuple(p), 1, (255, 0, 0), -1)
        # 추정된 위치 그리기 (초록색 사각형)
        cv2.rectangle(frame, (mean_pos[0]-w//2, mean_pos[1]-h//2), 
                      (mean_pos[0]+w//2, mean_pos[1]+h//2), (0, 255, 0), 2)

        out.write(frame)

    # --- 6. 종료 및 자원 해제 ---
    print("Processing finished.")
    out.release()
    cap.release()

if __name__ == "__main__":
    main()