import numpy as np
import cv2
import os
import urllib.request
from ultralytics import YOLO

# 비디오 파일 경로
video_filename = 'vtest.avi'
video_url = f'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{video_filename}'

# 예제 비디오 파일이 없으면 다운로드
if not os.path.exists(video_filename):
    print(f"'{video_filename}'을 다운로드합니다...")
    urllib.request.urlretrieve(video_url, video_filename)
    print("다운로드 완료.")

video_path = video_filename
cap = cv2.VideoCapture(video_path)
 
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Shi-Tomasi 코너 검출 파라미터
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Lucas-Kanade 옵티컬 플로우 파라미터
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 랜덤 색상 생성 (추적 경로 시각화용)
color = np.random.randint(0, 255, (100, 3))

# 첫 프레임 읽기 및 ROI 선택
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# --- YOLOv8로 첫 프레임에서 사람 감지 ---
print("YOLOv8 모델을 로드합니다...")
model = YOLO('yolov8n.pt')
print("첫 프레임에서 사람을 감지합니다...")
results = model(old_frame, classes=[0], verbose=False) # class 0 is 'person'

if len(results[0].boxes) == 0:
    print("에러: 첫 프레임에서 사람을 찾지 못했습니다.")
    cap.release()
    exit()

# 첫 번째로 감지된 사람의 바운딩 박스를 ROI로 사용
box = results[0].boxes[0]
x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
x, y, w, h = x1, y1, x2 - x1, y2 - y1
print(f"YOLO가 사람을 감지했습니다: x={x}, y={y}, w={w}, h={h}")

# 선택된 ROI 영역
roi = old_frame[y:y+h, x:x+w]
if roi.size == 0:
    print("Error: The hardcoded ROI is empty. Please check the coordinates.")
    exit()
old_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# ROI 내에서 추적할 초기 특징점 검출
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# ROI 좌표를 전체 프레임 좌표로 변환
if p0 is not None:
    p0[:, 0, 0] += x
    p0[:, 0, 1] += y
else:
    print("No features found to track.")
    cap.release()
    exit()

# 추적 경로를 그릴 마스크 생성
mask = np.zeros_like(old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# [추가] 결과를 비디오 파일로 저장하기 위한 VideoWriter 설정
frame_height, frame_width, _ = old_frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)
output_filename = 'output_klt.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 코덱 설정
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
print(f"Output will be saved to '{output_filename}'")

while(True):
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 옵티컬 플로우 계산
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 추적에 성공한 점들만 선택
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    else:
        good_new = [] # p1이 None이면 빈 리스트로 초기화

    # 추적할 포인트가 남아있는지 확인
    if len(good_new) == 0:
        print("모든 추적 포인트를 잃었습니다.")
        break

    # 추적 경로 그리기
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # 현재 프레임과 추적 경로를 합쳐서 보여주기
    img = cv2.add(frame, mask)

    # [수정] 화면에 보여주는 대신 비디오 파일에 프레임을 씁니다.
    out.write(img)

    # 다음 프레임을 위해 현재 상태를 이전 상태로 업데이트
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

print("Processing finished.")
out.release()
cap.release()
