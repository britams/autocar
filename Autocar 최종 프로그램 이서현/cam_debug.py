# 카메라가 실제로 어떤 포맷/해상도로 열리는지, 프레임 안에 진짜 데이터가 있는지 확인하는 진단용 스크립트.
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 카메라 압축 포맷을 MJPG로 강제 지정.
# (usbipd로 넘어온 웹캠은 기본 YUYV 포맷을 OpenCV가 잘못 해석해서 화면이 초록색으로만 나오는 경우가 있음)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 카메라가 잡고 있는 실제 설정값을 출력.
print("FOURCC:", int(cap.get(cv2.CAP_PROP_FOURCC)))
print("가로:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("세로:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
print("읽기 성공 여부:", ret)

if ret:
    print("frame shape:", frame.shape)
    # 프레임 안 픽셀 값들의 평균/최소/최대를 확인.
    # 다 초록색(0,255,0)이면 평균이 (0,255,0) 근처로 나올 것.
    print("픽셀 평균값(B,G,R):", frame.mean(axis=(0, 1)))
    print("픽셀 최솟값:", frame.min(), "최댓값:", frame.max())
    cv2.imwrite("debug_frame.jpg", frame)
    print("debug_frame.jpg 로 저장함")

cap.release()
