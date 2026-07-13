# ============================================================
# 방금 찍은 calib_original.jpg에서, "선 부분"과 "트랙 바닥 부분"의
# 정확한 색상(HSV) 값을 직접 뽑아서 비교하는 진단용 스크립트.
# 추측으로 값을 맞추는 대신, 정확한 숫자를 보고 한 번에 기준을 정하기 위함.
# ============================================================

import cv2
import numpy as np

frame = cv2.imread("calib_original.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

height, width = frame.shape[:2]
print(f"이미지 크기: {width} x {height}")


def sample(name, x_ratio, y_ratio, box=15):
    """(x_ratio, y_ratio) 비율 위치를 중심으로 작은 사각형 영역의 평균 HSV를 출력."""
    x = int(width * x_ratio)
    y = int(height * y_ratio)
    patch = hsv[max(0, y - box):y + box, max(0, x - box):x + box]
    h, s, v = patch[:, :, 0].mean(), patch[:, :, 1].mean(), patch[:, :, 2].mean()
    print(f"{name:12s} (x={x}, y={y}): H={h:.0f}, S={s:.0f}, V={v:.0f}")


# 사진을 보고 대략적인 위치를 비율로 지정.
# 화면 아래쪽(카메라 가까이)에서: 왼쪽 흰 선 / 가운데 빨간 트랙 / 오른쪽 흰 선
sample("왼쪽 흰선",   0.08, 0.85)
sample("가운데 트랙", 0.50, 0.85)
sample("오른쪽 흰선", 0.92, 0.85)

# 화면 중간쯤(조금 먼 곳)에서도 동일하게 확인.
sample("왼쪽 흰선(먼)",   0.20, 0.60)
sample("가운데 트랙(먼)", 0.50, 0.60)
sample("오른쪽 흰선(먼)", 0.78, 0.60)
