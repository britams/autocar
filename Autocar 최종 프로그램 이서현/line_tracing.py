# ============================================================
# 라인 트레이싱: 트랙의 두 선(왼쪽 선, 오른쪽 선) 사이 "가운데"를 유지하며 주행
#
# YOLO(딥러닝)가 아니라 OpenCV의 "색상 필터링"으로 만든다.
# 이유: 선은 모양이 단순하고 색이 뚜렷해서(흰색/노란색 vs 어두운 트랙 바닥),
#       딥러닝 학습 없이도 색상만으로 충분히 빠르고 정확하게 찾을 수 있음.
#
# [중요] 이 코드는 실제 트랙 사진으로 HSV 색상 범위를 보정해야 정확히 동작합니다.
#        아래 LOWER_WHITE/UPPER_WHITE, LOWER_YELLOW/UPPER_YELLOW 값은 "기본 추정값"이라
#        실제 트랙에서 line_calibration.py로 확인 후 조정이 필요할 수 있습니다.
# ============================================================

import time

import cv2
import numpy as np

from pop import Pilot, Util

Car = Pilot.AutoCar()

# ---- 색상 범위 설정 (HSV 기준) ----
# HSV: 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현하는 방식.
# 조명이 바뀌어도 RGB보다 색을 더 안정적으로 구분할 수 있어서 씀.

# 흰색 선: 채도(S)가 낮고, 명도(V)가 높음 (밝고 색이 옅음)
LOWER_WHITE = np.array([0, 0, 180])
UPPER_WHITE = np.array([180, 60, 255])

# 노란색 선: 색상(H)이 노란색 범위, 채도/명도는 어느 정도 높음
LOWER_YELLOW = np.array([20, 80, 120])
UPPER_YELLOW = np.array([35, 255, 255])

# ---- 주행 설정값 ----
FORWARD_SPEED = 35
STEER_GAIN = 1.5       # 중심에서 벗어난 정도를 조향값으로 바꿀 때 곱하는 값 (클수록 급하게 꺾음)
MAX_STEER = 1.0

# 화면에서 아래쪽(차와 가까운 부분)만 관심영역(ROI)으로 사용.
# 위쪽(멀리 보이는 부분)은 잡음이 많아서 제외.
ROI_TOP_RATIO = 0.6    # 화면 세로 기준 60% 지점부터 아래쪽만 사용


def get_line_mask(frame):
    """흰색 또는 노란색 선에 해당하는 픽셀만 하얗게(255) 표시한 마스크를 만든다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # 흰색 마스크와 노란색 마스크를 합침 (둘 중 하나라도 선이면 인정).
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    return mask


def find_line_center_x(mask):
    """
    마스크(흰색=선 픽셀)에서, "왼쪽 선"과 "오른쪽 선" 사이의 정가운데 위치를 계산한다.

    전체 선 픽셀의 평균을 그냥 구하면, 한쪽 선이 더 길게/많이 보일 때
    중심이 그쪽으로 쏠리는 문제가 있어서, 화면을 왼쪽 절반/오른쪽 절반으로 나눠
    "왼쪽 선의 위치"와 "오른쪽 선의 위치"를 각각 구한 뒤 그 둘의 중간을 쓴다.
    """
    height, width = mask.shape[:2]
    half = width // 2

    left_half = mask[:, :half]
    right_half = mask[:, half:]

    ys_l, xs_l = np.where(left_half > 0)
    ys_r, xs_r = np.where(right_half > 0)

    left_found = len(xs_l) > 0
    right_found = len(xs_r) > 0

    if left_found:
        left_x = float(np.mean(xs_l))
    if right_found:
        # right_half는 잘라낸 부분이라 x좌표에 half를 더해줘야 원래 화면 기준 좌표가 됨.
        right_x = float(np.mean(xs_r)) + half

    if left_found and right_found:
        # 양쪽 선 다 보임 -> 그 둘의 정가운데.
        return (left_x + right_x) / 2
    elif left_found:
        # 왼쪽 선만 보임 -> 왼쪽 선에서 트랙 폭의 절반만큼 오른쪽이 대략 중심일 것으로 추정.
        return left_x + (width * 0.25)
    elif right_found:
        # 오른쪽 선만 보임 -> 반대로 추정.
        return right_x - (width * 0.25)
    else:
        return None


def main():
    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    Car.setSpeed(FORWARD_SPEED)

    print("라인 트레이싱 시작. Ctrl+C로 종료하세요.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]

            # 관심영역(ROI): 화면 아래쪽(차와 가까운 부분)만 잘라서 사용.
            roi_top = int(height * ROI_TOP_RATIO)
            roi = frame[roi_top:height, :]

            mask = get_line_mask(roi)
            line_center_x = find_line_center_x(mask)

            if line_center_x is None:
                # 선을 못 찾음 -> 안전하게 정지.
                Car.setSpeed(0)
                Car.stop()
                print("선을 찾을 수 없습니다. 정지.")
                time.sleep(0.1)
                continue

            # 화면 중심과 선 중심의 차이 -> 얼마나, 어느 방향으로 틀어져 있는지 계산.
            frame_center_x = width / 2
            error = (line_center_x - frame_center_x) / frame_center_x  # -1 ~ 1

            # error가 양수면 선(트랙 중심)이 오른쪽에 있다는 뜻 -> 오른쪽으로 조향.
            # error가 음수면 왼쪽에 있다는 뜻 -> 왼쪽으로 조향.
            steer = error * STEER_GAIN
            steer = max(-MAX_STEER, min(MAX_STEER, steer))  # -1~1 범위로 제한

            Car.steering = steer
            Car.setSpeed(FORWARD_SPEED)
            Car.forward()

            print(f"line_center_x={line_center_x:.1f}, error={error:.2f}, steer={steer:.2f}")

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        Car.stop()
        cap.release()


if __name__ == "__main__":
    main()
