# ============================================================
# 라인 트레이싱 + 장애물 회피 통합 버전
#
# [핵심 설계 아이디어] "목표 지점을 옆으로 잠깐 옮기기"
#   회피할 때 "라인 추적을 끄고 blind하게 핸들을 꺾는" 방식이 아니라,
#   라인 추적 로직은 항상 계속 켜져 있고,
#   장애물이 있을 때만 "목표로 삼는 중심 위치"를 옆으로 잠깐 옮긴다.
#
#   예) 평소: 화면 정중앙(예: 320픽셀)을 목표로 조향
#       회피 중: 화면 중앙에서 오른쪽으로 100픽셀 옮긴 지점(420픽셀)을 목표로 조향
#       회피 시간이 끝나면: 다시 화면 정중앙(320픽셀)을 목표로 복귀
#
#   -> 항상 "선을 보면서" 조향하는 같은 제어 로직이 돌아가고 있기 때문에,
#      회피가 끝나면 자연스럽게 원래 라인 중심으로 돌아온다.
#      (장애물이 무엇인지는 중요하지 않으므로 YOLO는 "장애물 유무 판단"에만 사용)
#
# 실행 위치 주의: 반드시 ~/tensorrt_demos 폴더 안에서 실행해야 함.
# ============================================================

import sys
import time

sys.path.insert(1, '.')

import cv2
import numpy as np
import pycuda.autoinit  # noqa

from pop import Pilot, Util
from utils.yolo_with_plugins import TrtYOLO


Car = Pilot.AutoCar()

# ---- 라인 색상 범위 (line_tracing.py, line_calibration.py와 동일하게 맞춰야 함) ----
LOWER_WHITE = np.array([0, 0, 180])
UPPER_WHITE = np.array([180, 60, 255])
LOWER_YELLOW = np.array([20, 80, 120])
UPPER_YELLOW = np.array([35, 255, 255])
ROI_TOP_RATIO = 0.6

# ---- 주행 설정값 ----
FORWARD_SPEED = 35
STEER_GAIN = 1.5
MAX_STEER = 1.0

# ---- 장애물 판단 기준 (YOLO 박스 크기 기준) ----
CONF_THRESH = 0.5
AVOID_AREA_RATIO = 0.05     # 화면의 5% 이상 -> 회피 시작
AVOID_OFFSET_PIXELS = 150   # 회피할 때 목표 지점을 얼마나 옆으로 옮길지 (픽셀)
AVOID_DURATION_SEC = 1.5    # 회피 상태를 얼마나 유지할지 (이 시간 동안 목표가 옆으로 옮겨짐)


def get_line_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    return cv2.bitwise_or(mask_white, mask_yellow)


def find_line_center_x(mask):
    height, width = mask.shape[:2]
    half = width // 2

    ys_l, xs_l = np.where(mask[:, :half] > 0)
    ys_r, xs_r = np.where(mask[:, half:] > 0)

    left_found = len(xs_l) > 0
    right_found = len(xs_r) > 0

    if left_found:
        left_x = float(np.mean(xs_l))
    if right_found:
        right_x = float(np.mean(xs_r)) + half

    if left_found and right_found:
        return (left_x + right_x) / 2
    elif left_found:
        return left_x + (width * 0.25)
    elif right_found:
        return right_x - (width * 0.25)
    else:
        return None


def get_biggest_obstacle_side(boxes, frame_width, frame_height):
    """
    가장 큰(가장 가까운) 장애물이 있는지, 있다면 화면 왼쪽/오른쪽 어느 쪽에 있는지 확인.
    반환값: None(장애물 없음) 또는 "left" 또는 "right"
    """
    if len(boxes) == 0:
        return None

    biggest_area = 0
    biggest_box = None

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        if area > biggest_area:
            biggest_area = area
            biggest_box = box

    frame_area = frame_width * frame_height
    if biggest_area / frame_area < AVOID_AREA_RATIO:
        return None  # 너무 작음(=멀리 있음) -> 무시

    x_min, y_min, x_max, y_max = biggest_box
    box_center_x = (x_min + x_max) / 2

    return "left" if box_center_x < frame_width / 2 else "right"


def main():
    trt_yolo = TrtYOLO('yolov3-tiny-416', category_num=80)

    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    Car.setSpeed(FORWARD_SPEED)

    print("라인 트레이싱 + 장애물 회피 시작. Ctrl+C로 종료하세요.")

    # 회피 상태를 언제까지 유지할지 저장하는 변수.
    # 0이면 "회피 중이 아님" = 평소처럼 라인 정중앙을 목표로 함.
    avoid_until_time = 0
    avoid_offset = 0  # 지금 목표를 얼마나 옆으로 옮겨야 하는지 (픽셀, 음수=왼쪽/양수=오른쪽)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]
            now = time.time()

            # ---- 1) 장애물 확인 (YOLO) ----
            # 이미 회피 중이 아닐 때만 새로 장애물을 판단한다.
            # (회피 중간에 판단이 계속 바뀌면 갈팡질팡하므로, 한 번 정하면 AVOID_DURATION_SEC동안 유지)
            if now >= avoid_until_time:
                boxes, confs, clss = trt_yolo.detect(frame, conf_th=CONF_THRESH)
                side = get_biggest_obstacle_side(boxes, width, height)

                if side == "left":
                    # 장애물이 왼쪽에 있음 -> 오른쪽으로 목표를 옮겨서 피함.
                    avoid_offset = AVOID_OFFSET_PIXELS
                    avoid_until_time = now + AVOID_DURATION_SEC
                    print("장애물(왼쪽) 감지 -> 오른쪽으로 회피")
                elif side == "right":
                    avoid_offset = -AVOID_OFFSET_PIXELS
                    avoid_until_time = now + AVOID_DURATION_SEC
                    print("장애물(오른쪽) 감지 -> 왼쪽으로 회피")
                else:
                    avoid_offset = 0  # 장애물 없음 -> 목표 옮기지 않음(평소 라인 중심)

            # ---- 2) 라인 검출 (OpenCV) ----
            roi_top = int(height * ROI_TOP_RATIO)
            roi = frame[roi_top:height, :]
            mask = get_line_mask(roi)
            line_center_x = find_line_center_x(mask)

            if line_center_x is None:
                Car.setSpeed(0)
                Car.stop()
                print("선을 찾을 수 없습니다. 정지.")
                time.sleep(0.1)
                continue

            # ---- 3) 목표 지점 계산 (평소: 화면 중앙 / 회피 중: 옆으로 옮긴 지점) ----
            target_x = (width / 2) + avoid_offset

            # ---- 4) 목표 지점과 실제 선 위치의 차이로 조향 계산 ----
            # (이 계산 로직 자체는 회피 중이든 아니든 항상 동일함 -> 회피가 끝나면 자동으로 복귀)
            error = (line_center_x - target_x) / (width / 2)
            steer = max(-MAX_STEER, min(MAX_STEER, error * STEER_GAIN))

            Car.steering = steer
            Car.setSpeed(FORWARD_SPEED)
            Car.forward()

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        Car.stop()
        cap.release()


if __name__ == "__main__":
    main()
