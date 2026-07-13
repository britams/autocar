# ============================================================
# YOLO로 "사람"을 찾아서 계속 따라가는 오토카
#
# 원리:
#   1) 화면에서 사람(person)을 찾는다.
#   2) 사람이 화면 왼쪽/오른쪽 어디 있는지 봐서, 그쪽으로 핸들을 꺾어
#      사람이 계속 화면 가운데에 오도록 만든다.
#   3) 사람 박스의 크기(=거리 추정)를 보고
#      - 작으면(멀리 있으면) 앞으로 감
#      - 적당하면 그대로 유지(멈춤)
#      - 크면(너무 가까우면) 정지
#   4) 사람이 아예 안 보이면 안전하게 정지.
#
# 실행 위치 주의: 반드시 ~/tensorrt_demos 폴더 안에서 실행해야 함.
# ============================================================

import sys
import time

sys.path.insert(1, '.')

import cv2
import pycuda.autoinit  # noqa

from pop import Pilot, Util
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_cls_dict


Car = Pilot.AutoCar()

# ---- 설정값 ----
CONF_THRESH = 0.5

FOLLOW_SPEED = 35        # 사람을 향해 다가갈 때 속도
STEER_GAIN = 1.2         # 사람이 중앙에서 벗어난 정도를 조향값으로 바꿀 때 곱하는 값
MAX_STEER = 1.0

# 사람 박스 면적(화면 대비 비율)로 "거리"를 대략 판단.
# 값은 실측하면서 조정 필요.
TOO_FAR_RATIO = 0.05     # 이보다 작으면(=멀리 있으면) 전진
GOOD_DISTANCE_RATIO = 0.15  # 이 정도면 적당한 거리 (전진도 정지도 아님)
TOO_CLOSE_RATIO = 0.25   # 이보다 크면(=너무 가까우면) 정지


def get_person_box(boxes, confs, clss, cls_dict):
    """
    탐지된 것들 중 "person"만 골라서, 그 중 가장 큰(=가장 가까운) 사람 박스를 돌려준다.
    사람이 없으면 None.
    """
    biggest_area = 0
    biggest_box = None

    for box, conf, cls_id in zip(boxes, confs, clss):
        label = cls_dict.get(int(cls_id), "")
        if label != "person":
            continue

        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)

        if area > biggest_area:
            biggest_area = area
            biggest_box = box

    return biggest_box


def decide_action(box, frame_width, frame_height):
    """사람 박스를 보고 (조향값, 속도, 상태설명)을 결정한다."""
    if box is None:
        return 0.0, 0, "사람 없음 (정지)"

    x_min, y_min, x_max, y_max = box
    box_area = (x_max - x_min) * (y_max - y_min)
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    box_center_x = (x_min + x_max) / 2
    frame_center_x = frame_width / 2
    # -1(왼쪽) ~ 0(중앙) ~ +1(오른쪽): 사람이 화면 중앙에서 얼마나 벗어났는지.
    center_ratio = (box_center_x - frame_center_x) / frame_center_x

    # 사람을 화면 중앙에 유지하도록 조향.
    # 사람이 왼쪽에 있으면(center_ratio<0) 왼쪽으로, 오른쪽에 있으면 오른쪽으로 꺾음.
    steer = max(-MAX_STEER, min(MAX_STEER, center_ratio * STEER_GAIN))

    if area_ratio < TOO_FAR_RATIO:
        return steer, FOLLOW_SPEED, "멀리 있음 -> 전진"
    elif area_ratio < GOOD_DISTANCE_RATIO:
        return steer, FOLLOW_SPEED // 2, "적당한 거리 -> 천천히 따라감"
    elif area_ratio < TOO_CLOSE_RATIO:
        return steer, 0, "충분히 가까움 -> 정지(방향만 맞춤)"
    else:
        return 0.0, 0, "너무 가까움 -> 정지"


def main():
    trt_yolo = TrtYOLO('yolov3-tiny-416', category_num=80)
    cls_dict = get_cls_dict(80)

    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("사람 따라가기 시작. Ctrl+C로 종료하세요.")

    last_state = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]

            boxes, confs, clss = trt_yolo.detect(frame, conf_th=CONF_THRESH)
            person_box = get_person_box(boxes, confs, clss, cls_dict)

            steer, speed, state = decide_action(person_box, width, height)

            # 상태가 바뀔 때만 명령 전송 (반응속도 이슈로 이전에 이렇게 바꿨었음).
            if state != last_state:
                Car.steering = steer
                Car.setSpeed(speed)

                if speed == 0:
                    Car.stop()
                else:
                    Car.forward()

                print(state)
                last_state = state

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        Car.stop()
        cap.release()


if __name__ == "__main__":
    main()
