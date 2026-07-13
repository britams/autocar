# ============================================================
# Jetson 실전용: TensorRT YOLO + 실제 조향/속도 제어 + TTS 음성 알림
#
# WSL에서 만든 obstacle_avoidance.py(ultralytics 기반)는 Jetson에서 안 돌아가서,
# 오늘 검증한 TensorRT 방식(TrtYOLO, FPS 30)으로 다시 작성한 버전.
#
# 실행 위치 주의: 반드시 ~/tensorrt_demos 폴더 안에서 실행해야 함.
# (plugins/libyolo_layer.so 를 상대경로로 찾기 때문)
# ============================================================

import subprocess
import sys
import threading
import time

sys.path.insert(1, '.')

import cv2
import pycuda.autoinit  # noqa: GPU 초기화. import만 해도 자동으로 준비됨.

from pop import Pilot, Util
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_cls_dict


# ---- TTS(음성 알림) ----
last_tts_time = 0
TTS_COOLDOWN_SEC = 3


def _play_tts(text):
    espeak = subprocess.Popen(
        ["espeak-ng", "-v", "ko", text, "--stdout"],
        stdout=subprocess.PIPE,
    )
    # Jetson은 실제 사운드카드가 있으므로 aplay 사용 (WSL에서는 paplay를 썼음).
    subprocess.run(["aplay", "-q"], stdin=espeak.stdout)


def speak(text):
    global last_tts_time
    now = time.time()
    if now - last_tts_time < TTS_COOLDOWN_SEC:
        return
    last_tts_time = now
    threading.Thread(target=_play_tts, args=(text,), daemon=True).start()


# ---- 주행 설정값 ----
FORWARD_SPEED = 40
AVOID_STEER = 0.7
AVOID_AREA_RATIO = 0.05   # 화면의 5% 이상 -> 회피 시작
STOP_AREA_RATIO = 0.20    # 화면의 20% 이상 -> 정지

CONF_THRESH = 0.5


def get_closest_box(boxes, confs):
    """탐지된 박스들 중 화면에서 가장 큰(=가장 가까운) 것 하나를 고른다."""
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

    return biggest_box


def decide_action(box, frame_width, frame_height):
    """
    가장 가까운 장애물 박스를 보고 (조향값, 속도, 상태설명)을 결정한다.

    [설계 방식: "정지 후 생각"]
    움직이면서 실시간으로 방향을 트는 대신,
      1) 물체가 보이면 일단 무조건 정지
      2) 멈춘 상태에서 어느 쪽으로 피할지 판단
      3) 판단이 끝나면 그 방향으로 움직임
    순서로 동작하게 만들어서, 반응속도가 느려도 사고 위험 없이
    "멈칫거리며 피해가는" 형태로 자연스럽게 동작하게 함.
    """
    if box is None:
        # 아무것도 안 보이면 그냥 직진.
        return 0.0, FORWARD_SPEED, "직진"

    x_min, y_min, x_max, y_max = box
    box_area = (x_max - x_min) * (y_max - y_min)
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    box_center_x = (x_min + x_max) / 2
    frame_center_x = frame_width / 2
    center_ratio = (box_center_x - frame_center_x) / frame_center_x

    if area_ratio >= STOP_AREA_RATIO:
        # 아주 가까움 -> 완전 정지.
        return 0.0, 0, "정지 (장애물 매우 근접)"
    elif area_ratio >= AVOID_AREA_RATIO:
        # 물체가 보이면 "일단 무조건 정지"부터 한다.
        # (실시간으로 움직이면서 트는 게 아니라, 멈춘 뒤 다음 판단으로 넘어감)
        return 0.0, 0, "정지 후 방향 판단 중"
    else:
        # 아직 멀리 있어서 그냥 직진.
        return 0.0, FORWARD_SPEED, "직진 (장애물 감지되었으나 거리 있음)"


def decide_avoid_direction(box, frame_width):
    """
    정지한 상태에서, 어느 방향으로 피해야 할지만 따로 판단한다.
    (물체가 화면 왼쪽에 있으면 오른쪽으로, 오른쪽에 있으면 왼쪽으로)
    """
    x_min, y_min, x_max, y_max = box
    box_center_x = (x_min + x_max) / 2
    frame_center_x = frame_width / 2
    center_ratio = (box_center_x - frame_center_x) / frame_center_x

    return AVOID_STEER if center_ratio < 0 else -AVOID_STEER


def main():
    # TensorRT 엔진 로드. 'yolov3-tiny-416' -> yolo/yolov3-tiny-416.trt 를 찾아서 불러옴.
    trt_yolo = TrtYOLO('yolov3-tiny-416', category_num=80)
    cls_dict = get_cls_dict(80)

    Car = Pilot.AutoCar()

    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    Car.setSpeed(FORWARD_SPEED)

    print("장애물 회피 주행 시작 (TensorRT). Ctrl+C로 종료하세요.")

    last_state = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_height, frame_width = frame.shape[:2]

            boxes, confs, clss = trt_yolo.detect(frame, conf_th=CONF_THRESH)

            closest_box = get_closest_box(boxes, confs)
            steer, speed, state = decide_action(closest_box, frame_width, frame_height)

            if state != last_state:
                print(state)
                last_state = state

            if state == "정지 후 방향 판단 중":
                # ---- 1) 일단 정지 ----
                Car.setSpeed(0)
                Car.stop()
                speak("장애물을 발견했습니다. 방향을 판단합니다")
                time.sleep(0.5)  # 완전히 멈출 때까지 잠깐 대기

                # ---- 2) 멈춘 상태에서 다시 한 번 카메라를 보고 방향 판단 ----
                ret, frame = cap.read()
                if ret:
                    boxes2, confs2, clss2 = trt_yolo.detect(frame, conf_th=CONF_THRESH)
                    box2 = get_closest_box(boxes2, confs2)
                else:
                    box2 = closest_box

                if box2 is not None:
                    avoid_steer = decide_avoid_direction(box2, frame_width)

                    # ---- 3) 판단한 방향으로 짧게 이동 ----
                    Car.steering = avoid_steer
                    Car.setSpeed(FORWARD_SPEED)
                    Car.forward()
                    speak("장애물을 피해갑니다")
                    time.sleep(1.0)  # 이 시간만큼만 회피 방향으로 이동

                    # ---- 4) 다시 정지하고 정면 방향으로 되돌림 ----
                    Car.stop()
                    Car.steering = 0.0
                    time.sleep(0.3)

                last_state = None  # 다음 프레임에서 다시 처음부터 판단하도록 초기화

            elif state == "정지 (장애물 매우 근접)":
                Car.setSpeed(0)
                Car.steering = 0.0
                Car.stop()
                speak("장애물이 매우 가깝습니다. 정지합니다")

            else:
                # "직진" 또는 "직진 (장애물 감지되었으나 거리 있음)"
                Car.steering = 0.0
                Car.setSpeed(FORWARD_SPEED)
                Car.forward()

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        Car.stop()
        cap.release()


if __name__ == "__main__":
    main()
