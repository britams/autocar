# ============================================================
# YOLO 기반 장애물 회피 주행
#
# [중요] 이 코드는 실제 오토카(Jetson)에서만 실행 가능합니다.
#        Pilot.AutoCar(), CSI 카메라(GStreamer)가 WSL에는 없어서
#        여기서는 테스트할 수 없고, 로직만 미리 짜둔 것입니다.
#
# 동작 방식 요약:
#   1) 카메라로 앞을 계속 촬영하면서 YOLO로 물체를 찾는다.
#   2) 화면에서 "가장 크게(=가장 가깝게) 보이는" 물체를 기준 장애물로 삼는다.
#   3) 그 장애물이
#      - 아주 가까우면       -> 정지 + 음성 경고
#      - 어느 정도 가까우면  -> 장애물 반대쪽으로 핸들을 꺾어서 피해감
#      - 안 보이거나 멀면    -> 그냥 직진
# ============================================================

from ultralytics import YOLO
import cv2

from pop import Pilot, Util
from tts_util import speak

# ---- 오토카 객체 준비 ----
Car = Pilot.AutoCar()

# ---- YOLO 모델 준비 ----
model = YOLO("yolov8n.pt")  # 경량 nano 모델

# ---- 주행 설정값 ----
FORWARD_SPEED = 40      # 평소 직진 속도 (0~100). 너무 빠르면 회피가 늦으므로 낮게 시작.
AVOID_STEER = 0.7       # 회피할 때 핸들을 얼마나 꺾을지 (steering 범위 -1.0 ~ 1.0 기준)

# 박스 면적이 전체 화면의 몇 %를 넘으면 "가깝다"/"매우 가깝다"로 볼지.
# 실측을 통해 나중에 튜닝이 필요한 값.
AVOID_AREA_RATIO = 0.05   # 화면의 5% 이상 -> 회피 시작
STOP_AREA_RATIO = 0.20    # 화면의 20% 이상 -> 정지


def get_closest_box(results):
    """
    탐지된 물체들 중, 화면에서 가장 큰(=가장 가까운) 박스 하나를 찾아서 돌려준다.
    탐지된 게 없으면 None.
    """
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    biggest_box = None
    biggest_area = 0

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = float((x2 - x1) * (y2 - y1))

        if area > biggest_area:
            biggest_area = area
            biggest_box = box

    return biggest_box


def decide_action(box, frame_width, frame_height):
    """
    가장 가까운 장애물 박스(box)를 보고, 어떻게 행동할지 결정한다.
    반환값: (steering 값, speed 값, 상태 설명 문자열)
    """
    if box is None:
        # 장애물이 없으면 그냥 직진.
        return 0.0, FORWARD_SPEED, "직진"

    x1, y1, x2, y2 = box.xyxy[0]
    box_area = float((x2 - x1) * (y2 - y1))
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    # 박스의 가로 중심이 화면 중심보다 왼쪽/오른쪽 어디에 있는지 계산.
    # -1(완전 왼쪽) ~ 0(중앙) ~ +1(완전 오른쪽)
    box_center_x = float((x1 + x2) / 2)
    frame_center_x = frame_width / 2
    center_ratio = (box_center_x - frame_center_x) / frame_center_x

    if area_ratio >= STOP_AREA_RATIO:
        # 너무 가까움 -> 정지.
        return 0.0, 0, "정지 (장애물 매우 근접)"

    elif area_ratio >= AVOID_AREA_RATIO:
        # 어느 정도 가까움 -> 장애물의 반대 방향으로 핸들을 꺾어서 피해감.
        # 장애물이 화면 왼쪽(center_ratio < 0)에 있으면 오른쪽으로 피함(+AVOID_STEER).
        # 장애물이 화면 오른쪽(center_ratio > 0)에 있으면 왼쪽으로 피함(-AVOID_STEER).
        if center_ratio < 0:
            steer = AVOID_STEER
        else:
            steer = -AVOID_STEER
        return steer, FORWARD_SPEED, "회피 중"

    else:
        # 장애물이 있긴 하지만 아직 멀어서 그냥 직진.
        return 0.0, FORWARD_SPEED, "직진 (장애물 감지되었으나 거리 있음)"


def main():
    # Jetson CSI 카메라 열기.
    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    Car.setSpeed(FORWARD_SPEED)

    print("장애물 회피 주행 시작. Ctrl+C로 종료하세요.")

    last_state = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_height, frame_width = frame.shape[:2]

            # 이 프레임에서 물체 탐지.
            results = model(frame, conf=0.5, verbose=False)

            # 가장 가까운(가장 큰) 장애물 하나만 기준으로 판단.
            closest_box = get_closest_box(results)

            steer, speed, state = decide_action(closest_box, frame_width, frame_height)

            # 실제 조향/속도 명령 적용.
            Car.steering = steer
            Car.setSpeed(speed)

            if speed == 0:
                Car.stop()
            else:
                Car.forward()

            # 상태가 바뀌었을 때만 출력 + 음성 알림 (매 프레임마다 말하면 시끄러움).
            if state != last_state:
                print(state)
                if state == "정지 (장애물 매우 근접)":
                    speak("장애물이 매우 가깝습니다. 정지합니다")
                elif state == "회피 중":
                    speak("장애물을 피해갑니다")
                last_state = state

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        Car.stop()
        cap.release()


if __name__ == "__main__":
    main()
