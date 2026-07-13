# ============================================================
# 카메라(YOLO) + 라이다 센서 퓨전
#
# [중요] 이 코드는 실제 오토카(Jetson)에서만 실행 가능합니다.
#        WSL에는 라이다 드라이버(_rplidar.so, ARM 전용 바이너리)와
#        실제 라이다 장치가 없어서 여기서는 테스트할 수 없습니다.
# ============================================================

from ultralytics import YOLO
import cv2
import time

# Jetson 전용 라이브러리. WSL에서는 import 자체가 실패한다.
from pop import LiDAR, Util

# ---- 설정값 (카메라 사양에 맞게 나중에 실측해서 바꿔야 함) ----
CAMERA_WIDTH = 640          # 카메라 가로 해상도(픽셀)
CAMERA_FOV_DEG = 62.2       # 카메라의 가로 화각(도). 라즈베리파이캠 v2 기준 대략값.
                             # 실제 카메라 스펙에 맞게 나중에 수정 필요.
ANGLE_TOLERANCE = 5         # 물체 방향 기준 +-몇 도까지를 "같은 방향"으로 볼지
STOP_DISTANCE_MM = 400      # 이 거리(mm) 이내면 "위험"으로 판단 (예: 40cm)


def pixel_x_to_angle(pixel_x):
    """
    화면 속 가로 픽셀 위치(pixel_x)를,
    카메라 정면을 0도로 하는 "좌우 각도"로 변환한다.

    예: 화면 정중앙(320) -> 0도
        화면 맨 왼쪽(0)   -> -FOV/2 도 (마이너스 = 왼쪽)
        화면 맨 오른쪽(640)-> +FOV/2 도 (플러스 = 오른쪽)
    """
    center = CAMERA_WIDTH / 2
    # 픽셀이 중심에서 얼마나 떨어져 있는지 비율(-0.5 ~ 0.5)로 구하고
    # 그 비율만큼 전체 화각(FOV)을 곱해서 각도로 변환.
    ratio = (pixel_x - center) / CAMERA_WIDTH
    return ratio * CAMERA_FOV_DEG


def find_lidar_distance(lidar, target_angle):
    """
    target_angle(도) 근처에 있는 라이다 포인트들 중,
    가장 가까운 거리(mm)를 찾아서 돌려준다.
    해당 방향에 데이터가 없으면 None.
    """
    # getVectors() -> [[각도, 거리, 신뢰도], ...] 형태의 배열을 준다.
    vectors = lidar.getVectors()

    nearest_distance = None

    for angle, distance, quality in vectors:
        # 라이다 각도와 목표 각도의 차이를 계산.
        # 0~360도로 표현되므로 360도를 넘나드는 경우도 있을 수 있어 보정.
        diff = abs(angle - target_angle)
        diff = min(diff, 360 - diff)

        if diff <= ANGLE_TOLERANCE:
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance

    return nearest_distance


def main():
    # ---- YOLO 모델 준비 ----
    model = YOLO("yolov8n.pt")  # 경량 nano 모델

    # ---- 카메라 준비 (Jetson CSI 카메라) ----
    cam = Util.gstrmer(width=CAMERA_WIDTH, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    # ---- 라이다 준비 ----
    lidar = LiDAR.Rplidar()
    lidar.connect()
    lidar.startMotor()

    print("센서 퓨전 시작. Ctrl+C로 종료하세요.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 이 프레임에서 물체 탐지.
            results = model(frame, conf=0.5, verbose=False)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # 박스 좌표(x1,y1,x2,y2) 중, 가로 중심 픽셀만 필요.
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = float((x1 + x2) / 2)

                # 픽셀 위치 -> 각도로 변환.
                angle = pixel_x_to_angle(center_x)

                # 그 각도 방향의 라이다 거리 찾기.
                distance = find_lidar_distance(lidar, angle)

                if distance is not None:
                    print(f"{label} | 방향: {angle:.1f}도 | 거리: {distance:.0f}mm")

                    # 너무 가까우면 위험 알림.
                    # -> 여기에 정지 명령(Car.stop()) + TTS 알림을 연결하면 됨.
                    if distance < STOP_DISTANCE_MM:
                        print(f"[경고] {label} 이(가) {distance:.0f}mm 앞에 있습니다! 정지 필요.")
                else:
                    print(f"{label} | 방향: {angle:.1f}도 | 라이다 데이터 없음")

            time.sleep(0.05)  # 너무 빠르게 반복하지 않도록 약간 쉬어줌

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        # 사용한 장치들을 깔끔하게 정리.
        cap.release()
        lidar.stopMotor()
        lidar.destroy()


if __name__ == "__main__":
    main()
