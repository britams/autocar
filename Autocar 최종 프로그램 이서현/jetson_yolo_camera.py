# ============================================================
# Jetson 실시간 카메라 + YOLOv3-tiny 탐지
#
# jetson_yolo_test.py 에서 확인한 detect() 함수를,
# 정지 이미지가 아니라 실제 카메라 영상에 매 프레임 적용한다.
# ============================================================

import time

import cv2
import numpy as np

from pop import Util

CFG_PATH = "/home/soda/pop/model/yolov3-tiny/yolov3-tiny.cfg"
WEIGHTS_PATH = "/home/soda/pop/model/yolov3-tiny/yolov3-tiny.weights"
NAMES_PATH = "/home/soda/pop/model/yolov3-tiny/coco.names"

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

with open(NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
# CPU 대신 Jetson의 GPU(CUDA)로 연산을 넘겨서 속도를 높인다.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i[0] - 1] for i in output_layer_indices]


def detect(frame):
    """frame 한 장에서 물체를 찾아 [(라벨, 신뢰도, (x,y,w,h)), ...] 로 돌려준다."""
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        label = class_names[class_ids[i]]
        results.append((label, confidences[i], boxes[i]))

    return results


def main():
    # Jetson CSI 카메라 열기. (WSL 코드와 달리 실제 오토카에선 바로 이 방식만 씀)
    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("실시간 탐지 시작. Ctrl+C로 종료하세요.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = detect(frame)

            for label, conf, box in results:
                print(f"{label} (신뢰도 {conf:.2f}) 위치: {box}")

            # 몇 FPS로 처리되고 있는지 10프레임마다 한 번씩 출력.
            # (Jetson CPU로만 돌리면 느릴 수 있어서 실제 속도 확인용)
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[FPS: {fps:.1f}]")

    except KeyboardInterrupt:
        print("종료합니다.")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
