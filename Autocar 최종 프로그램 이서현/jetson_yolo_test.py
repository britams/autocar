# ============================================================
# Jetson(Python 3.6 + OpenCV 4.3) 전용 YOLOv3-tiny 테스트
#
# ultralytics(YOLOv8)는 Python 3.8 이상이 필요해서 이 오토카(Python 3.6.9)에는
# 설치할 수 없다. 대신 OpenCV의 dnn(딥러닝 실행) 기능으로
# 다크넷(Darknet) 형식의 가중치를 직접 불러와서 사용한다.
#
# [참고] 처음엔 yolov4-tiny로 시도했으나, OpenCV 4.3.0이 yolov4 계열의
#        최신 레이어 구조(CSP group 연산)를 지원하지 않아 탐지가 전혀 안 됐음
#        (신뢰도가 항상 0으로 나옴). OpenCV의 yolov4 정식 지원은 4.4부터라서,
#        더 단순한 구조라 OpenCV 4.3에서도 확실히 동작하는 yolov3-tiny로 변경함.
# ============================================================

import cv2
import numpy as np

# 이 세 파일이 있어야 함:
#   yolov3-tiny.cfg      -> 모델 구조 설명
#   yolov3-tiny.weights  -> 학습된 가중치(숫자들)
#   coco.names           -> 클래스 번호 -> 이름(사람, 자동차 등) 매핑표
CFG_PATH = "/home/soda/pop/model/yolov3-tiny/yolov3-tiny.cfg"
WEIGHTS_PATH = "/home/soda/pop/model/yolov3-tiny/yolov3-tiny.weights"
NAMES_PATH = "/home/soda/pop/model/yolov3-tiny/coco.names"

CONF_THRESHOLD = 0.5   # 이 값 이상 확신할 때만 탐지 결과로 인정
NMS_THRESHOLD = 0.4    # 겹치는 박스를 정리(중복 제거)할 때 쓰는 값

# ---- 클래스 이름 목록 읽기 ----
with open(NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---- 다크넷 모델 불러오기 ----
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)

# 아직 CUDA/TensorRT 가속은 안 걸고, 기본(CPU) 백엔드로 먼저 동작 확인.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 모델의 "출력을 내보내는 레이어" 이름들을 가져온다. (yolo 계열 특유의 구조)
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i[0] - 1] for i in output_layer_indices]


def detect(frame):
    """
    frame(카메라 이미지 한 장)을 입력받아,
    탐지된 물체들의 리스트 [(라벨, 신뢰도, (x, y, w, h)), ...] 를 돌려준다.
    """
    height, width = frame.shape[:2]

    # 이미지를 신경망 입력 형식(blob)으로 변환.
    # 416x416 크기로 리사이즈, 픽셀값을 0~1 사이로 정규화(1/255).
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 순전파(forward) 실행 -> 결과(출력 레이어들의 값) 받아옴.
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            # detection[0:4] = 박스 중심 x,y 와 너비,높이 (0~1 사이 비율)
            # detection[5:]  = 80개 클래스 각각에 대한 확신도 점수
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                # 비율(0~1)로 되어있는 좌표를 실제 픽셀 좌표로 변환.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 겹치는 박스 중 가장 확실한 것만 남기고 나머지 제거(NMS = Non-Max Suppression).
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        label = class_names[class_ids[i]]
        results.append((label, confidences[i], boxes[i]))

    return results


if __name__ == "__main__":
    # 정지 이미지로 먼저 테스트 (카메라 없이도 동작 확인 가능).
    # 오토카에 테스트용 이미지가 없다면, 카메라로 한 장 찍은 사진 경로로 바꿔서 테스트.
    test_image_path = "/home/soda/pop/model/yolov4-tiny/test.jpg"

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"이미지를 못 찾았습니다: {test_image_path}")
        print("테스트할 이미지 파일 경로를 test_image_path 에 넣어주세요.")
    else:
        results = detect(frame)
        print(f"탐지된 물체 수: {len(results)}")
        for label, conf, box in results:
            print(f"{label} (신뢰도 {conf:.2f}) 위치: {box}")
