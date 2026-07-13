# ultralytics 라이브러리에서 YOLO 클래스를 가져온다.
# 이 YOLO 클래스가 "객체 탐지 AI 모델"을 다루는 역할을 함.
from ultralytics import YOLO

# YOLO 모델을 만든다.
# "yolov8n.pt" = YOLOv8의 nano(가장 작은/가벼운) 버전 가중치 파일.
# 파일이 없으면 인터넷에서 자동으로 다운로드된다.
# Jetson처럼 성능이 낮은 기기에서도 돌아가야 하므로 가장 가벼운 nano를 씀.
model = YOLO("yolov8n.pt")

# model(이미지)를 호출하면 그 이미지 안에서 물체를 찾아준다.
# conf=0.5 는 "신뢰도(확신 정도) 50% 이상인 것만 결과로 인정" 이라는 뜻.
# 너무 낮추면 잘못 탐지한 것도 섞이고, 너무 높이면 진짜 물체도 놓칠 수 있음.
results = model("https://ultralytics.com/images/bus.jpg", conf=0.5)

# results 는 리스트 형태라서 첫 번째 결과를 results[0] 로 꺼낸다.
# .save() 는 탐지된 박스가 그려진 이미지를 파일로 저장해주는 기능.
results[0].save(filename="result.jpg")

# results[0].boxes 안에는 탐지된 물체들이 하나씩(박스 단위로) 들어있다.
# for문으로 하나씩 꺼내서 확인한다.
for box in results[0].boxes:
    # box.cls[0] = 탐지된 물체의 "클래스 번호" (예: 0번 = 사람, 2번 = 자동차 ...)
    # int()로 정수로 바꿔줌 (원래는 텐서라는 숫자 배열 형태라서 변환 필요)
    cls_id = int(box.cls[0])

    # model.names 는 {번호: 이름} 형태의 딕셔너리.
    # 번호를 넣으면 "person", "car" 같은 실제 이름으로 바꿔준다.
    label = model.names[cls_id]

    # box.conf[0] = 이 탐지 결과가 얼마나 확신하는지(0~1 사이 값, 1이 100%)
    confidence = float(box.conf[0])

    # 사람이 읽기 편하게 출력. :.2f 는 소수점 둘째 자리까지만 표시.
    print(f"{label} 감지 (신뢰도: {confidence:.2f})")

print("완료! result.jpg 파일 확인하세요.")
