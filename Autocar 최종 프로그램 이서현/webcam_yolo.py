# ultralytics 에서 YOLO 클래스를 가져온다 (객체 탐지 AI).
from ultralytics import YOLO
# opencv: 카메라 열기, 화면에 영상 띄우기, 이미지 처리에 쓰는 라이브러리.
import cv2
# speak(): 음성 알림 기능. tts_util.py 에 공통으로 모아둠 (다른 파일에서도 재사용).
from tts_util import speak

# YOLOv8 nano(가장 가벼운) 모델을 불러온다. Jetson에서도 부담 없이 돌아가는 크기.
model = YOLO("yolov8n.pt")

# ---- 카메라 열기 ----
# camera_api.py 에 있던 것과 같은 방식: Jetson이면 CSI 카메라(GStreamer)를 쓰고,
# 아니면(WSL, 일반 PC) 그냥 USB 웹캠(0번 장치)을 쓴다.
try:
    # pop 라이브러리는 실제 오토카(Jetson)에만 설치되어 있음.
    # WSL에는 이 라이브러리가 없으므로 import가 실패 -> except로 넘어감.
    from pop import Util
    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
except Exception:
    # 0번 = 컴퓨터에 연결된 첫 번째 웹캠.
    cap = cv2.VideoCapture(0)

# 카메라 압축 포맷을 MJPG로 강제 지정.
# (usbipd로 WSL에 연결한 웹캠은 기본 YUYV 포맷을 OpenCV가 잘못 해석해서
#  화면이 초록색으로만 나오는 경우가 있어서 명시적으로 지정해줌)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 카메라가 정상적으로 열렸는지 확인.
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 웹캠 연결을 확인하세요.")
    exit()

print("실시간 탐지 시작. 종료하려면 'q' 키를 누르세요.")

# ---- 실시간 탐지 반복문 ----
while True:
    # cap.read() 는 카메라에서 프레임(사진 한 장) 하나를 읽어온다.
    # ret 은 성공 여부(True/False), frame 은 실제 이미지 데이터.
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    # 이 프레임 한 장에서 물체 탐지 실행.
    # conf=0.5 -> 신뢰도 50% 이상인 것만 결과로 사용.
    # verbose=False -> 매 프레임마다 로그 줄줄이 안 찍히게 함.
    results = model(frame, conf=0.5, verbose=False)

    # 탐지된 박스가 그려진 이미지를 만들어준다.
    annotated_frame = results[0].plot()

    # 화면에 결과 영상을 띄운다. (WSLg 덕분에 WSL에서도 창이 뜬다)
    cv2.imshow("YOLO 실시간 탐지", annotated_frame)

    # 탐지된 물체 중 사람(person)이 있으면 콘솔에 알림 출력.
    # -> 나중에 여기서 TTS 호출, 정지 명령 등으로 연결하면 됨.
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == "person":
            print("사람(장애물) 감지!")
            speak("전방에 장애물이 있습니다")

    # 1밀리초 동안 키 입력을 기다리고, 'q' 를 누르면 반복문 탈출.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 장치를 놓아주고, 열린 창을 모두 닫는다. (마무리 정리)
cap.release()
cv2.destroyAllWindows()
