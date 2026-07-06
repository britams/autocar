# /video 카메라 정보가 나오는 것
# flask로 서버를 열 것.
# #cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
# cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

# for _ in range(120):
#     ret, frame = cap.read()
#     if not ret:
#         print(ret)
#         continue
#     cv2.imshow("frame", frame)
# cap.release()
# 위 코드를 기준으로 작성
# jpd 압축 60% 해서 데이터 보내기
# 5000번 포트로 api 열기
# 쓰레드 사용

import cv2
import threading
from flask import Flask, Response

app = Flask(__name__)

# 카메라에서 읽은 최신 프레임을 저장하는 변수
latest_frame = None
lock = threading.Lock()


def capture_camera():
    """카메라에서 계속 프레임을 읽어오는 쓰레드"""
    global latest_frame

    # Jetson 환경이면 GStreamer 사용, 아니면 일반 웹캠(0번)
    try:
        from pop import Util
        cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
        cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
    except Exception:
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # JPEG 60% 품질로 압축
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

        with lock:
            latest_frame = encoded.tobytes()


def generate():
    """브라우저로 영상을 스트리밍하는 함수"""
    while True:
        with lock:
            frame = latest_frame

        if frame is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


@app.route('/video')
def video():
    """/video 주소로 접속하면 카메라 영상 스트리밍"""
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    t = threading.Thread(target=capture_camera, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000)
