import cv2
import threading
from flask import Flask, Response, render_template_string
from pop import Util

app = Flask(__name__)

# 일반 영상 / 외곽선 영상을 각각 저장하는 변수
latest_normal = None
latest_outline = None
lock = threading.Lock()


def capture_camera():
    """카메라에서 한 번만 읽어서 일반 영상 + 외곽선 영상 둘 다 만드는 쓰레드"""
    global latest_normal, latest_outline

    cam = Util.gstrmer(width=640, height=480)
    camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not camera.isOpened():
        print("Not found camera")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        outline = cv2.Canny(frame, 100, 200)

        _, normal_encoded = cv2.imencode('.jpg', frame)
        _, outline_encoded = cv2.imencode('.jpg', outline)

        with lock:
            latest_normal = normal_encoded.tobytes()
            latest_outline = outline_encoded.tobytes()


def generate(get_frame):
    """브라우저로 영상을 스트리밍하는 함수"""
    while True:
        with lock:
            frame = get_frame()

        if frame is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


@app.route('/')
def index():
    """일반 영상 + 외곽선 영상을 한 페이지에 보여줌"""
    return render_template_string("""
    <html>
      <body style="background:#222; text-align:center;">
        <h2 style="color:white;">일반 영상</h2>
        <img src="/video" width="480">
        <h2 style="color:white;">외곽선 영상</h2>
        <img src="/outline" width="480">
      </body>
    </html>
    """)


@app.route('/video')
def video():
    """일반 카메라 영상"""
    return Response(
        generate(lambda: latest_normal),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/outline')
def outline():
    """외곽선 검출 영상"""
    return Response(
        generate(lambda: latest_outline),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    t = threading.Thread(target=capture_camera, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000)
