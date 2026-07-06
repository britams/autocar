import cv2  # 버전 4.12
from pop import Pilot, Util

car = Pilot.AutoCar()

car.forward()
car.backward()
car.joystick()
car.camPan(10)
car.camTilt(10)

value = car.getGyro()
print(value)

cam = Util.gstrmer(width=640, height=480, fps=30)

cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Not found camera")

for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.Canny(frame, 100, 200)
    cv2.imshow("soda", img)
cap.release()
# cv2.destroyAllWindows()



# ---

import cv2  # 버전 4.12
from pop import Util

Util.enable_imshow()

m = Util.gstrmer(width=640, height=480, fps=30)

haar_face = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_face)
cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Not found camera")
try:
    for _ in range(12000):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(100, 100))
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y, w, h), (255, 0, 0), 2)
            cv2.putText(frame, "FACE", (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("soda", frame)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
# cv2.destroyAllWindows()


# --- 음성 재생

import wave

import pyaudio

p = pyaudio.PyAudio()
with wave.open("/usr/share/sounds/alsa/Side_Left.wav", "rb") as w:
    data = w.readframes(w.getnframes())
    stream = p.open(format=p.get_format_from_width(2), channels=1, rate=48000, output=True)
    stream.write(data)
    stream.stop_stream()
    stream.close()

# --- 음성 재생 논 블러킹 모드 ( multi-thread , 내부 기능)

import time
import wave

import pyaudio

WAV_PATH = "/usr/share/sounds/alsa/Side_Left.wav"

p = pyaudio.PyAudio()

# 출력 가능한 장치 확인
print("=== Output devices ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxOutputChannels"] > 0:
        print(i, info["name"], info["maxOutputChannels"], info["defaultSampleRate"])

# 여기 장치 번호를 필요하면 바꾸세요.
# 위 출력에서 스피커 장치 번호를 보고 지정합니다.
OUTPUT_DEVICE_INDEX = None
# 예: OUTPUT_DEVICE_INDEX = 0
# 예: OUTPUT_DEVICE_INDEX = 1

w = wave.open(WAV_PATH, "rb")

def play_cb(in_data, frame_count, time_info, status):
    data = w.readframes(frame_count)

    if len(data) == 0:
        return (data, pyaudio.paComplete)

    return (data, pyaudio.paContinue)

stream = p.open(
    format=p.get_format_from_width(w.getsampwidth()),
    channels=w.getnchannels(),
    rate=w.getframerate(),
    output=True,
    output_device_index=OUTPUT_DEVICE_INDEX,
    stream_callback=play_cb
)

stream.start_stream()

while stream.is_active():
    print("main work...")
    time.sleep(0.1)

stream.stop_stream()
stream.close()
w.close()
p.terminate()



# sine 함수로 음 출력하기

import numpy as np
import pyaudio

volume = 0.5
fs = 48000
duration = 5.0
f = 440.0       # 라 음

data = (np.sin(2 * np.pi * np.arange(fs*duration) * f/fs)).astype(np.float32)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
stream.write(volume * data)

stream.stop_stream()
stream.close()
p.terminate()