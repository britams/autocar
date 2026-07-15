# 폰 웹뷰 하나로 전부 통합:
#   - MANUAL 모드: 폰 화면 방향키로 직접 조종
#   - AUTO 모드: 카메라+YOLO로 사람을 추적해서 따라감
#   - LINE 모드: OpenCV 색상 검출로 트랙 두 선 사이 중앙을 유지하며 주행
#   - 세 모드 공통: 라이다가 좌우 근접 장애물(700mm 이내)을 감지하면
#                   YOLO/라인/폰 조작보다 항상 라이다 회피가 우선 (안전이 최우선)
#
# [카메라는 하나뿐] YOLO(AUTO)와 라인트레이싱(LINE)을 별도 프로그램/프로세스로
# 동시에 띄우면 카메라를 서로 못 잡아서 충돌난다. 그래서 이 프로그램 하나 안에서
# 모드만 전환하는 방식으로 만들었다 (카메라는 계속 하나만 열려있음).
#
# 화면 스트리밍(웹으로 카메라 보기)은 뺐음 - 폰 리모컨 대시보드만 사용.
#
# [중요] pycuda(YOLO의 GPU 연산)는 처음 초기화된 스레드에서만 동작하므로
# 카메라/YOLO 처리(hardware_engine)는 반드시 메인 스레드에서 실행해야 함.
# 그래서 Flask 서버 쪽을 별도 스레드로 돌린다 (기존 app.py와 순서 반대).
#
# 실행 위치 주의: 반드시 ~/tensorrt_demos 폴더 안에서 실행해야 함
# (YOLO 모델/플러그인 파일들이 그 폴더 기준 상대경로로 되어있음).
import json
import os
import subprocess
import sys
from collections import deque
import threading
import time

sys.path.insert(1, '.')

import cv2
import numpy as np
import pycuda.autoinit  # noqa
from flask import Flask, render_template, request, jsonify, redirect, Response
from flask_socketio import SocketIO

from pop import Pilot, Cds
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_cls_dict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'autocar_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

Car = Pilot.AutoCar()

# ---- LED 설정 ----
# 카메라 팬/틸트를 제어하는 것과 같은 PWM 칩(I2C bus 1, 주소 0x5c)의 다른 채널.
# 채널 하나씩 켜보고 실물로 직접 확인해서 확정한 매핑 (2026-07-15).
LED_CHANNELS = {
    "front_right": 0,
    "front_left": 1,
    "rear_right": 2,
    "rear_left": 3,
}
led_pwm = Pilot.PWM(1, 0x5c)
led_pwm.setFreq(50)


def led_set(on):
    duty = 99 if on else 0
    for ch in LED_CHANNELS.values():
        led_pwm.setDuty(ch, duty)


# 조도센서: SPI ADC로 연결된 pop.Cds, 채널 7이 실측으로 확정됨 (2026-07-15).
# 방향은 실측(손으로 가리기)으로 맞춤: 값이 threshold 이하일 때 "어둡다"로 판단해서 LED 켬.
# 정확한 밝다/어둡다 기준값은 실제 새벽/낮 환경에서 다시 재보고 조정 필요.
light_sensor = Cds(7)
LIGHT_DARK_THRESHOLD = 1600
LIGHT_POLL_INTERVAL_SEC = 2.0  # 너무 자주 읽으면 부하만 커서 이 간격으로만 확인

telemetry = {
    "steering": 0, "speed": 0,
    "mode": "MANUAL", "status": "READY",
    "obstacle": "CLEAR", "action_text": "STOP",
    "cam_tilt": 0,
    "led_mode": "AUTO", "led_on": False,  # led_mode: AUTO(조도센서 자동) / ON / OFF(수동 고정)
}

# 카메라 상하 각도(tilt)는 자동 동작 없이 수동 버튼으로만 조작한다.
# Car.camTilt(value)는 문서 기준 -30~200 범위의 절대각도를 그대로 받는다.
CAM_TILT_MIN = -30
CAM_TILT_MAX = 200
CAM_TILT_STEP = 10

# ---- YOLO 사람 추적 설정 ----
CONF_THRESH = 0.3
FOLLOW_SPEED = 75
SLOW_SPEED = 40
STEER_GAIN = 1.2
MAX_STEER = 1.0

TOO_FAR_RATIO = 0.08
GOOD_DISTANCE_RATIO = 0.2
TOO_CLOSE_RATIO = 0.35

EMA_ALPHA = 0.35
MISS_GRACE_FRAMES = 6
MIN_COMMAND_INTERVAL_SEC = 0.4

# ---- 사람을 놓쳤을 때 카메라를 천천히 좌우로 돌려서 찾는 설정 ----
# 카메라 서보 실제 각도는 0~180도, 90도가 정중앙.
# 1렙(rep) = 90(중앙) -> 0(왼쪽 끝) -> 180(오른쪽 끝) -> 90(중앙) 순서로 한 바퀴.
# 이걸 SEARCH_MAX_CYCLES번 반복해도 못 찾으면 자율주행(AUTO) 모드를 끈다.
# 카메라는 수평(pan)으로만 움직이고 수직(tilt)은 절대 사용하지 않는다.
# [중요] 90(중앙)으로 맞추는 것도 순간 점프가 아니라 다른 구간과 똑같이 한 스텝씩
# 천천히 움직여서 도달한다 (인식이 깜빡거려 탐색이 자주 재시작돼도 확 튀지 않도록).
SEARCH_PAN_CENTER = 90                    # 실제 서보 각도 기준 정중앙
SEARCH_PAN_WAYPOINTS = [90, 0, 180, 90]   # 1렙 동안 순서대로 찾아갈 목표 각도
SEARCH_PAN_STEP = 1                       # 한 번에 몇 도씩 움직일지 (아주 천천히 돌도록 작게)
SEARCH_PAN_INTERVAL_SEC = 0.25            # 이 시간마다 한 스텝씩
SEARCH_MAX_CYCLES = 20                    # 몇 렙까지 반복하고 포기할지

# ---- 라이다 안전 회피 설정 (MANUAL/AUTO 공통, 항상 우선) ----
LIDAR_RANGE_MM = 700
LIDAR_FRONT_DEG = 45
LIDAR_MIN_POINTS = 3

# ---- 카메라 노출(밝기) 설정 ----
# 라인트레이싱 때 강한 햇빛(역광)으로 화면이 하얗게 날아가서 선/트랙 색 구분이
# 안 되던 문제가 있었음 -> 카메라 노출을 낮춰서(어둡게 찍어서) 색이 뭉개지지
# 않고 살아있게 만든다. -2(가장 어둡게) ~ 2(가장 밝게) 범위, 0이 기본값.
EXPOSURE_COMPENSATION = -1.5

# ---- 라인트레이싱 설정 ----
# [2026-07-15 변경] 절대 밝기 기준(V>=170)이 노출값/날씨에 따라 계속 틀어져서
# (흐린 날 노출을 낮춰놓으니 흰 선도 V=170을 못 넘어 아예 인식 자체가 안 됐음),
# "주변 트랙보다 상대적으로 밝은가"를 보는 adaptiveThreshold 방식으로 교체.
# 노출이 바뀌어도 선-트랙 간 상대적 밝기 차이는 유지되므로 더 안정적.
LINE_ROI_TOP_RATIO = 0.45  # 화면 위쪽 45%(하늘/건물)는 아예 제외하고 트랙 바닥만 봄 -> 건물 윤곽선을 선으로 오인하던 문제 해결

# [2026-07-15] 처리 속도가 차 속도를 못 따라가서 선을 놓치는 문제 -> 절반 해상도로 계산해서 속도 확보.
# 아래 블록/커널 크기는 전부 이 절반 해상도 기준으로 다시 튜닝한 값.
LINE_DOWNSCALE = 0.5
LINE_ADAPTIVE_BLOCK = 25   # 주변을 얼마나 넓게 보고 평균 밝기를 잴지 (홀수, 절반 해상도 기준)
LINE_ADAPTIVE_C = -4       # 이 값(절대값)만큼 주변 평균보다 밝아야 선으로 인정 (실측 사진에서 -4가 노이즈 없이 선만 잡음)
LINE_MIN_BLOB_AREA = 40    # 절반 해상도라 픽셀 수도 1/4로 줄어듦 (원래 기준 150의 1/4)
LINE_FORWARD_SPEED = 60  # 모터 특성상 50 미만이면 실제로 안 굴러감 (경험적으로 확인된 값)
LINE_STEER_GAIN = 1.5
LINE_MAX_STEER = 1.0

# 라인 디버그용: 계속 스트리밍하지 않고, 요청할 때마다(새로고침) 최신 한 장만 보여준다.
# hardware_engine 루프가 LINE 모드일 때마다 이 변수를 최신 프레임+마스크로 갱신한다.
line_debug_lock = threading.Lock()
line_debug_jpeg = None


@app.route('/')
def index():
    return redirect('/remote')


@app.route('/remote')
def remote_view():
    # 폰마다 예전 버전 페이지가 캐시되어 새 버튼/기능이 안 보이는 문제 방지 -> 캐시 금지.
    resp = Response(render_template('remote.html'), mimetype='text/html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/api/status')
def api_status():
    return jsonify(telemetry)


@app.route('/api/control')
def api_control():
    global telemetry
    if telemetry['mode'] == "MANUAL" or telemetry['mode'] == "MANUAL_NOLIDAR":
        if telemetry['status'] == "EMERGENCY_STOP":
            telemetry['status'] = "READY"
        try:
            telemetry['steering'] = int(request.args.get('steering', 0))
            telemetry['speed'] = int(request.args.get('speed', 0))
            if telemetry['speed'] > 0: telemetry['action_text'] = "FORWARD"
            elif telemetry['speed'] < 0: telemetry['action_text'] = "BACKWARD"
            else: telemetry['action_text'] = "STOP"
        except: pass
    return jsonify(telemetry)


@app.route('/api/mode')
def api_mode():
    global telemetry
    telemetry['mode'] = request.args.get('mode', 'MANUAL')
    telemetry['steering'] = 0; telemetry['speed'] = 0; telemetry['status'] = "READY"

    # 라인트레이싱은 카메라가 정면(수평 90도, 중앙) + 아래쪽(-10도)을 보고 있어야
    # 트랙 인식이 안정적이므로, LINE 모드로 바꾸는 순간 카메라를 자동으로 정렬한다.
    if telemetry['mode'] == "LINE":
        try:
            Car.camPan(90)
            telemetry['cam_pan'] = 90
            Car.camTilt(-10)
            telemetry['cam_tilt'] = -10
        except Exception as e:
            print(f"[카메라 정렬 에러]: {e}")

    return jsonify(telemetry)


@app.route('/api/led')
def api_led():
    """mode=AUTO(조도센서 자동)/ON(수동 켬)/OFF(수동 끔)."""
    global telemetry
    mode = request.args.get('mode', 'AUTO')
    telemetry['led_mode'] = mode
    if mode == "ON":
        telemetry['led_on'] = True
        led_set(True)
    elif mode == "OFF":
        telemetry['led_on'] = False
        led_set(False)
    # AUTO면 hardware_engine 루프가 조도센서 값 보고 알아서 켜고 끔
    return jsonify(telemetry)


@app.route('/api/camtilt')
def api_camtilt():
    """카메라 상하 각도 수동 조작 (자동 동작 없음, 버튼으로만).
    direction=up/down으로 CAM_TILT_STEP만큼 움직이거나, value=<정수>로 절대각도 지정."""
    global telemetry
    direction = request.args.get('direction')
    if direction == 'up':
        telemetry['cam_tilt'] = min(CAM_TILT_MAX, telemetry['cam_tilt'] + CAM_TILT_STEP)
    elif direction == 'down':
        telemetry['cam_tilt'] = max(CAM_TILT_MIN, telemetry['cam_tilt'] - CAM_TILT_STEP)
    else:
        try:
            telemetry['cam_tilt'] = max(CAM_TILT_MIN, min(CAM_TILT_MAX, int(request.args.get('value', 0))))
        except (TypeError, ValueError):
            pass
    try:
        Car.camTilt(telemetry['cam_tilt'])
    except Exception as e:
        print(f"[카메라 tilt 에러]: {e}")
    return jsonify(telemetry)


@app.route('/api/kill')
def api_kill():
    global telemetry
    telemetry['status'] = "EMERGENCY_STOP"
    telemetry['mode'] = "MANUAL"; telemetry['speed'] = 0; telemetry['steering'] = 0
    telemetry['action_text'] = "EMERGENCY STOP"
    print("[긴급 제동] 스마트폰 킬 스위치 작동! 차량 즉시 정지")
    return jsonify(telemetry)


@app.route('/debug/line.jpg')
def debug_line():
    """LINE 모드일 때 카메라가 뭘 보고 있는지 + 인식된 선(초록색)을 확인하는 디버그 이미지.
    계속 스트리밍하는 게 아니라, 새로고침할 때마다 최신 한 장만 보여준다."""
    with line_debug_lock:
        jpeg = line_debug_jpeg
    if jpeg is None:
        return "아직 이미지 없음 (LINE 모드로 전환하고 잠시 기다려주세요)", 404
    resp = Response(jpeg, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


# 라이다는 완전히 별도의 OS 프로세스(lidar_reader.py)로 분리해서 돌린다.
# _rplidar.so가 카메라/YOLO(GPU)와 같은 프로세스 안에서 동시에 활동하면
# 가끔 GIL을 붙잡은 채로 완전히 멈춰버려서(스레드/타임아웃으로도 못 막음)
# 폰 조종까지 전부 먹통이 되는 문제가 있었음.
# -> 별도 프로세스에서 죽거나 멈춰도 이쪽 메인 프로그램은 전혀 영향 없음.
LIDAR_STATE_FILE = "/tmp/lidar_state.json"
LIDAR_STATE_MAX_AGE_SEC = 1.0  # 이보다 오래된 데이터면 라이다 프로세스가 죽은 것으로 보고 무시
_lidar_process = None


def start_lidar_process():
    global _lidar_process
    try:
        os.remove(LIDAR_STATE_FILE)
    except OSError:
        pass
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lidar_reader.py')
    _lidar_process = subprocess.Popen(['python3', '-u', script_path])
    print("[라이다] 별도 프로세스로 시작함 (메인 프로그램과 완전히 분리됨)")


def read_obstacles():
    """lidar_reader.py가 써놓은 최신 상태 파일을 읽기만 한다 (라이다와 직접 통신 안 함)."""
    try:
        with open(LIDAR_STATE_FILE) as f:
            data = json.load(f)
        if time.time() - data.get('ts', 0) > LIDAR_STATE_MAX_AGE_SEC:
            return False, False  # 데이터가 오래됨 -> 라이다 프로세스가 멈췄거나 죽음
        if not data.get('connected'):
            return False, False
        return bool(data.get('left')), bool(data.get('right'))
    except Exception:
        return False, False


def gstreamer_pipeline(width=640, height=480, fps=30, flip=0, exposure_compensation=0.0):
    """pop.Util.gstrmer()와 거의 같지만 노출(exposurecompensation)을 조절할 수 있게
    직접 만든 파이프라인. 값이 음수일수록 어둡게 찍어서 햇빛에 화면이 날아가는 걸 줄인다."""
    return (
        "nvarguscamerasrc exposurecompensation=%.1f ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        % (exposure_compensation, width, height, fps, flip, width, height)
    )


def get_line_center_x(frame, last_x=None):
    """라인트레이싱: 트랙 중앙의 흰 점선 하나를 추적해서 x좌표를 찾는다.
    (양옆 레인 경계선이 아니라 중앙 점선을 따라가는 방식 - 2026-07-15 확정)
    last_x: 직전에 따라가던 선의 x좌표(있으면). 장애물 회피 등으로 잠깐 놓쳤다가 다시 찾을 때,
    엉뚱한 다른 선이 아니라 원래 따라가던 선으로 복귀하도록 후보 선택에 참고한다.
    반환값: (center_x 또는 None, mask, roi_top) - mask/roi_top은 디버그 화면 만드는 데 씀."""
    height, width = frame.shape[:2]
    roi_top = int(height * LINE_ROI_TOP_RATIO)
    roi = frame[roi_top:height, :]
    roi_h, roi_w = roi.shape[:2]

    # 처리 속도 확보를 위해 절반 해상도로 축소해서 계산 (아래 커널 크기들도 이 기준으로 튜닝됨)
    small = cv2.resize(roi, None, fx=LINE_DOWNSCALE, fy=LINE_DOWNSCALE, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        LINE_ADAPTIVE_BLOCK, LINE_ADAPTIVE_C,
    )

    # 반사/그림자 경계에서 생기는 점모양 노이즈 제거 (진짜 선은 뭉쳐있고, 노이즈는 흩어져있음)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 트랙 페인트가 닳아서 점선처럼 끊어져 있으므로, 세로로 넉넉하게 팽창시켜
    # 같은 선의 끊어진 조각들을 하나의 덩어리로 이어붙인다.
    connect_kernel = np.ones((13, 5), np.uint8)
    merged = cv2.dilate(mask, connect_kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    # 노이즈(균열 등) 제외하고 남은 후보 중, 가장 큰(=가장 길게 이어진) 덩어리를 선으로 채택.
    # 노이즈(균열 등) 제외하고 남은 후보 중, 가장 큰(=가장 길게 이어진) 덩어리를 선으로 채택.
    # 비슷하게 큰 덩어리가 여러 개면 그중 차와 가장 가까운(화면 아래쪽) 것을 우선한다.
    # [주의] last_x(직전 위치)만 보고 크기를 무시하면, 진짜 선보다 작은 노이즈/다른 선 조각에
    # 위치만 가깝다는 이유로 눌러붙는 더 나쁜 문제가 생겨서(2026-07-15 실기 확인) 이 방식은 폐기.
    candidates = []
    for i in range(1, n_labels):
        x, y, w_, h_, area = stats[i]
        if area < LINE_MIN_BLOB_AREA:
            continue
        candidates.append((i, x + w_ / 2, y + h_, area))

    center_x = None
    final_mask = np.zeros_like(mask)
    if candidates:
        max_area = max(c[3] for c in candidates)
        big_enough = [c for c in candidates if c[3] >= max_area * 0.7]
        if last_x is not None:
            chosen_label = min(big_enough, key=lambda c: abs(c[1] - last_x / LINE_DOWNSCALE))[0]
        else:
            chosen_label = max(big_enough, key=lambda c: c[2])[0]

        # 팽창시키기 전 원본 마스크 픽셀만으로 정확한 선 위치를 다시 계산 (디버그 화면도 원본 굵기로 표시)
        region = (labels == chosen_label)
        final_mask[region] = mask[region]
        ys, xs = np.where(final_mask > 0)
        if len(xs) > 0:
            center_x = float(np.mean(xs)) / LINE_DOWNSCALE

    # 디버그 화면은 원래 ROI 크기로 다시 키워서 보여준다 (계산 자체는 축소 해상도로 함)
    final_mask = cv2.resize(final_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return center_x, final_mask, roi_top


def get_person_box(boxes, confs, clss, cls_dict, last_center=None):
    """사람이 여러 명 잡히면, 원래 따라가던 사람(last_center와 가장 가까운 사람)을 우선 채택.
    last_center가 없으면(처음 찾을 때) 제일 크게(가깝게) 잡힌 사람을 채택한다."""
    people = []
    for box, conf, cls_id in zip(boxes, confs, clss):
        if cls_dict.get(int(cls_id), "") != "person":
            continue
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        people.append((box, conf, area, center))

    if not people:
        return None, 0, None

    if last_center is not None:
        lx, ly = last_center
        chosen = min(people, key=lambda p: (p[3][0] - lx) ** 2 + (p[3][1] - ly) ** 2)
    else:
        chosen = max(people, key=lambda p: p[2])

    return chosen[0], chosen[1], chosen[3]


def decide_follow_action(area_ratio, center_ratio):
    steer = max(-MAX_STEER, min(MAX_STEER, center_ratio * STEER_GAIN))
    if area_ratio < TOO_FAR_RATIO:
        return steer, FOLLOW_SPEED, "FOLLOW (FAR)"
    elif area_ratio < GOOD_DISTANCE_RATIO:
        return steer, SLOW_SPEED, "FOLLOW (SLOW)"
    elif area_ratio < TOO_CLOSE_RATIO:
        return steer, 0, "FOLLOW (CLOSE-STOP)"
    else:
        return 0.0, 0, "FOLLOW (TOO CLOSE-STOP)"


def hardware_engine():
    """[메인 스레드] 카메라+YOLO 인식 -> MANUAL/AUTO 처리 -> 모터 명령.
    pycuda 때문에 반드시 메인 스레드에서 실행."""
    global telemetry, line_debug_jpeg

    trt_yolo = TrtYOLO('yolov3-tiny-416', category_num=80)
    cls_dict = get_cls_dict(80)

    cam = gstreamer_pipeline(width=640, height=480, fps=30, flip=0, exposure_compensation=EXPOSURE_COMPENSATION)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("[시스템] 카메라+YOLO 준비 완료.")

    if os.environ.get("DISABLE_LIDAR") == "1":
        print("[라이다] DISABLE_LIDAR=1 -> 라이다 건너뜀 (안전 회피 비활성화)")
    else:
        # 별도 프로세스로 라이다를 띄운다 (죽거나 멈춰도 이 프로그램엔 영향 없음).
        start_lidar_process()

    print("[시스템] 하드웨어 엔진 시동 완료! (http://<오토카IP>:5000/remote)")

    smoothed_area = None
    smoothed_center = None
    miss_count = 0
    last_speed = None
    last_command_time = 0.0

    search_angle = SEARCH_PAN_CENTER   # 카메라 서보의 현재 실제 각도(0~180)
    search_waypoint_idx = 0            # SEARCH_PAN_WAYPOINTS 중 지금 향하는 목표 인덱스
    last_pan_time = 0.0
    is_searching = False
    search_cycles_done = 0
    found_count = 0
    FOUND_CONFIRM_COUNT = 3  # 탐색 중 사람을 "진짜로" 다시 찾았다고 인정하는 데 필요한 연속 프레임 수

    # LINE 모드에서 선을 놓쳤을 때 재탐색하는 상태 (AUTO의 사람 탐색과 같은 방식, 카메라 좌우로 훑기)
    line_search_angle = SEARCH_PAN_CENTER
    line_search_waypoint_idx = 0
    line_is_searching = False
    line_search_cycles_done = 0
    line_found_count = 0
    line_lost_count = 0
    last_line_x = None  # 직전에 따라가던 선 위치 (장애물 회피 후 원래 선으로 복귀할 때 씀)
    line_search_waypoints = SEARCH_PAN_WAYPOINTS  # 재탐색 시 어느 방향부터 훑을지 (아래서 최근 추세로 정해짐)
    line_error_history = deque()  # (시각, 화면중앙 대비 선의 좌우 오차) 최근 기록, 재탐색 방향 판단용
    LINE_HISTORY_WINDOW_SEC = 30  # 이 시간(초)만큼만 최근 기록으로 보고 그 이전 건 버림
    LINE_TREND_THRESHOLD = 0.15   # 이 이상 한쪽으로 치우쳐 있었어야 "그쪽으로 가던 중"이라고 판단

    last_person_center = None  # 직전에 따라가던 사람의 화면상 위치 (사람 여럿일 때 원래 사람 계속 추적하는 데 씀)
    LINE_LOST_CONFIRM_COUNT = 5   # 이만큼 연속으로 선을 못 찾아야 "진짜로 놓쳤다"고 보고 탐색 시작 (한두 프레임 깜빡임 무시)
    LINE_FOUND_CONFIRM_COUNT = 3  # 탐색 중 선을 "진짜로" 다시 찾았다고 인정하는 데 필요한 연속 프레임 수

    obs_l, obs_r = False, False
    obs_l_count, obs_r_count = 0, 0
    OBSTACLE_CONFIRM_COUNT = 5  # 한두 번 튀는 오탐으로 차가 갑자기 혼자 움직이지 않도록, 연속으로 이만큼 감지돼야 진짜로 인정
    last_lidar_time = 0.0
    LIDAR_POLL_INTERVAL_SEC = 0.1  # 매 프레임마다 읽지 않고 이 간격으로만 읽어서 부하를 줄임

    last_line_debug_time = 0.0
    LINE_DEBUG_INTERVAL_SEC = 0.3  # 디버그 사진은 사람이 보는 용도라 이 간격으로만 새로 만들어서 부하를 줄임

    last_light_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if telemetry['status'] == "EMERGENCY_STOP":
            Car.stop()
            time.sleep(0.05)
            continue

        now_lidar = time.time()
        if now_lidar - last_lidar_time >= LIDAR_POLL_INTERVAL_SEC:
            raw_l, raw_r = read_obstacles()
            obs_l_count = obs_l_count + 1 if raw_l else 0
            obs_r_count = obs_r_count + 1 if raw_r else 0
            obs_l = obs_l_count >= OBSTACLE_CONFIRM_COUNT
            obs_r = obs_r_count >= OBSTACLE_CONFIRM_COUNT
            last_lidar_time = now_lidar
        telemetry['obstacle'] = "WARNING" if (obs_l or obs_r) else "CLEAR"

        # LED 자동 모드일 때만 조도센서로 어둡다/밝다 판단해서 켜고 끔 (수동 ON/OFF일 땐 건드리지 않음)
        now_light = time.time()
        if telemetry['led_mode'] == "AUTO" and now_light - last_light_time >= LIGHT_POLL_INTERVAL_SEC:
            last_light_time = now_light
            is_dark = light_sensor.readAverage() <= LIGHT_DARK_THRESHOLD
            if is_dark != telemetry['led_on']:
                telemetry['led_on'] = is_dark
                led_set(is_dark)

        try:
            # [안전 우선] 모드와 상관없이 라이다 근접 장애물이면 무조건 회피부터.
            # 단, "MANUAL_NOLIDAR" 모드일 때는 라이다 개입 없이 순수 수동 조종만 한다
            # (벽에 가까이 붙여서 일부러 조종해야 할 때 등).
            lidar_active = telemetry['mode'] != "MANUAL_NOLIDAR"

            # LINE 모드일 때는 후진/회전 회피를 하면 트랙에서 완전히 벗어나 엉뚱한 흰 무늬(로고 등)로
            # 갈아탈 위험이 커서, 다른 모드와 달리 그냥 멈췄다가 장애물이 사라지면 다시 출발한다.
            if lidar_active and telemetry['mode'] == "LINE" and (obs_l or obs_r):
                telemetry['action_text'] = "AVOIDING (STOP)"
                Car.steering = 0
                Car.stop()
                last_speed = 0
                continue

            if lidar_active and obs_l and obs_r:
                telemetry['action_text'] = "AVOIDING (BACK)"
                Car.steering = 0; Car.backward(70); time.sleep(1.0)
                Car.steering = -1; Car.forward(75); time.sleep(0.6)
                Car.steering = 0; Car.stop()
                last_speed = 0; last_command_time = time.time()
                continue
            elif lidar_active and obs_l:
                telemetry['action_text'] = "AVOIDING (RIGHT)"
                Car.steering = 1; Car.forward(75)
                last_speed = 75; last_command_time = time.time()
                continue
            elif lidar_active and obs_r:
                telemetry['action_text'] = "AVOIDING (LEFT)"
                Car.steering = -1; Car.forward(75)
                last_speed = 75; last_command_time = time.time()
                continue

            if telemetry['mode'] == "MANUAL" or telemetry['mode'] == "MANUAL_NOLIDAR":
                Car.steering = telemetry['steering']
                if telemetry['speed'] > 0: Car.forward(telemetry['speed'])
                elif telemetry['speed'] < 0: Car.backward(abs(telemetry['speed']))
                else: Car.stop()

            elif telemetry['mode'] == "AUTO":
                height, width = frame.shape[:2]
                boxes, confs, clss = trt_yolo.detect(frame, conf_th=CONF_THRESH)
                person_box, person_conf, person_center = get_person_box(
                    boxes, confs, clss, cls_dict, last_center=last_person_center
                )

                if person_box is not None:
                    last_person_center = person_center
                    x_min, y_min, x_max, y_max = person_box
                    area_ratio = ((x_max - x_min) * (y_max - y_min)) / (width * height)
                    center_ratio = (((x_min + x_max) / 2) - width / 2) / (width / 2)
                    if smoothed_area is None:
                        smoothed_area, smoothed_center = area_ratio, center_ratio
                    else:
                        smoothed_area = EMA_ALPHA * area_ratio + (1 - EMA_ALPHA) * smoothed_area
                        smoothed_center = EMA_ALPHA * center_ratio + (1 - EMA_ALPHA) * smoothed_center
                    miss_count = 0

                    # 인식이 한두 프레임만 깜빡 잡히는 것만으로 탐색이 취소/재시작되지
                    # 않도록, 연속으로 몇 프레임은 계속 잡혀야 "진짜로 찾음"으로 인정한다.
                    # (이게 없으면 탐색이 처음(90->0)부터 계속 리셋돼서 오른쪽 끝까지
                    # 못 가고 왼쪽-중앙만 왔다갔다 하는 것처럼 보임)
                    found_count += 1
                    if is_searching and found_count >= FOUND_CONFIRM_COUNT:
                        search_waypoint_idx = 0
                        search_cycles_done = 0
                        is_searching = False
                else:
                    miss_count += 1
                    found_count = 0

                if smoothed_area is not None and miss_count <= MISS_GRACE_FRAMES:
                    steer, speed, state = decide_follow_action(smoothed_area, smoothed_center)
                else:
                    steer, speed, state = 0.0, 0, "FOLLOW (NO PERSON - SEARCHING)"
                    smoothed_area = None
                    smoothed_center = None

                    # 탐색을 처음 시작하는 순간엔 목표 지점 목록(90->0->180->90)의
                    # 맨 앞(90, 중앙)부터 다시 시작한다. search_angle은 순간 점프시키지
                    # 않고 그대로 둬서, 지금 카메라가 어디를 보고 있든 거기서부터
                    # 한 스텝씩 천천히 중앙으로 움직이게 한다.
                    if not is_searching:
                        is_searching = True
                        search_waypoint_idx = 0
                        search_cycles_done = 0
                        last_pan_time = time.time()

                    # 아주 천천히 다음 목표 각도(waypoint)를 향해 한 스텝씩 이동
                    # (수평 pan만 사용, 수직 tilt는 사용 안 함).
                    now_pan = time.time()
                    if now_pan - last_pan_time >= SEARCH_PAN_INTERVAL_SEC:
                        target = SEARCH_PAN_WAYPOINTS[search_waypoint_idx]
                        if search_angle < target:
                            search_angle = min(target, search_angle + SEARCH_PAN_STEP)
                        elif search_angle > target:
                            search_angle = max(target, search_angle - SEARCH_PAN_STEP)

                        # Car.camPan(n)은 0~180 절대각도를 그대로 받는 함수이므로
                        # search_angle(0~180)을 변환 없이 그대로 넘긴다.
                        Car.camPan(search_angle)
                        last_pan_time = now_pan

                        if search_angle == target:
                            search_waypoint_idx += 1
                            if search_waypoint_idx >= len(SEARCH_PAN_WAYPOINTS):
                                search_waypoint_idx = 0
                                search_cycles_done += 1  # 90->180->0->90 한 바퀴(1렙) 완료

                    # 정해진 렙 수를 다 반복했는데도 못 찾으면 자율주행을 끄고 정지한다.
                    if search_cycles_done >= SEARCH_MAX_CYCLES:
                        telemetry['mode'] = "MANUAL"
                        telemetry['action_text'] = "AUTO OFF (사람 못 찾음)"
                        Car.camPan(SEARCH_PAN_CENTER)
                        is_searching = False
                        search_angle = SEARCH_PAN_CENTER
                        search_waypoint_idx = 0
                        search_cycles_done = 0
                        Car.steering = 0
                        Car.stop()
                        last_speed = 0
                        last_command_time = time.time()
                        continue

                Car.steering = steer
                telemetry['steering'] = steer

                now = time.time()
                if speed != last_speed and (now - last_command_time) >= MIN_COMMAND_INTERVAL_SEC:
                    if speed == 0:
                        Car.stop()
                    else:
                        Car.forward(speed)
                    last_speed = speed
                    last_command_time = now

                telemetry['speed'] = speed
                telemetry['action_text'] = state

            elif telemetry['mode'] == "LINE":
                line_center_x, line_mask, line_roi_top = get_line_center_x(frame, last_x=last_line_x)

                width = frame.shape[1]
                frame_center_x = width / 2

                # 선을 찾았는지/놓쳤는지에 따라 재탐색 상태를 갱신 (AUTO 모드의 사람 재탐색과 같은 방식)
                if line_center_x is not None:
                    last_line_x = line_center_x  # 원래 따라가던 선 위치 기억 (놓쳤다가 복귀할 때 씀)
                    error = (line_center_x - frame_center_x) / frame_center_x

                    # 최근 30초 동안 선이 좌우 어느 쪽으로 치우쳐 있었는지 기록해둔다.
                    # 큰 장애물 등으로 놓쳤을 때, 이 추세를 보고 재탐색을 엉뚱한 반대쪽이 아니라
                    # "가던 방향"부터 훑도록 하는 데 쓴다.
                    now_t = time.time()
                    line_error_history.append((now_t, error))
                    while line_error_history and now_t - line_error_history[0][0] > LINE_HISTORY_WINDOW_SEC:
                        line_error_history.popleft()

                    line_lost_count = 0
                    line_found_count += 1
                    if line_is_searching and line_found_count >= LINE_FOUND_CONFIRM_COUNT:
                        line_is_searching = False
                        line_search_waypoint_idx = 0
                        line_search_cycles_done = 0
                        Car.camPan(SEARCH_PAN_CENTER)
                        line_search_angle = SEARCH_PAN_CENTER
                else:
                    line_found_count = 0
                    line_lost_count += 1
                    if line_lost_count >= LINE_LOST_CONFIRM_COUNT and not line_is_searching:
                        line_is_searching = True
                        line_search_waypoint_idx = 0
                        line_search_cycles_done = 0

                        # 최근 추세를 보고 재탐색 순서를 정한다 (기본은 오른쪽부터, 최근에 왼쪽으로
                        # 많이 치우쳐 있었으면 왼쪽부터 훑도록 순서를 바꿔서 반대 방향으로 헛도는 걸 방지).
                        if line_error_history:
                            avg_error = sum(e for _, e in line_error_history) / len(line_error_history)
                        else:
                            avg_error = 0.0
                        if avg_error < -LINE_TREND_THRESHOLD:
                            line_search_waypoints = [SEARCH_PAN_CENTER, 180, 0, SEARCH_PAN_CENTER]
                        elif avg_error > LINE_TREND_THRESHOLD:
                            line_search_waypoints = [SEARCH_PAN_CENTER, 0, 180, SEARCH_PAN_CENTER]
                        else:
                            line_search_waypoints = SEARCH_PAN_WAYPOINTS

                if line_is_searching:
                    steer, speed = 0.0, 0
                    state = "LINE (선 재탐색중...)"
                    now_pan = time.time()
                    if now_pan - last_pan_time >= SEARCH_PAN_INTERVAL_SEC:
                        target = line_search_waypoints[line_search_waypoint_idx]
                        if line_search_angle < target:
                            line_search_angle = min(target, line_search_angle + SEARCH_PAN_STEP)
                        elif line_search_angle > target:
                            line_search_angle = max(target, line_search_angle - SEARCH_PAN_STEP)
                        Car.camPan(line_search_angle)
                        last_pan_time = now_pan
                        if line_search_angle == target:
                            line_search_waypoint_idx += 1
                            if line_search_waypoint_idx >= len(line_search_waypoints):
                                line_search_waypoint_idx = 0
                                line_search_cycles_done += 1
                    if line_search_cycles_done >= SEARCH_MAX_CYCLES:
                        # 다 훑어봐도 못 찾으면 포기하고 카메라 정중앙으로 되돌린 채 대기
                        # (다시 선이 보이면 위의 line_center_x is not None 쪽에서 자동으로 재개됨)
                        state = "LINE (선 못 찾음-대기)"
                        Car.camPan(SEARCH_PAN_CENTER)
                        line_is_searching = False
                        line_search_angle = SEARCH_PAN_CENTER
                        line_search_waypoint_idx = 0
                        line_search_cycles_done = 0
                elif line_center_x is None:
                    steer, speed, state = 0.0, 0, "LINE (선 없음-정지)"
                else:
                    steer = max(-LINE_MAX_STEER, min(LINE_MAX_STEER, error * LINE_STEER_GAIN))
                    speed = LINE_FORWARD_SPEED
                    state = f"LINE (steer={steer:.2f})"

                Car.steering = steer
                telemetry['steering'] = steer

                # [중요] setSpeed()는 "이미 주행 중"일 때만 반영되는 함수라 정지 상태에서
                # 부르면 무효였던 버그가 예전에 있었음 -> forward(speed)로 직접 전달.
                if speed == 0:
                    Car.stop()
                else:
                    Car.forward(speed)
                last_speed = speed

                telemetry['speed'] = speed
                telemetry['action_text'] = state

                # 디버그용 화면 만들기: ROI 영역에 인식된 선(마스크)을 초록색으로 덧칠,
                # 중앙 기준선(파란색)과 인식된 선 중앙(빨간 점)도 표시.
                # [2026-07-15] 사람이 새로고침 버튼 눌렀을 때 보는 용도라, 매 프레임 만들 필요 없이
                # LINE_DEBUG_INTERVAL_SEC 간격으로만 만들어서 조향 루프 속도를 아낀다.
                now = time.time()
                if now - last_line_debug_time >= LINE_DEBUG_INTERVAL_SEC:
                    last_line_debug_time = now
                    debug_frame = frame.copy()
                    roi_view = debug_frame[line_roi_top:, :]
                    roi_view[line_mask > 0] = [0, 255, 0]
                    cv2.line(debug_frame, (debug_frame.shape[1] // 2, line_roi_top),
                             (debug_frame.shape[1] // 2, debug_frame.shape[0]), (255, 0, 0), 2)
                    if line_center_x is not None:
                        cv2.circle(debug_frame, (int(line_center_x), debug_frame.shape[0] - 20), 8, (0, 0, 255), -1)
                    cv2.putText(debug_frame, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    ok, encoded = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        with line_debug_lock:
                            line_debug_jpeg = encoded.tobytes()
        except Exception as e:
            print(f"[모터 에러]: {e}")


if __name__ == '__main__':
    flask_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    hardware_engine()
