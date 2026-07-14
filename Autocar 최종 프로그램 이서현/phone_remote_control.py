# 폰 웹뷰 하나로 전부 통합:
#   - MANUAL 모드: 폰 화면 방향키로 직접 조종
#   - AUTO 모드: 카메라+YOLO로 사람을 추적해서 따라감
#   - 두 모드 공통: 라이다가 좌우 근접 장애물(700mm 이내)을 감지하면
#                   YOLO/폰 조작보다 항상 라이다 회피가 우선 (안전이 최우선)
#
# 화면 스트리밍(웹으로 카메라 보기)은 뺐음 - 폰 리모컨 대시보드만 사용.
#
# [중요] pycuda(YOLO의 GPU 연산)는 처음 초기화된 스레드에서만 동작하므로
# 카메라/YOLO 처리(hardware_engine)는 반드시 메인 스레드에서 실행해야 함.
# 그래서 Flask 서버 쪽을 별도 스레드로 돌린다 (기존 app.py와 순서 반대).
#
# 실행 위치 주의: 반드시 ~/tensorrt_demos 폴더 안에서 실행해야 함
# (YOLO 모델/플러그인 파일들이 그 폴더 기준 상대경로로 되어있음).
import sys
import threading
import time

sys.path.insert(1, '.')

import cv2
import pycuda.autoinit  # noqa
from flask import Flask, render_template, request, jsonify, redirect
from flask_socketio import SocketIO

from pop import Pilot, Util
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_cls_dict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'autocar_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

Car = Pilot.AutoCar()

telemetry = {
    "steering": 0, "speed": 0,
    "mode": "MANUAL", "status": "READY",
    "obstacle": "CLEAR", "action_text": "STOP"
}

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



@app.route('/')
def index():
    return redirect('/remote')


@app.route('/remote')
def remote_view():
    return render_template('remote.html')


@app.route('/api/status')
def api_status():
    return jsonify(telemetry)


@app.route('/api/control')
def api_control():
    global telemetry
    if telemetry['mode'] == "MANUAL":
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
    return jsonify(telemetry)


@app.route('/api/kill')
def api_kill():
    global telemetry
    telemetry['status'] = "EMERGENCY_STOP"
    telemetry['mode'] = "MANUAL"; telemetry['speed'] = 0; telemetry['steering'] = 0
    telemetry['action_text'] = "EMERGENCY STOP"
    print("[긴급 제동] 스마트폰 킬 스위치 작동! 차량 즉시 정지")
    return jsonify(telemetry)


def connect_lidar():
    """라이다 연결 (반드시 메인 스레드에서, 카메라/YOLO와 동시에 말고 순서대로 호출할 것).
    _rplidar.so가 스레드 안전하지 않아서 별도 스레드로 돌리면 프로세스가 통째로
    죽는 문제가 있었음 -> 스레드 없이 메인 루프 안에서 직접 폴링하는 방식으로 변경."""
    try:
        from pop import Lidar
    except ImportError:
        from pop import LiDAR as Lidar

    try:
        lidar = Lidar.Rplidar()
        lidar.connect()
        lidar.startMotor()
        print("[라이다] 연결 성공, 안전 감지 시작")
        return lidar
    except Exception as e:
        print(f"[라이다] 연결 실패 - 안전 회피 없이 동작함: {e}")
        return None


def read_obstacles(lidar):
    """라이다에서 좌/우 근접 장애물 여부만 읽어온다. 메인 루프에서 직접 호출."""
    if lidar is None:
        return False, False
    try:
        vectors = lidar.getVectors()
        left_points = 0
        right_points = 0
        if vectors is not None:
            for v in vectors:
                if len(v) >= 2 and 50 < v[1] < LIDAR_RANGE_MM:
                    if 360 - LIDAR_FRONT_DEG <= v[0] <= 360:
                        left_points += 1
                    elif 0 <= v[0] <= LIDAR_FRONT_DEG:
                        right_points += 1
        return left_points > LIDAR_MIN_POINTS, right_points > LIDAR_MIN_POINTS
    except Exception:
        return False, False


def get_person_box(boxes, confs, clss, cls_dict):
    biggest_area = 0
    biggest_box = None
    biggest_conf = 0
    for box, conf, cls_id in zip(boxes, confs, clss):
        if cls_dict.get(int(cls_id), "") != "person":
            continue
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        if area > biggest_area:
            biggest_area = area
            biggest_box = box
            biggest_conf = conf
    return biggest_box, biggest_conf


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
    global telemetry

    trt_yolo = TrtYOLO('yolov3-tiny-416', category_num=80)
    cls_dict = get_cls_dict(80)

    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("[시스템] 카메라+YOLO 준비 완료. 전원 안정화 대기 중...")

    # 카메라+GPU가 이미 전력을 쓰고 있는 상태에서 라이다 모터가 갑자기 추가로
    # 전류를 끌어쓰면 전원이 불안정해져서 프로세스가 죽거나, connect()가 아예
    # 끝없이 멈춰버리는 현상이 있었음. 잠깐 대기해서 전원을 안정시킨 뒤 연결한다.
    time.sleep(3)

    # [중요] connect()가 멈춰버릴 수 있어서, 제한시간(LIDAR_CONNECT_TIMEOUT_SEC) 안에
    # 안 끝나면 포기하고 라이다 없이 진행한다 (그래야 폰 조종/YOLO라도 정상 작동함).
    print("[시스템] 라이다 연결 시작... (최대 10초 대기)")
    lidar_result = {"lidar": None}

    def _connect_lidar_bg():
        lidar_result["lidar"] = connect_lidar()

    lidar_connect_thread = threading.Thread(target=_connect_lidar_bg, daemon=True)
    lidar_connect_thread.start()
    lidar_connect_thread.join(timeout=10)

    if lidar_connect_thread.is_alive():
        print("[라이다] 10초 내에 연결이 안 끝남 - 라이다 없이 진행 (안전 회피 비활성화)")
        lidar = None
    else:
        lidar = lidar_result["lidar"]

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

    obs_l, obs_r = False, False
    last_lidar_time = 0.0
    LIDAR_POLL_INTERVAL_SEC = 0.1  # 매 프레임마다 읽지 않고 이 간격으로만 읽어서 부하를 줄임

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
            obs_l, obs_r = read_obstacles(lidar)
            last_lidar_time = now_lidar
        telemetry['obstacle'] = "WARNING" if (obs_l or obs_r) else "CLEAR"

        try:
            # [안전 우선] 모드와 상관없이 라이다 근접 장애물이면 무조건 회피부터.
            if obs_l and obs_r:
                telemetry['action_text'] = "AVOIDING (BACK)"
                Car.steering = 0; Car.backward(70); time.sleep(1.0)
                Car.steering = -1; Car.forward(75); time.sleep(0.6)
                Car.steering = 0; Car.stop()
                last_speed = 0; last_command_time = time.time()
                continue
            elif obs_l:
                telemetry['action_text'] = "AVOIDING (RIGHT)"
                Car.steering = 1; Car.forward(75)
                last_speed = 75; last_command_time = time.time()
                continue
            elif obs_r:
                telemetry['action_text'] = "AVOIDING (LEFT)"
                Car.steering = -1; Car.forward(75)
                last_speed = 75; last_command_time = time.time()
                continue

            if telemetry['mode'] == "MANUAL":
                Car.steering = telemetry['steering']
                if telemetry['speed'] > 0: Car.forward(telemetry['speed'])
                elif telemetry['speed'] < 0: Car.backward(abs(telemetry['speed']))
                else: Car.stop()

            elif telemetry['mode'] == "AUTO":
                height, width = frame.shape[:2]
                boxes, confs, clss = trt_yolo.detect(frame, conf_th=CONF_THRESH)
                person_box, person_conf = get_person_box(boxes, confs, clss, cls_dict)

                if person_box is not None:
                    x_min, y_min, x_max, y_max = person_box
                    area_ratio = ((x_max - x_min) * (y_max - y_min)) / (width * height)
                    center_ratio = (((x_min + x_max) / 2) - width / 2) / (width / 2)
                    if smoothed_area is None:
                        smoothed_area, smoothed_center = area_ratio, center_ratio
                    else:
                        smoothed_area = EMA_ALPHA * area_ratio + (1 - EMA_ALPHA) * smoothed_area
                        smoothed_center = EMA_ALPHA * center_ratio + (1 - EMA_ALPHA) * smoothed_center
                    miss_count = 0

                    # 사람을 다시 찾았으면 그 자리에서 카메라를 멈추고(되돌리지 않고)
                    # 다시 정상적으로 사람을 따라간다.
                    if is_searching:
                        search_waypoint_idx = 0
                        search_cycles_done = 0
                        is_searching = False
                else:
                    miss_count += 1

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

                        # Car.camPan(n)은 "정중앙(90도) 기준 오프셋"을 받으므로
                        # 실제 각도(search_angle)를 오프셋으로 변환해서 넘긴다.
                        Car.camPan(SEARCH_PAN_CENTER - search_angle)
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
                        Car.camPan(0)
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
        except Exception as e:
            print(f"[모터 에러]: {e}")


if __name__ == '__main__':
    flask_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    hardware_engine()
