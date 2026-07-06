# -*- coding: utf-8 -*-
"""
track_dashboard.py
──────────────────────────────────────────────────────────────
[7/3 과제] 오토카 운동장 트랙 돌기 - 센서 수집 + 웹 대시보드

■ 이 프로그램이 하는 일
  1) 오토카를 조종합니다. (컴퓨터 키보드 방향키 조종 / 자동주행 버튼)
  2) "운전 시작" 버튼을 누르면 조종이 활성화되면서 동시에 CDS(조도) 센서
     값 + 위치(오도메트리) 데이터 수집이 시작됩니다.
  3) "운전 정지" 버튼을 누르면 오토카가 멈추고, 수집도 즉시 종료되면서
     CSV 파일이 "csv 파일 모음" 폴더에 바로 저장됩니다.
  4) 웹 브라우저(대시보드)에서 실시간으로 주행 경로와 센서 값 그래프를
     보면서 확인할 수 있습니다.

■ 조작 방법
  - 브라우저 화면에서 "운전 시작"을 누른 뒤, 키보드의 방향키를 누릅니다.
      ↑ : 전진        ↓ : 후진
      ← : 왼쪽으로 조향   → : 오른쪽으로 조향
  - 방향키에서 손을 떼면 그 즉시 오토카가 멈춥니다. (계속 누르고 있어야
    움직입니다 - 오래 눌러도 안전하도록 워치독이 같이 동작합니다.)
  - "운전 정지"를 누르면 오토카가 멈추고 그 순간까지 모은 데이터가
    CSV 파일로 저장됩니다.

■ 실행 방법 (오토카 터미널 = soda@192.168.0.57 에서 실행)
    1) 이 폴더를 오토카로 옮깁니다. (WSL 터미널에서 실행)
         scp -r "오토카 운동장 돌기" soda@192.168.0.57:~/
    2) 오토카에 접속합니다. (WSL 터미널에서 실행)
         ssh soda@192.168.0.57
    3) 오토카 터미널에서 이 파일을 실행합니다.
         cd ~/오토카\ 운동장\ 돌기
         python3 track_dashboard.py
    4) 아무 컴퓨터/휴대폰 브라우저에서 아래 주소로 접속합니다.
         http://192.168.0.57:5000

■ 파일 구성
    track_dashboard.py  : 이 파일. Flask 웹 서버 + 대시보드 화면.
    track_odometry.py   : 위치(오도메트리) 계산 전용 모듈.
    csv 파일 모음/         : 수집한 CSV 파일이 자동으로 저장되는 폴더.
──────────────────────────────────────────────────────────────
"""

import sys
import os
import csv
import time
import threading

# pop 모듈 경로 추가 (오토카 홈 디렉터리인 /home/soda 를 파이썬 경로에 넣어줌)
sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.getcwd())

from flask import Flask, jsonify, request

from track_odometry import TrackOdometry

# ════════════════════════════════════════════════════════════
# 0. 조정 가능한 숫자값 모음 (여기 값들을 바꾸면 동작이 어떻게 바뀌는지 설명)
# ════════════════════════════════════════════════════════════

# CDS(조도) 센서가 연결된 SPI ADC 채널 번호입니다.
# - pop.Util 의 Cds 클래스는 SPI ADC 여러 채널(0~7) 중 하나에서 값을
#   읽어오는데, 실제로 CDS 센서를 몇 번 채널에 꽂았는지에 따라 값이
#   달라집니다. 대시보드를 켰을 때 CDS 값이 항상 0 이거나 이상하게
#   나오면 이 번호를 0~7 사이에서 바꿔가며 실제로 빛을 손으로 가려보고
#   값이 바뀌는 채널을 찾아주세요.
CDS_ADC_CHANNEL = 0

# 센서/위치를 얼마나 자주 측정할지 (초 단위)
# - 값이 작을수록(예: 0.05초): 더 촘촘하게 기록되어 그래프가 부드럽지만
#   CSV 파일 크기가 커지고 데이터 양이 많아집니다.
# - 값이 클수록(예: 0.5초): 데이터는 적지만 빠르게 지나가는 구간의
#   센서값 변화를 놓칠 수 있습니다.
SAMPLE_INTERVAL_SEC = 0.1  # 10Hz(1초에 10번)

# 키보드(방향키) 조종 시, 이 시간(초) 동안 /control 신호가 안 오면
# 자동으로 정지합니다. (통신이 끊겼는데 계속 달리는 사고를 막기 위한
# "워치독". 방향키에서 손을 떼면 브라우저가 즉시 정지 신호를 보내지만,
# 혹시 그 신호마저 도착하지 못하는 상황을 대비한 안전장치입니다.)
# - 값이 작을수록: 통신이 살짝만 끊겨도 바로 멈춰서 더 안전하지만,
#   네트워크가 느리면 자꾸 멈출 수 있습니다.
# - 값이 클수록: 덜 예민하게 멈추지만, 문제가 생겼을 때 더 오래
#   멈추지 않고 달릴 수 있어 위험합니다.
WATCHDOG_TIMEOUT_SEC = 0.3

# CSV 파일을 저장할 폴더 이름 (이 스크립트와 같은 위치에 자동 생성됨)
# 내 컴퓨터(WSL)의 같은 프로젝트 폴더 안에도 "csv 파일 모음" 폴더가
# 미리 만들어져 있어서, scp로 이 폴더만 그대로 가져오면 경로가 딱
# 맞아떨어집니다.
#   scp -r soda@<오토카IP>:"~/오토카 운동장 돌기/csv 파일 모음" \
#       "/home/eohyun_ee/autocar/오토카 운동장 돌기/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv 파일 모음")


# ════════════════════════════════════════════════════════════
# 1. 오토카 하드웨어 초기화
# ════════════════════════════════════════════════════════════
# pop/Pilot.py 내부에서 numpy(np)를 미리 import 하지 않고 사용하는 부분이
# 있어서, 터미널에서 그냥 실행하면 오류가 날 수 있습니다.
# 아래처럼 builtins(파이썬이 항상 기본으로 갖고 있는 공간)에 np를 잠깐
# 넣어줬다가 사용한 뒤 다시 지우는 방식으로 문제를 피합니다.
import builtins as _builtins
import numpy as _np
_builtins.np = _np
try:
    import torchvision as _tv
    _builtins.torchvision = _tv
except ImportError:
    pass

HAS_CAR = False
car = None
try:
    from pop.Pilot import AutoCar
    car = AutoCar()
    HAS_CAR = True
    print("[오토카] 하드웨어 연결 성공")
except Exception as _e:
    print("[오토카] 시뮬레이션 모드로 동작합니다 (하드웨어 연결 실패: %s)" % _e)

HAS_CDS = False
cds_sensor = None
try:
    from pop.Util import Cds
    cds_sensor = Cds(channel=CDS_ADC_CHANNEL)
    HAS_CDS = True
    print("[CDS 센서] 연결 성공 (채널 %d)" % CDS_ADC_CHANNEL)
except Exception as _e:
    print("[CDS 센서] 시뮬레이션 모드로 동작합니다 (센서 연결 실패: %s)" % _e)
finally:
    for _attr in ('np', 'torchvision'):
        if hasattr(_builtins, _attr):
            delattr(_builtins, _attr)


def read_cds_value():
    """
    CDS(조도) 센서 값을 하나 읽어옵니다.
    - 실제 센서가 있으면 진짜 밝기 값을 반환합니다. (값이 클수록 밝음)
    - 센서가 없는 시뮬레이션 모드에서는, 그래도 대시보드 테스트를 할 수
      있도록 시간에 따라 부드럽게 오르내리는 가짜 값을 만들어 줍니다.
    """
    if HAS_CDS:
        return cds_sensor.readAverage()
    import math
    return int(500 + 400 * math.sin(time.time() * 0.5))


# ════════════════════════════════════════════════════════════
# 2. 오토카를 조종하는 두 가지 방법을 위한 공용 상태값
#    (키보드 방향키 조종 / 자동주행 버튼 중 어느 쪽을 쓰든
#     아래 값들만 최신 상태로 맞춰두면 됨)
# ════════════════════════════════════════════════════════════
_state_lock = threading.Lock()
_state = {
    "speed": 0,       # 0~99 사이 값 (현재 목표 속력의 크기)
    "direction": 0,   # 1=전진, -1=후진, 0=정지
    "steering": 0.0,  # -1~1 사이 값 (현재 목표 조향)
}

_last_ping = time.time()   # 워치독용 - 마지막으로 조종 신호를 받은 시각
_odometry = TrackOdometry()

# 주행 경로를 그리기 위해 화면에 보여줄 점들을 순서대로 저장하는 리스트
# 각 원소: {"t": 경과초, "x": x좌표(m), "y": y좌표(m), "cds": 조도값}
_trail = []
_trail_lock = threading.Lock()

# 수집(레코딩) 상태 - "운전 시작/정지" 버튼과 자동주행 시작/정지에
# 같이 연결되어 있습니다. (둘 중 어떤 방식으로 운전을 시작하든 수집도
# 같이 시작되고, 정지하면 같이 종료됩니다.)
_recording = False
_record_lock = threading.Lock()
_record_file = None
_record_writer = None
_record_path = None
_record_start_time = None

# 자동주행(이동 프로그램) 상태
_auto_running = False
_auto_thread = None


def _apply_drive(direction, speed, steering):
    """공용 상태값을 실제 오토카 명령으로 내려보내는 함수."""
    with _state_lock:
        _state["direction"] = direction
        _state["speed"] = speed
        _state["steering"] = steering

    if not car:
        return

    car.steering = steering
    if direction > 0:
        car.setSpeed(speed)
        car.forward()
    elif direction < 0:
        car.setSpeed(speed)
        car.backward()
    else:
        car.stop()


def _stop_drive():
    _apply_drive(0, 0, 0.0)


# ════════════════════════════════════════════════════════════
# 3. 배경 스레드 1 - 조종 워치독
#    (조종 신호가 WATCHDOG_TIMEOUT_SEC 초 이상 안 오면 자동 정지)
# ════════════════════════════════════════════════════════════
def _watchdog_loop():
    global _last_ping
    while True:
        time.sleep(0.05)
        if car and (time.time() - _last_ping > WATCHDOG_TIMEOUT_SEC):
            _stop_drive()


# ════════════════════════════════════════════════════════════
# 4. 배경 스레드 2 - 센서/위치 샘플링
#    SAMPLE_INTERVAL_SEC 마다 한 번씩:
#      - 위치(오도메트리) 갱신
#      - CDS 값 읽기
#      - 화면에 보여줄 trail 리스트에 추가
#      - 레코딩 중이면 CSV 파일에 한 줄 기록
# ════════════════════════════════════════════════════════════
def _sampling_loop():
    global _record_writer, _record_file
    last_time = time.time()

    while True:
        time.sleep(SAMPLE_INTERVAL_SEC)

        now = time.time()
        dt = now - last_time
        last_time = now

        with _state_lock:
            speed = _state["speed"] * _state["direction"]  # 부호 있는 속도
            steering = _state["steering"]

        # 위치 갱신 (track_odometry.py 의 자전거 모델 계산)
        _odometry.update(speed=speed, steering=steering, dt=dt)

        cds_value = read_cds_value()

        point = {
            "t": round(now - (_record_start_time or now), 2),
            "x": round(_odometry.x, 3),
            "y": round(_odometry.y, 3),
            "cds": cds_value,
        }

        with _trail_lock:
            _trail.append(point)
            # 대시보드가 너무 느려지지 않도록 최근 5000개까지만 화면용으로 보관
            if len(_trail) > 5000:
                del _trail[: len(_trail) - 5000]

        with _record_lock:
            if _recording and _record_writer is not None:
                _record_writer.writerow([
                    round(now, 3),
                    point["t"],
                    _odometry.x, _odometry.y, _odometry.heading_deg,
                    speed, steering, cds_value,
                ])
                _record_file.flush()  # 중간에 프로그램이 꺼져도 데이터가 남도록 즉시 저장


# ════════════════════════════════════════════════════════════
# 5. 배경 스레드 3 - 간단한 자동주행(이동) 프로그램
#    운동장 트랙을 한 바퀴 돌 때 쓸 수 있는 아주 단순한 방식으로,
#    "일정한 속도로 전진하면서 한쪽으로 살짝 조향각을 준 상태를 유지"
#    합니다. 트랙이 원/타원에 가깝다면 이 방식만으로 계속 원을 그리며
#    돌 수 있습니다. (키보드 방향키로 직접 몰아도 됩니다 - 과제에서도
#    수동 조작을 허용하고 있습니다.)
# ════════════════════════════════════════════════════════════
def _auto_drive_loop(speed, steering):
    global _auto_running
    while _auto_running:
        _apply_drive(direction=1, speed=speed, steering=steering)
        time.sleep(0.1)
    _stop_drive()


# ════════════════════════════════════════════════════════════
# 6. CSV 저장 관련 함수
# ════════════════════════════════════════════════════════════
CSV_HEADER = [
    "timestamp_unix",   # 실제 시각 (1970년부터 흐른 초)
    "elapsed_sec",      # 레코딩을 시작한 뒤로부터 흐른 시간(초)
    "x_m", "y_m",        # 오도메트리로 추정한 위치 (미터)
    "heading_deg",       # 오도메트리로 추정한 진행 방향 (도)
    "speed",             # 그 순간의 부호 있는 속력 (-99~99)
    "steering",          # 그 순간의 조향 값 (-1~1)
    "cds_value",         # CDS 조도 센서 값
]


def _start_recording():
    """
    CSV 수집을 시작합니다. 이미 수집 중이면 아무것도 하지 않고 현재
    저장 중인 파일 이름을 그대로 반환합니다. (운전 시작 버튼과 자동주행
    시작 버튼이 둘 다 이 함수를 부를 수 있어서, 중복 시작을 막습니다.)
    """
    global _recording, _record_file, _record_writer, _record_path, _record_start_time

    with _record_lock:
        if _recording:
            return os.path.basename(_record_path)

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = "track_data_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
    path = os.path.join(DATA_DIR, filename)

    f = open(path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)

    with _record_lock:
        _record_file = f
        _record_writer = writer
        _record_path = path
        _record_start_time = time.time()
        _recording = True

    with _trail_lock:
        _trail.clear()
    _odometry.reset()

    return filename


def _stop_recording():
    """CSV 수집을 종료하고 파일을 바로 저장(닫기)합니다."""
    global _recording, _record_file, _record_writer

    with _record_lock:
        _recording = False
        if _record_file is not None:
            _record_file.close()
        _record_file = None
        _record_writer = None


# ════════════════════════════════════════════════════════════
# 7. Flask 웹 서버
# ════════════════════════════════════════════════════════════
app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오토카 운동장 트랙 대시보드</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: #161b22; padding: 10px 20px; display: flex;
    align-items: center; gap: 12px; border-bottom: 1px solid #30363d;
  }
  header h1 { font-size: 1.15em; color: #58a6ff; }
  .badge {
    font-size: 0.75em; padding: 2px 8px; border-radius: 12px;
    background: #21262d; border: 1px solid #30363d;
  }
  .grid {
    display: grid; grid-template-columns: 280px 1fr 300px;
    gap: 12px; padding: 12px; flex: 1;
  }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 12px; display: flex; flex-direction: column; gap: 10px;
  }
  .card h2 {
    font-size: 0.8em; color: #8b949e; text-transform: uppercase;
    letter-spacing: 1px; border-bottom: 1px solid #21262d; padding-bottom: 6px;
  }
  .sl-row { display: flex; flex-direction: column; gap: 4px; }
  .sl-row label { display: flex; justify-content: space-between; font-size: 0.78em; color: #8b949e; }
  input[type=range] { width: 100%; accent-color: #58a6ff; cursor: pointer; }
  .btn {
    border: none; border-radius: 6px; padding: 12px 12px; font-size: 0.9em;
    font-weight: 700; cursor: pointer; width: 100%;
  }
  .btn-red   { background: #da3633; color: #fff; }
  .btn-green { background: #238636; color: #fff; }
  .btn-blue  { background: #1f6feb; color: #fff; }
  .btn-dark  { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  #pathCanvas, #cdsCanvas { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; width: 100%; }
  .stat { display: flex; justify-content: space-between; font-size: 0.8em; color: #8b949e; }
  .stat b { color: #c9d1d9; }
  /* 방향키 안내판 - 지금 눌려있는 키는 파란색으로 강조됨 */
  .keypad {
    display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(2, 1fr);
    gap: 6px; width: 160px; margin: 4px auto;
  }
  .key {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3em; color: #484f58; height: 44px;
  }
  .key.active { background: #1f6feb; border-color: #58a6ff; color: #fff; }
  #up { grid-column: 2; grid-row: 1; }
  #left { grid-column: 1; grid-row: 2; }
  #down { grid-column: 2; grid-row: 2; }
  #right { grid-column: 3; grid-row: 2; }
</style>
</head>
<body>

<header>
  <h1>&#127939; 오토카 운동장 트랙 대시보드</h1>
  <span class="badge" id="hwBadge">HW: 확인중...</span>
  <span class="badge" id="cdsBadge">CDS: 확인중...</span>
</header>

<div class="grid">

  <!-- 왼쪽: 조종 + 수집 -->
  <div class="card">
    <h2>&#8593; 키보드 방향키 조종</h2>
    <div class="keypad">
      <div class="key" id="up">&#8593;</div>
      <div class="key" id="left">&#8592;</div>
      <div class="key" id="down">&#8595;</div>
      <div class="key" id="right">&#8594;</div>
    </div>
    <div class="sl-row">
      <label>키보드 조종 속력 <span id="kbSpeedVal">50</span></label>
      <input type="range" id="kbSpeed" min="0" max="99" value="50">
    </div>
    <div id="driveStatus" style="font-size:0.82em;">대기중 (운전 시작을 눌러주세요)</div>
    <button class="btn btn-green" id="driveBtn" onclick="toggleDrive()">&#9654; 운전 시작</button>
    <button class="btn btn-red" onclick="emergencyStop()">긴급 정지</button>

    <h2 style="margin-top:8px;">&#128260; 자동 주행(이동 프로그램)</h2>
    <div class="sl-row">
      <label>속력 <span id="autoSpeedVal">50</span></label>
      <input type="range" id="autoSpeed" min="0" max="99" value="50">
    </div>
    <div class="sl-row">
      <label>조향(왼쪽 -1 ~ 오른쪽 1) <span id="autoSteerVal">0.30</span></label>
      <input type="range" id="autoSteer" min="-100" max="100" value="30">
    </div>
    <button class="btn btn-blue" id="autoBtn" onclick="toggleAuto()">자동 주행 시작</button>
  </div>

  <!-- 가운데: 주행 경로 시각화 -->
  <div class="card">
    <h2>&#128506; 주행 경로 (위치별 CDS 값)</h2>
    <canvas id="pathCanvas" width="600" height="420"></canvas>
    <div class="stat"><span>현재 위치</span><b id="posInfo">x=0.00m, y=0.00m</b></div>
    <div class="stat"><span>현재 CDS 값</span><b id="cdsInfo">-</b></div>
    <h2 style="margin-top:4px;">CDS 값 변화 그래프</h2>
    <canvas id="cdsCanvas" width="600" height="140"></canvas>
  </div>

  <!-- 오른쪽: 수집 현황 -->
  <div class="card">
    <h2>&#128190; 데이터 수집 현황</h2>
    <p style="font-size:0.75em; color:#484f58; line-height:1.6;">
      "운전 시작"을 누르면 자동으로 CSV 수집도 같이 시작되고,
      "운전 정지"를 누르면 그 즉시 <b>csv 파일 모음</b> 폴더 안에
      track_data_YYYYMMDD_HHMMSS.csv 이름으로 저장됩니다.
    </p>
    <div class="stat"><span>기록된 점 개수</span><b id="ptCount">0</b></div>
    <div class="stat"><span>마지막 저장 파일</span><b id="fileInfo">-</b></div>
  </div>

</div>

<script>
// ── 공용 fetch 함수 ──
async function api(path, body) {
  try {
    const r = await fetch(path, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})
    });
    return await r.json();
  } catch (e) { return null; }
}

// ── 초기 상태 조회 ──
fetch('/status').then(r => r.json()).then(d => {
  const hw = document.getElementById('hwBadge');
  hw.textContent = 'HW: ' + (d.has_car ? '연결됨' : '시뮬레이션');
  hw.style.color = d.has_car ? '#2ecc71' : '#e3b341';
  const cds = document.getElementById('cdsBadge');
  cds.textContent = 'CDS: ' + (d.has_cds ? '연결됨' : '시뮬레이션');
  cds.style.color = d.has_cds ? '#2ecc71' : '#e3b341';
}).catch(function(){});

// ══════════════════════════════════════════════════════
// 키보드 방향키 조종 - "누르자마자 바로" 반응하도록 만든 부분
//   - "운전 시작"을 누른 뒤부터만 방향키가 실제로 동작합니다.
//   - 핵심 아이디어: setInterval로 "다음 순번이 올 때까지 기다렸다가"
//     보내는 게 아니라, keydown/keyup 이벤트가 "발생한 바로 그 순간"
//     sendKeyboardControl()을 즉시 호출합니다. 그래서 키를 누르는
//     순간과 서버로 신호가 나가는 순간 사이에 기다리는 시간이 없습니다.
//   - setInterval은 "혹시 신호가 중간에 하나 유실돼도 금방 다시
//     채워주는 보험(하트비트)" 용도로만 남겨뒀습니다.
// ══════════════════════════════════════════════════════
let driveActive = false;             // "운전 시작" 눌렀는지 여부
const pressedKeys = { up:false, down:false, left:false, right:false };

// 키를 누르고 있는 동안 "혹시 몰라서" 보험으로 다시 보내는 주기(ms).
// keydown/keyup 순간에는 이미 즉시 전송되므로, 이 값은 반응속도에
// 큰 영향은 없고 "네트워크 패킷이 한 번 유실됐을 때 얼마나 빨리
// 복구되는지"만 결정합니다. 값을 더 줄이면 복구는 빨라지지만 서버에
// 보내는 요청 수가 늘어납니다.
const KEY_HEARTBEAT_MS = 50;

function keyElemId(key) {
  return { ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right' }[key];
}

// ── "요청 밀림(pile-up)" 방지 ──────────────────────────
// 문제 상황: 오토카에 명령을 실제로 내리는 부분(CAN 통신)은 아주
// 조금이라도 시간이 걸릴 수 있습니다. 만약 그 처리가 끝나기 전에
// keydown/keyup/하트비트가 계속 새 요청을 또 보내버리면, 오래된
// 요청들이 줄줄이 밀려서 쌓이고, 화면(오토카)에는 "몇 번 전에 눌렀던
// 오래된 키 상태"가 뒤늦게 반영됩니다 - 이게 반응이 느려 보이는
// 가장 큰 원인입니다.
//
// 해결 방법: "서버로 보낸 요청이 아직 안 끝났으면 새로 보내지 않고,
// 대신 '보내야 할 최신 상태가 있다'는 표시만 해둔다. 요청이 끝나는
// 순간, 그 표시가 남아있으면 그때의 가장 최신 키 상태로 바로 한 번
// 더 보낸다." 이렇게 하면 서버에는 항상 최대 1개의 요청만 진행 중이라
// 밀리는 일이 없고, 그러면서도 손을 뗀 최신 상태가 가능한 한 빨리
// 반영됩니다.
let _controlInFlight = false;   // 지금 서버로 보낸 요청이 처리되길 기다리는 중인지
let _controlDirty = false;      // 기다리는 동안 새로운 키 입력이 더 있었는지

function sendKeyboardControl() {
  if (!driveActive) return;
  if (_controlInFlight) {
    _controlDirty = true;   // 지금은 못 보내니, 끝나면 최신 상태로 다시 보내라고 표시만 해둠
    return;
  }
  _flushControlNow();
}

async function _flushControlNow() {
  _controlInFlight = true;
  _controlDirty = false;

  const x = (pressedKeys.right ? 1 : 0) - (pressedKeys.left ? 1 : 0);
  const y = (pressedKeys.up ? 1 : 0) - (pressedKeys.down ? 1 : 0);
  const speedRatio = parseInt(document.getElementById('kbSpeed').value) / 99;

  try {
    await fetch('/control', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ x: x, y: y * speedRatio })
    });
  } catch (e) { /* 네트워크 순간 오류는 무시 - 다음 하트비트가 다시 시도함 */ }

  _controlInFlight = false;
  if (_controlDirty) {
    _flushControlNow();   // 기다리는 동안 키가 더 눌렸다면, 그 최신 상태를 바로 이어서 전송
  }
}

window.addEventListener('keydown', function(e) {
  if (!driveActive) return;
  const id = keyElemId(e.key);
  if (!id) return;
  e.preventDefault();
  if (pressedKeys[id]) return;  // 키보드 자동반복(auto-repeat)으로 같은
                                 // keydown이 계속 들어와도 중복 전송하지
                                 // 않도록 막음 (이미 눌린 상태면 무시)
  pressedKeys[id] = true;
  document.getElementById(id).classList.add('active');
  sendKeyboardControl();  // ★ 누르는 순간 바로 전송 (기다리지 않음)
});

window.addEventListener('keyup', function(e) {
  const id = keyElemId(e.key);
  if (!id) return;
  e.preventDefault();
  pressedKeys[id] = false;
  document.getElementById(id).classList.remove('active');
  sendKeyboardControl();  // ★ 떼는 순간 바로 전송 (기다리지 않음)
});

setInterval(sendKeyboardControl, KEY_HEARTBEAT_MS);

// 100ms 마다 워치독용 ping 전송 (신호가 끊기면 서버가 자동 정지시킴)
setInterval(function() { api('/ping'); }, 100);

document.getElementById('kbSpeed').addEventListener('input', function() {
  document.getElementById('kbSpeedVal').textContent = this.value;
});

// ══════════════════════════════════════════════════════
// 운전 시작 / 정지 (= 조종 활성화 + CSV 수집 시작/종료)
// ══════════════════════════════════════════════════════
function toggleDrive() {
  driveActive = !driveActive;
  const btn = document.getElementById('driveBtn');
  const status = document.getElementById('driveStatus');
  if (driveActive) {
    api('/drive/start').then(d => {
      document.getElementById('fileInfo').textContent = d ? d.filename : '-';
    });
    btn.textContent = '&#9632; 운전 정지';
    btn.classList.remove('btn-green'); btn.classList.add('btn-red');
    status.textContent = '운전 중 - 방향키로 조종하세요 (수집 중)';
  } else {
    api('/drive/stop');
    Object.keys(pressedKeys).forEach(k => { pressedKeys[k] = false; document.getElementById(k).classList.remove('active'); });
    btn.textContent = '▶ 운전 시작';
    btn.classList.remove('btn-red'); btn.classList.add('btn-green');
    status.textContent = '대기중 (저장 완료)';
  }
}

function emergencyStop() {
  driveActive = false;
  autoOn = false;
  Object.keys(pressedKeys).forEach(k => { pressedKeys[k] = false; document.getElementById(k).classList.remove('active'); });
  const driveBtn = document.getElementById('driveBtn');
  driveBtn.textContent = '▶ 운전 시작';
  driveBtn.classList.remove('btn-red'); driveBtn.classList.add('btn-green');
  document.getElementById('autoBtn').textContent = '자동 주행 시작';
  document.getElementById('driveStatus').textContent = '긴급 정지됨';
  api('/stop');
}

// ── 자동 주행 (자동 주행도 시작/정지 시 CSV 수집이 같이 시작/종료됩니다) ──
let autoOn = false;
document.getElementById('autoSpeed').addEventListener('input', function() {
  document.getElementById('autoSpeedVal').textContent = this.value;
});
document.getElementById('autoSteer').addEventListener('input', function() {
  document.getElementById('autoSteerVal').textContent = (this.value/100).toFixed(2);
});
function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('autoBtn');
  if (autoOn) {
    const speed = parseInt(document.getElementById('autoSpeed').value);
    const steer = parseInt(document.getElementById('autoSteer').value) / 100;
    api('/auto/start', {speed: speed, steering: steer}).then(d => {
      document.getElementById('fileInfo').textContent = d ? d.filename : '-';
    });
    btn.textContent = '자동 주행 정지';
    document.getElementById('driveStatus').textContent = '자동 주행 중 (수집 중)';
  } else {
    api('/auto/stop');
    btn.textContent = '자동 주행 시작';
    document.getElementById('driveStatus').textContent = '대기중 (저장 완료)';
  }
}

// ── 경로 + CDS 그래프 그리기 ──
const pathCanvas = document.getElementById('pathCanvas');
const pctx = pathCanvas.getContext('2d');
const PW = pathCanvas.width, PH = pathCanvas.height;
// 미터를 픽셀로 바꾸는 배율. 값이 클수록 지도가 확대되어 보입니다.
// 운동장이 커서 경로가 화면 밖으로 나가면 이 값을 줄이세요.
const METERS_TO_PIXELS = 40;

const cdsCanvas = document.getElementById('cdsCanvas');
const cctx = cdsCanvas.getContext('2d');
const CW = cdsCanvas.width, CH = cdsCanvas.height;

function cdsToColor(v) {
  // CDS 값(밝을수록 큼)을 파랑(어두움) ~ 노랑(밝음) 색으로 표현
  const t = Math.max(0, Math.min(1, v / 1000));
  const r = Math.round(40 + t * 215);
  const g = Math.round(60 + t * 195);
  const b = Math.round(200 - t * 160);
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function drawPath(trail) {
  pctx.clearRect(0, 0, PW, PH);
  const cx = PW/2, cy = PH/2;
  // 격자
  pctx.strokeStyle = 'rgba(88,166,255,0.08)'; pctx.lineWidth = 1;
  pctx.beginPath(); pctx.moveTo(cx, 0); pctx.lineTo(cx, PH); pctx.moveTo(0, cy); pctx.lineTo(PW, cy); pctx.stroke();

  for (const p of trail) {
    const px = cx + p.x * METERS_TO_PIXELS;
    const py = cy - p.y * METERS_TO_PIXELS;
    pctx.beginPath(); pctx.arc(px, py, 3, 0, Math.PI*2);
    pctx.fillStyle = cdsToColor(p.cds);
    pctx.fill();
  }
  if (trail.length > 0) {
    const last = trail[trail.length - 1];
    document.getElementById('posInfo').textContent =
      'x=' + last.x.toFixed(2) + 'm, y=' + last.y.toFixed(2) + 'm';
    document.getElementById('cdsInfo').textContent = last.cds;
  }
  document.getElementById('ptCount').textContent = trail.length;
}

function drawCdsChart(trail) {
  cctx.clearRect(0, 0, CW, CH);
  if (trail.length < 2) return;
  const recent = trail.slice(-200); // 최근 200개 점만 표시 (너무 빽빽해지지 않도록)
  const maxV = 1000;
  cctx.strokeStyle = '#58a6ff'; cctx.lineWidth = 2; cctx.beginPath();
  recent.forEach((p, i) => {
    const px = (i / (recent.length - 1)) * CW;
    const py = CH - (Math.min(p.cds, maxV) / maxV) * CH;
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py);
  });
  cctx.stroke();
}

// 300ms 마다 서버에서 최신 경로 데이터를 받아와 그림 (값이 작을수록
// 더 실시간으로 보이지만 서버에 더 자주 요청을 보내게 됩니다)
setInterval(function() {
  fetch('/track').then(r => r.json()).then(d => {
    drawPath(d.trail);
    drawCdsChart(d.trail);
  }).catch(function(){});
}, 300);
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return _HTML


@app.route('/status')
def status():
    return jsonify({"has_car": HAS_CAR, "has_cds": HAS_CDS})


@app.route('/control', methods=['POST'])
def control():
    """키보드 방향키로 계산된 (x=조향, y=속력방향) 값을 받아 오토카에 적용."""
    data = request.get_json(silent=True) or {}
    x = float(data.get('x', 0))
    y = float(data.get('y', 0))

    steering = max(-1.0, min(1.0, x))

    if y > 0.05:
        _apply_drive(direction=1, speed=int(min(1.0, y) * 99), steering=steering)
    elif y < -0.05:
        _apply_drive(direction=-1, speed=int(min(1.0, -y) * 99), steering=steering)
    else:
        _apply_drive(direction=0, speed=0, steering=steering)

    return jsonify({"status": "ok"})


@app.route('/stop', methods=['POST'])
def stop_route():
    global _auto_running
    _auto_running = False
    _stop_drive()
    return jsonify({"status": "stopped"})


@app.route('/ping', methods=['POST'])
def ping():
    global _last_ping
    _last_ping = time.time()
    return jsonify({"status": "ok"})


@app.route('/drive/start', methods=['POST'])
def drive_start():
    """
    "운전 시작" 버튼: 키보드 조종을 받을 준비 + CSV 수집을 동시에 시작합니다.
    (실제 조향/속력 명령은 /control 로 별도로 계속 전송됩니다.)
    """
    global _last_ping
    _last_ping = time.time()
    filename = _start_recording()
    return jsonify({"status": "driving", "filename": filename})


@app.route('/drive/stop', methods=['POST'])
def drive_stop():
    """"운전 정지" 버튼: 오토카를 멈추고 CSV 수집을 종료(저장)합니다."""
    _stop_drive()
    _stop_recording()
    return jsonify({"status": "stopped", "path": _record_path})


@app.route('/auto/start', methods=['POST'])
def auto_start():
    """간단한 자동 주행(이동 프로그램)을 시작하면서 CSV 수집도 같이 시작합니다."""
    global _auto_running, _auto_thread
    data = request.get_json(silent=True) or {}
    speed = int(max(0, min(99, data.get('speed', 50))))
    steering = max(-1.0, min(1.0, float(data.get('steering', 0.0))))

    if _auto_running:
        return jsonify({"status": "already_running"})

    filename = _start_recording()

    _auto_running = True
    _auto_thread = threading.Thread(
        target=_auto_drive_loop, args=(speed, steering), daemon=True
    )
    _auto_thread.start()
    return jsonify({"status": "started", "filename": filename})


@app.route('/auto/stop', methods=['POST'])
def auto_stop():
    """자동 주행을 멈추고 CSV 수집도 같이 종료(저장)합니다."""
    global _auto_running
    _auto_running = False
    _stop_drive()
    _stop_recording()
    return jsonify({"status": "stopped", "path": _record_path})


@app.route('/track')
def track_route():
    with _trail_lock:
        # 화면에는 최근 400개 점만 보내서 브라우저가 너무 무거워지지 않게 함
        trail_copy = list(_trail[-400:])
    return jsonify({"trail": trail_copy})


# ════════════════════════════════════════════════════════════
# 8. 메인 실행부
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  오토카 운동장 트랙 대시보드 시작")
    print("  AutoCar HW : " + ("연결됨" if HAS_CAR else "시뮬레이션"))
    print("  CDS 센서   : " + ("연결됨" if HAS_CDS else "시뮬레이션"))
    print("=" * 55)

    threading.Thread(target=_watchdog_loop, daemon=True).start()
    threading.Thread(target=_sampling_loop, daemon=True).start()

    print("브라우저에서 http://192.168.0.57:5000 으로 접속하세요!")
    print("종료하려면 Ctrl+C")

    # threaded=True 로 켜두면, 브라우저에서 /control(방향키 조종),
    # /ping(워치독), /track(경로 그리기) 요청이 동시에 들어와도 서버가
    # 하나씩 순서대로 처리하지 않고 요청마다 별도 스레드에서 즉시
    # 처리합니다. 즉, 경로 그래프를 불러오는 중이라도 방향키 조종
    # 신호는 기다리지 않고 바로 처리됩니다. (threaded=False로 바꾸면
    # 요청을 한 번에 하나씩만 처리해서 반응이 눈에 띄게 느려집니다.)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
