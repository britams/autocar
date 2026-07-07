# -*- coding: utf-8 -*-
"""
track_dashboard.py
──────────────────────────────────────────────────────────────
[7/3 과제] 오토카 운동장 트랙 돌기 - 센서 수집 + 웹 대시보드

■ 이 프로그램이 하는 일
  1) 오토카를 조종합니다. 조종 방법은 2가지입니다.
       (가) 컴퓨터 키보드 방향키로 직접 조종 (수동)
       (나) 일정한 속력+조향으로 원을 그리며 도는 간단한 자동주행
  2) "운전 시작"(수동) 또는 "자동 주행 시작"(자동) 버튼을 누르면
     조종이 시작되면서 동시에 CDS(조도) 센서 값 + 위치 데이터 수집이
     시작됩니다. 위치는 두 가지 방법으로 함께 추정합니다.
       - 모터 속도 + 조향각 기반 위치 추정 (track_odometry.py)
       - 카메라 영상 기반 위치 추정, Visual Odometry (track_vision.py)
         → 라인트레이싱(선 따라가기) 용도가 아니라, 오로지 위치 추정을
           모터 기반 값과 서로 비교/보완하기 위한 참고용입니다.
  3) "정지" 버튼을 누르면 오토카가 멈추고, 수집도 즉시 종료되면서
     CSV 파일이 "csv 파일 모음" 폴더에 바로 저장됩니다.
  4) 웹 브라우저(대시보드)에서 실시간으로 주행 경로와 센서 값 그래프를
     보면서 확인할 수 있습니다.

■ 조작 방법
  - 수동 조종: "운전 시작"을 누른 뒤 키보드 방향키를 누릅니다.
      ↑ : 전진        ↓ : 후진
      ← : 왼쪽으로 조향   → : 오른쪽으로 조향
    방향키에서 손을 떼면 그 즉시 오토카가 멈춥니다. (계속 누르고 있어야
    움직입니다 - 오래 눌러도 안전하도록 워치독이 같이 동작합니다.)
  - 자동 주행: "자동 주행" 카드에서 "자동 주행 시작"을 누르면 사람이
    조종하지 않아도 오토카가 정해진 속력+조향으로 스스로 움직입니다.
  - 수동/자동은 동시에 켤 수 없습니다. 한쪽을 시작하면 다른 쪽은 자동으로
    꺼집니다 (서로 조종 명령이 충돌하지 않도록).
  - "정지"를 누르면 오토카가 멈추고 그 순간까지 모은 데이터가 CSV
    파일로 저장됩니다.

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

■ 이 프로그램은 오토카 "파워선"(모터/CAN 통신 보드 전원)이 빠져 있고
  노트북과 젯슨 보드만 USB 케이블로 연결된 상태에서도 그대로 실행됩니다.
  이런 경우 오토카/CDS 센서/카메라 중 연결 안 된 것들은 자동으로
  "시뮬레이션 모드"로 동작하고, 나머지(예: 카메라만 연결된 경우)는
  정상적으로 동작합니다. (하드웨어 연결 실패는 프로그램을 멈추지 않고
  그 부분만 조용히 시뮬레이션으로 대체됩니다.)

■ 파일 구성
    track_dashboard.py  : 이 파일. Flask 웹 서버 + 대시보드 화면.
    track_odometry.py   : 모터 기반 위치(오도메트리) 계산 전용 모듈.
    track_vision.py      : 카메라 기반 위치 추정(Visual Odometry) 전용 모듈.
    csv 파일 모음/         : 수집한 CSV 파일이 자동으로 저장되는 폴더.
──────────────────────────────────────────────────────────────
"""

import sys
import os
import csv
import time
import threading

import cv2

# pop 모듈 경로 추가 (오토카 홈 디렉터리인 /home/soda 를 파이썬 경로에 넣어줌)
sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.getcwd())

from flask import Flask, jsonify, request

from track_odometry import TrackOdometry
from track_vision import VisualOdometry

# ════════════════════════════════════════════════════════════
# 0. 조정 가능한 숫자값 모음 (여기 값들을 바꾸면 동작이 어떻게 바뀌는지 설명)
# ════════════════════════════════════════════════════════════

# CDS(조도) 센서가 연결된 SPI ADC 채널 번호입니다.
# - pop.Util 의 Cds 클래스는 SPI ADC 여러 채널(0~7) 중 하나에서 값을
#   읽어오는데, 실제로 CDS 센서를 몇 번 채널에 꽂았는지에 따라 값이
#   달라집니다. 실제 오토카에서 8개 채널을 모두 읽어보며 손으로 빛을
#   가려서 테스트한 결과, 이 오토카는 7번 채널에 연결되어 있었습니다.
#   (다른 오토카 개체라면 배선이 다를 수 있으니, 값이 안 바뀌면 다시
#   0~7 사이에서 테스트해보세요.)
CDS_ADC_CHANNEL = 7

# CDS 센서 값을 한 번 읽을 때 몇 번 샘플링해서 평균낼지.
# - pop.Util 의 Cds 클래스는 기본값이 1024번인데, 이렇게 많이 샘플링
#   하면 값은 더 안정적이지만 한 번 읽는 데 걸리는 시간이 길어지고,
#   그동안 파이썬이 다른 작업(방향키 조종 명령 처리 등)을 잠깐 못 하게
#   되어 조종 반응이 느려질 수 있습니다. (마이크로컨트롤러처럼 센서만
#   전담으로 읽는 별도 하드웨어가 없어서, 소프트웨어에서 샘플 수를
#   줄여 "가볍게" 만드는 방식으로 반응 속도를 확보합니다.)
# - 값이 작을수록(예: 8): 훨씬 빠르지만 값이 조금 더 흔들릴 수 있습니다.
# - 값이 클수록(예: 1024): 더 안정적이지만 느려집니다.
#
# ※ 주의: 파이썬은 스레드가 여러 개 있어도 GIL(Global Interpreter
#   Lock)이라는 규칙 때문에 한순간에는 딱 하나의 스레드만 실제로
#   코드를 실행할 수 있습니다. 그래서 CDS 값을 읽는 배경 스레드가
#   샘플을 너무 많이/너무 자주 읽으면, 그 계산을 하는 동안 방향키
#   조종을 처리하는 스레드가 순서를 기다리게 되어 오히려 반응이
#   느려질 수 있습니다. (진짜 별도의 물리적 MCU 칩이 있다면 이런
#   경합이 없겠지만, 이 오토카의 CDS 센서는 젯슨 보드에 SPI로 직접
#   연결되어 있어서 파이썬 스레드로 흉내낼 수밖에 없습니다.) 그래서
#   기본값을 낮게(8) 잡았습니다.
CDS_SAMPLE_COUNT = 8

# CDS 센서를 전담으로 읽는 배경 스레드가 한 번 읽고 나서 얼마나
# 쉬었다가 또 읽을지 (초 단위). 이 값은 CSV에 기록되는 주기
# (SAMPLE_INTERVAL_SEC, 기본 0.1초)보다 더 빠르게 읽을 필요가 없어서
# 똑같이 맞췄습니다 - 더 빠르게 읽어봐야 어차피 못 쓰이고 GIL 경합만
# 늘어납니다.
# - 값이 작을수록: CDS 값이 더 최신 상태로 자주 갱신되지만 CPU를 더 쓰고,
#   조종 반응 속도에 영향을 줄 수 있습니다.
# - 값이 클수록: CPU 사용/GIL 경합은 줄지만 값 갱신이 조금 늦어질 수 있음.
CDS_READ_INTERVAL_SEC = 0.1

# 카메라 프레임 크기 - 작을수록 처리 속도(FPS)가 빨라지고, 클수록
# 화질은 좋아지지만 Visual Odometry 계산이 느려집니다.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 20

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

# True로 두면, 이 프로그램을 새로 실행할 때마다(=보통 오토카를 껐다
# 켜고 다시 실행할 때마다) DATA_DIR 안에 있던 CSV 파일을 전부 지우고
# 깨끗한 상태로 시작합니다.
#
# ※ 주의: 젯슨 보드는 파이썬 코드 안에서 "방금 전원이 켜졌는지"를 직접
#   알 방법이 없습니다. 그래서 "전원 켤 때"가 아니라 "이 스크립트가
#   시작될 때"를 기준으로 삼았습니다. 실제로는 오토카를 껐다 켤 때마다
#   이 스크립트도 다시 실행해야 하니 결과적으로 같은 효과를 내지만,
#   전원을 끄지 않고 Ctrl+C로 껐다가 다시 실행해도 똑같이 지워진다는
#   점을 꼭 기억하세요. 다시 실행하기 전에는 필요한 CSV를 미리
#   노트북으로 옮겨두는 것이 안전합니다.
CLEAR_CSV_ON_STARTUP = True


def _reset_data_dir():
    """DATA_DIR 안의 CSV 파일을 전부 지워서 매 실행마다 깨끗하게 시작합니다."""
    if not CLEAR_CSV_ON_STARTUP:
        return
    if not os.path.isdir(DATA_DIR):
        return
    removed = 0
    for name in os.listdir(DATA_DIR):
        if name.endswith(".csv"):
            os.remove(os.path.join(DATA_DIR, name))
            removed += 1
    print("[csv 파일 모음] 이전 CSV %d개를 정리하고 새로 시작합니다." % removed)


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
    from pop import Cds  # Cds 클래스는 pop.Util이 아니라 pop 최상위에 있음
    cds_sensor = Cds(channel=CDS_ADC_CHANNEL)
    cds_sensor.setSample(CDS_SAMPLE_COUNT)  # 반응 속도를 위해 샘플 수를 가볍게 줄임
    HAS_CDS = True
    print("[CDS 센서] 연결 성공 (채널 %d, 샘플 %d개)" % (CDS_ADC_CHANNEL, CDS_SAMPLE_COUNT))
except Exception as _e:
    print("[CDS 센서] 시뮬레이션 모드로 동작합니다 (센서 연결 실패: %s)" % _e)
finally:
    for _attr in ('np', 'torchvision'):
        if hasattr(_builtins, _attr):
            delattr(_builtins, _attr)

# 카메라는 아직 열지 않습니다! GStreamer 카메라는 열리는 데 시간이 걸리거나
# 드물게 멈춰버릴 수 있는데, 만약 프로그램이 시작되는 이 시점에 바로
# 열려고 하면 카메라가 느릴 때 웹 서버(app.run())가 시작되는 지점까지
# 코드가 도달하지 못해서 브라우저 접속 자체가 안 되는 문제가 생깁니다.
# 그래서 실제 카메라 열기는 아래 _camera_loop() 배경 스레드 안에서,
# "웹 서버가 이미 켜진 뒤에" 한 번만 시도하도록 미뤄뒀습니다.
HAS_CAM = False
_camera = None


def _read_cds_hardware():
    """
    CDS(조도) 센서 값을 실제로 하나 읽어옵니다. (시간이 조금 걸릴 수 있는
    "느린" 부분 - 그래서 아래 _cds_reader_loop() 배경 스레드 안에서만
    호출하고, 조종 명령을 처리하는 코드에서는 절대 직접 부르지 않습니다.)

    ※ pop.Cds 클래스의 readAverage()는 전압값을 "럭스(lux, 밝기 단위)"로
    억지로 환산해서 정수로 반올림해버리는 함수라서, 값이 몇 단계로만
    뚝뚝 끊겨 나옵니다(예: 계속 1만 나오는 문제). 그래서 여기서는 그
    대신 readVoltAverage()를 불러서, 실제로 센서가 측정한 "아날로그
    전압값"(0.0V ~ 3.3V 사이의 연속된 소수 값)을 그대로 사용합니다.
    - 실제 센서가 있으면 진짜 전압을 반환합니다. (밝을수록 전압이 변함 -
      정확히 어느 방향으로 변하는지는 센서 회로 설계에 따라 다르므로,
      대시보드에서 손으로 가려보며 값이 오르는지 내리는지 직접 확인하세요.)
    - 센서가 없는 시뮬레이션 모드에서는, 그래도 대시보드 테스트를 할 수
      있도록 시간에 따라 부드럽게 오르내리는 가짜 전압값(0.0~3.3V)을
      만들어 줍니다.
    """
    if HAS_CDS:
        return round(cds_sensor.readVoltAverage(), 4)
    import math
    return round(1.65 + 1.5 * math.sin(time.time() * 0.5), 4)


# CDS 값을 "지금 막 읽은 최신 값"으로 캐시해두는 공용 변수.
# _cds_reader_loop() 배경 스레드가 계속 새로 채워 넣고, read_cds_value()는
# 이 값을 즉시 꺼내 쓰기만 해서 절대 오래 기다리지 않습니다.
_cds_value_lock = threading.Lock()
_latest_cds_value = 0


def read_cds_value():
    """
    CDS 값이 필요할 때(조종/기록 코드에서) 부르는 함수. 실제 센서를
    읽는 게 아니라, 배경 스레드가 미리 읽어서 캐시해둔 "가장 최근 값"을
    그냥 꺼내오기만 하므로 거의 즉시(0에 가까운 시간) 끝납니다.
    """
    with _cds_value_lock:
        return _latest_cds_value


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
_odometry = TrackOdometry()          # 모터 기반 위치 추정
_visual_odometry = VisualOdometry()  # 카메라 기반 위치 추정 (Visual Odometry, 참고/보완용)

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
    """
    "이 방향/속력/조향으로 달리고 싶다"는 목표 상태를 저장만 하는 함수.

    CDS 센서를 전용 스레드로 뺀 것과 똑같은 이유로, 실제 오토카에 CAN
    통신 명령을 내려보내는 부분은 여기서 하지 않습니다. car.steering=...,
    car.setSpeed(), car.forward() 같은 명령은 CAN 통신(전선으로 모터
    제어 보드와 주고받는 통신)이라 몇 ms 정도 걸릴 수 있는데, 방향키를
    누를 때마다 이 통신까지 같이 기다리면 그만큼 조종 반응이 느려집니다.
    그래서 여기서는 목표값 저장만 아주 빠르게 끝내고, 실제 하드웨어
    통신은 아래 _drive_writer_loop() 전용 스레드가 쉬지 않고 전담해서
    처리합니다.
    """
    with _state_lock:
        _state["direction"] = direction
        _state["speed"] = speed
        _state["steering"] = steering


def _send_drive_to_hardware(direction, speed, steering):
    """
    실제로 오토카에 CAN 명령을 내려보내는 부분입니다. _drive_writer_loop()
    전용 스레드 안에서만 호출됩니다 - 방향키 조종을 처리하는 /control
    요청 코드는 이 함수를 직접 부르지 않고 절대 기다리지 않습니다.

    오토카의 "파워선"(모터/CAN 통신 보드에 전기를 공급하는 선)이 빠져
    있으면, car.steering / car.forward() 같은 명령이 응답을 못 받아서
    에러를 내거나 멈춰버릴 수 있습니다. 여기서 try/except로 감싸서,
    그런 경우에도 프로그램 전체가 죽지 않고 "이번 명령은 실패했다"고만
    조용히 넘어가도록 만들었습니다. (모터는 당연히 안 움직이지만, 카메라
    기반 위치 추정이나 대시보드 화면 자체는 계속 정상 동작합니다.)
    """
    if not car:
        return

    try:
        car.steering = steering
        if direction > 0:
            car.setSpeed(speed)
            car.forward()
        elif direction < 0:
            car.setSpeed(speed)
            car.backward()
        else:
            car.stop()
    except Exception as _e:
        # 파워선이 빠졌거나 CAN 통신이 응답하지 않는 상황 등.
        # 여기서 예외를 처리하지 않으면 이 배경 스레드가 통째로 멈춰서
        # 그 뒤로 오토카에 아무 명령도 전달되지 않게 됩니다.
        print("[오토카] 명령 전달 실패 (파워선 확인 필요): %s" % _e)


def _stop_drive():
    """
    즉시 정지가 필요할 때(워치독, 긴급정지, 운전/자동주행 정지) 씁니다.
    목표 상태를 0으로 바꿔두는 것과 동시에, 전용 스레드의 다음 차례를
    기다리지 않고 이 자리에서 바로 한 번 더 정지 명령을 직접 보내서
    최대한 빨리 멈추도록 합니다.
    """
    _apply_drive(0, 0, 0.0)
    _send_drive_to_hardware(0, 0, 0.0)


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
# 3-0. 배경 스레드 0 - 주행 명령 전용 쓰기 스레드 ("작은 MCU"처럼 동작)
#    CDS 센서 전용 스레드와 똑같은 구조입니다. 이 스레드 하나만 쉬지
#    않고 "목표 상태(_state)"를 오토카 하드웨어(CAN 통신)에 실제로
#    전달합니다. 방향키 조종을 처리하는 /control 요청은 목표 상태를
#    저장만 하고 이 스레드를 절대 기다리지 않으므로, CAN 통신이 몇 ms
#    걸리든 조종 반응 속도에는 영향을 주지 않습니다.
# ════════════════════════════════════════════════════════════
# 이 스레드가 목표 상태를 오토카에 다시 전달하는 주기(초). 방향키를
# 계속 누르고 있으면 매번 같은 값을 다시 보내는 것뿐이라, 너무 빠르게
# 반복할 필요는 없습니다.
# - 값이 작을수록(예: 0.01): 목표가 바뀌었을 때 더 빨리 반영되지만
#   CAN 통신 횟수가 늘어나 다른 스레드와의 GIL 경합이 늘어날 수 있음.
# - 값이 클수록(예: 0.1): 통신 횟수는 줄지만 목표가 바뀐 뒤 실제
#   반영까지 그만큼 시간이 더 걸림.
DRIVE_WRITE_INTERVAL_SEC = 0.02  # 초당 최대 50번


def _drive_writer_loop():
    while True:
        with _state_lock:
            direction = _state["direction"]
            speed = _state["speed"]
            steering = _state["steering"]
        _send_drive_to_hardware(direction, speed, steering)
        time.sleep(DRIVE_WRITE_INTERVAL_SEC)


# ════════════════════════════════════════════════════════════
# 3-1. 배경 스레드 1-B - CDS 센서 전용 읽기 스레드 ("작은 MCU"처럼 동작)
#    이 스레드 하나만 쉬지 않고 CDS 센서를 읽고 _latest_cds_value에
#    저장합니다. 조종 명령을 처리하는 코드는 이 스레드를 전혀 기다리지
#    않고 캐시된 값만 즉시 읽어가므로, 센서를 읽는 데 걸리는 시간이
#    조종 반응 속도에 영향을 주지 않습니다.
# ════════════════════════════════════════════════════════════
def _cds_reader_loop():
    global _latest_cds_value
    while True:
        value = _read_cds_hardware()
        with _cds_value_lock:
            _latest_cds_value = value
        time.sleep(CDS_READ_INTERVAL_SEC)


# ════════════════════════════════════════════════════════════
# 3-2. 배경 스레드 1-C - 카메라 캡처 + Visual Odometry 갱신
#    카메라가 있으면 계속 새 프레임을 받아와 VisualOdometry 위치를
#    갱신합니다. 카메라가 없는(시뮬레이션) 환경이면 그냥 잠깐씩
#    쉬면서 아무 것도 하지 않습니다. (라인트레이싱 등 다른 용도로는
#    쓰지 않고, 오직 위치 추정 보완용으로만 사용합니다.)
# ════════════════════════════════════════════════════════════
def _open_camera():
    """
    실제로 카메라를 여는 부분. 이 함수는 웹 서버가 이미 켜진 뒤에,
    별도 스레드 안에서만 호출됩니다 - 그래서 카메라가 열리는 데
    오래 걸리거나 실패해도(예: 오토카 파워선이 빠져 카메라가 없는
    상태) 웹 대시보드 접속 자체에는 영향이 없습니다.

    1순위: pop.Util.gstrmer() 로 만든 GStreamer 파이프라인 (오토카 전용 카메라)
    2순위: 그냥 일반 USB 웹캠(cv2.VideoCapture(0))
    둘 다 실패하면 카메라 없이 시뮬레이션 모드로 동작합니다.
    """
    global HAS_CAM, _camera

    # isOpened()만 확인하면 "파이프라인 객체는 만들어졌지만 실제로는
    # 사진을 한 장도 못 찍는" 상태(예: 물리적으로 카메라가 없는 경우)를
    # "성공"으로 잘못 판단할 수 있습니다. (실제로 오토카에서 "No cameras
    # available"이라는 오류가 났는데도 "연결 성공"이라고 나온 사례가
    # 있었습니다.) 그래서 실제로 사진을 한 장 읽어보고, 진짜로 읽히는지
    # 확인한 뒤에만 "성공"으로 판단하도록 만들었습니다.
    def _try_read_one_frame(cap, tries=5, wait_sec=0.3):
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(wait_sec)
        return False

    try:
        from pop.Util import gstrmer
        _cam_pipeline = gstrmer(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, flip=0)
        _camera = cv2.VideoCapture(_cam_pipeline, cv2.CAP_GSTREAMER)
        if _camera.isOpened() and _try_read_one_frame(_camera):
            HAS_CAM = True
            print("[카메라] 연결 성공 (GStreamer)")
            return
        _camera.release()
        raise RuntimeError("GStreamer 카메라에서 실제 프레임을 받지 못했습니다.")
    except Exception:
        pass

    try:
        _camera = cv2.VideoCapture(0)
        if _camera.isOpened() and _try_read_one_frame(_camera):
            HAS_CAM = True
            print("[카메라] 연결 성공 (기본 웹캠)")
        else:
            _camera.release()
            print("[카메라] 시뮬레이션 모드로 동작합니다 (카메라를 찾지 못함)")
    except Exception as _e2:
        print("[카메라] 시뮬레이션 모드로 동작합니다 (연결 실패: %s)" % _e2)


def _camera_loop():
    _open_camera()  # 웹 서버가 이미 실행 중인 상태에서 카메라를 엽니다.

    while True:
        if not HAS_CAM:
            time.sleep(0.5)
            continue

        ok, frame = _camera.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        _visual_odometry.update(frame)


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
                    _visual_odometry.x, _visual_odometry.y,
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
    "x_m", "y_m",        # 모터+조향각 기반(오도메트리) 위치 (미터)
    "heading_deg",       # 오도메트리로 추정한 진행 방향 (도)
    "vo_x_m", "vo_y_m",  # 카메라 기반(Visual Odometry) 위치 (미터, 참고용)
    "speed",             # 그 순간의 부호 있는 속력 (-99~99)
    "steering",          # 그 순간의 조향 값 (-1~1)
    "cds_value",         # CDS 조도 센서의 아날로그 전압값 (단위: V, 0.0~3.3)
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
    _visual_odometry.reset()

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
  <span class="badge" id="camBadge">CAM: 확인중...</span>
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
    <div class="stat"><span>현재 위치 (모터 기반)</span><b id="posInfo">x=0.00m, y=0.00m</b></div>
    <div class="stat"><span>현재 위치 (카메라 기반, 참고용)</span><b id="voInfo">x=0.00m, y=0.00m</b></div>
    <div class="stat"><span>현재 CDS 전압값 (V)</span><b id="cdsInfo">-</b></div>
    <h2 style="margin-top:4px;">CDS 전압값 변화 그래프 (0~3.3V)</h2>
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
  const cam = document.getElementById('camBadge');
  cam.textContent = 'CAM: ' + (d.has_cam ? '연결됨' : '시뮬레이션');
  cam.style.color = d.has_cam ? '#2ecc71' : '#e3b341';
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
//   - 수동(키보드)과 자동 주행은 동시에 켤 수 없습니다.
//     한쪽을 시작하면 다른 쪽은 자동으로 꺼서 서로 충돌하지 않게 합니다.
//   - 중요: 모드를 서로 전환할 때는 "/stop"(오토카만 잠깐 멈춤)을 쓰고
//     "/drive/stop", "/auto/stop"(오토카를 멈추면서 CSV 파일도 닫음)은
//     쓰지 않습니다. 예전에는 모드를 바꿀 때마다 CSV 파일이 끊기고
//     새로 생겨서, "운전 시작~운전 정지" 한 번 사이에 파일이 여러 개로
//     쪼개지는 문제가 있었습니다. 이제는 실제로 "운전 정지"/"자동 주행
//     정지" 버튼을 눌렀을 때만 파일이 닫히고, 중간에 모드를 바꿔도
//     하나의 파일에 계속 이어서 저장됩니다.
// ══════════════════════════════════════════════════════
function toggleDrive() {
  driveActive = !driveActive;
  const btn = document.getElementById('driveBtn');
  const status = document.getElementById('driveStatus');
  if (driveActive) {
    if (autoOn) { autoOn = false; api('/stop'); resetAutoButton(); }
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
  resetAutoButton();
  document.getElementById('driveStatus').textContent = '긴급 정지됨';
  api('/stop');
}

// ── 자동 주행 (시작/정지 시 CSV 수집이 같이 시작/종료됩니다) ──
let autoOn = false;
document.getElementById('autoSpeed').addEventListener('input', function() {
  document.getElementById('autoSpeedVal').textContent = this.value;
});
document.getElementById('autoSteer').addEventListener('input', function() {
  document.getElementById('autoSteerVal').textContent = (this.value/100).toFixed(2);
});

function resetAutoButton() {
  document.getElementById('autoBtn').textContent = '자동 주행 시작';
}

function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('autoBtn');
  if (autoOn) {
    if (driveActive) { driveActive = false; api('/stop'); toggleDriveButtonReset(); }
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

function toggleDriveButtonReset() {
  const driveBtn = document.getElementById('driveBtn');
  driveBtn.textContent = '▶ 운전 시작';
  driveBtn.classList.remove('btn-red'); driveBtn.classList.add('btn-green');
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

// CDS 값은 이제 럭스(lux)가 아니라 센서가 실제로 측정한 아날로그
// 전압값(0.0V ~ 3.3V)입니다. 그래서 색/그래프 눈금도 0~3.3 범위에
// 맞춰뒀습니다.
const CDS_MAX_VOLT = 3.3;

function cdsToColor(v) {
  // CDS 전압값(0~3.3V)을 파랑(낮음) ~ 노랑(높음) 색으로 표현
  const t = Math.max(0, Math.min(1, v / CDS_MAX_VOLT));
  const r = Math.round(40 + t * 215);
  const g = Math.round(60 + t * 195);
  const b = Math.round(200 - t * 160);
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

// 오토카의 "현재 위치"를 화면 정중앙에 고정시키는 방식입니다.
//   - 예전 방식: 시작 지점(0,0)을 화면 중앙에 고정 → 오토카가 멀리
//     가면 화면 밖으로 나가버림.
//   - 지금 방식: 매번 "가장 최근 위치(마지막 점)"를 화면 중앙에 두고,
//     지나온 점들은 그 최근 위치를 기준으로 상대 좌표를 계산해서
//     그립니다. 그래서 오토카는 항상 정중앙에 있고, 지나온 경로가
//     오토카 뒤쪽으로 흘러가는 것처럼 보입니다.
function drawPath(trail) {
  pctx.clearRect(0, 0, PW, PH);
  const cx = PW/2, cy = PH/2;
  // 격자
  pctx.strokeStyle = 'rgba(88,166,255,0.08)'; pctx.lineWidth = 1;
  pctx.beginPath(); pctx.moveTo(cx, 0); pctx.lineTo(cx, PH); pctx.moveTo(0, cy); pctx.lineTo(PW, cy); pctx.stroke();

  if (trail.length === 0) {
    document.getElementById('ptCount').textContent = 0;
    return;
  }

  // 기준점 = 가장 최근(현재) 위치. 모든 점을 이 기준점으로부터 얼마나
  // 떨어져 있는지로 다시 계산해서, 현재 위치가 항상 (cx, cy)에 오게 함.
  const origin = trail[trail.length - 1];

  for (const p of trail) {
    const px = cx + (p.x - origin.x) * METERS_TO_PIXELS;
    const py = cy - (p.y - origin.y) * METERS_TO_PIXELS;
    pctx.beginPath(); pctx.arc(px, py, 3, 0, Math.PI*2);
    pctx.fillStyle = cdsToColor(p.cds);
    pctx.fill();
  }

  // 현재 위치(항상 화면 정중앙)를 오토카 아이콘처럼 눈에 띄게 표시
  pctx.beginPath(); pctx.arc(cx, cy, 7, 0, Math.PI*2);
  pctx.fillStyle = '#ffffff';
  pctx.fill();
  pctx.strokeStyle = '#58a6ff'; pctx.lineWidth = 2; pctx.stroke();

  document.getElementById('posInfo').textContent =
    'x=' + origin.x.toFixed(2) + 'm, y=' + origin.y.toFixed(2) + 'm';
  document.getElementById('cdsInfo').textContent = origin.cds.toFixed(3) + 'V';
  document.getElementById('ptCount').textContent = trail.length;
}

function drawCdsChart(trail) {
  cctx.clearRect(0, 0, CW, CH);
  if (trail.length < 2) return;
  const recent = trail.slice(-200); // 최근 200개 점만 표시 (너무 빽빽해지지 않도록)
  cctx.strokeStyle = '#58a6ff'; cctx.lineWidth = 2; cctx.beginPath();
  recent.forEach((p, i) => {
    const px = (i / (recent.length - 1)) * CW;
    const py = CH - (Math.min(p.cds, CDS_MAX_VOLT) / CDS_MAX_VOLT) * CH;
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
    if (d.vo) {
      document.getElementById('voInfo').textContent =
        'x=' + d.vo.x.toFixed(2) + 'm, y=' + d.vo.y.toFixed(2) + 'm';
    }
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
    return jsonify({"has_car": HAS_CAR, "has_cds": HAS_CDS, "has_cam": HAS_CAM})


@app.route('/control', methods=['POST'])
def control():
    """키보드 방향키로 계산된 (x=조향, y=속력방향) 값을 받아 오토카에 적용.

    자동 주행이 켜져있는 동안에는 무시합니다 - 그렇지 않으면
    자동주행 스레드와 방향키 명령이 동시에 오토카에 명령을 내려서
    서로 충돌(덜컹거림)할 수 있기 때문입니다.
    """
    if _auto_running:
        return jsonify({"status": "ignored_auto_active"})

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
    자동 주행이 켜져 있었다면 충돌하지 않도록 먼저 꺼줍니다.
    """
    global _last_ping, _auto_running
    _auto_running = False
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
    """
    일정 속력+조향으로 원을 그리며 도는 단순 자동주행을 시작하면서
    CSV 수집도 같이 시작합니다. 수동 조종(키보드)이 켜져 있었다면
    충돌하지 않도록 프론트엔드에서 미리 꺼주지만, 혹시 몰라 여기서도
    안전하게 오토카를 한 번 세웁니다.
    """
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
    return jsonify({
        "trail": trail_copy,
        "vo": {"x": round(_visual_odometry.x, 3), "y": round(_visual_odometry.y, 3)},
    })


# ════════════════════════════════════════════════════════════
# 8. 메인 실행부
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _reset_data_dir()

    print("=" * 55)
    print("  오토카 운동장 트랙 대시보드 시작")
    print("  AutoCar HW : " + ("연결됨" if HAS_CAR else "시뮬레이션") + " (전용 스레드에서 조종 명령 전달 중)")
    print("  CDS 센서   : " + ("연결됨" if HAS_CDS else "시뮬레이션") + " (전용 스레드에서 읽는 중)")
    print("  카메라     : 백그라운드에서 여는 중... (대시보드의 CAM 배지에서 확인)")
    print("=" * 55)

    threading.Thread(target=_watchdog_loop, daemon=True).start()
    threading.Thread(target=_drive_writer_loop, daemon=True).start()
    threading.Thread(target=_cds_reader_loop, daemon=True).start()
    threading.Thread(target=_sampling_loop, daemon=True).start()
    # 카메라는 열리는 데 시간이 걸릴 수 있어서 별도 스레드에서 엽니다.
    # 이 스레드가 아직 카메라를 여는 중이어도 아래 app.run()은 곧바로
    # 실행되어 웹 서버가 먼저 뜨므로, 카메라 문제가 대시보드 접속 자체를
    # 막지 않습니다.
    threading.Thread(target=_camera_loop, daemon=True).start()

    print("브라우저에서 http://192.168.0.57:5000 으로 접속하세요!")
    print("종료하려면 Ctrl+C")

    # threaded=True 로 켜두면, 브라우저에서 /control(방향키 조종),
    # /ping(워치독), /track(경로 그리기) 요청이 동시에 들어와도 서버가
    # 하나씩 순서대로 처리하지 않고 요청마다 별도 스레드에서 즉시
    # 처리합니다. 즉, 경로 그래프를 불러오는 중이라도 방향키 조종
    # 신호는 기다리지 않고 바로 처리됩니다. (threaded=False로 바꾸면
    # 요청을 한 번에 하나씩만 처리해서 반응이 눈에 띄게 느려집니다.)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
