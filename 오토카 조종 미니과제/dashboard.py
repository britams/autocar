# -*- coding: utf-8 -*-
"""
오토카 조종 대시보드
Flask + 브라우저 UI
실행: python3 dashboard.py
접속: http://192.168.0.47:5000
"""

import sys
import os
import threading
import time
import wave
import struct
import math
import subprocess
import tempfile

# pop 모듈 경로 추가
sys.path.insert(0, os.path.expanduser('~'))  # /home/soda/
sys.path.insert(0, os.getcwd())

import cv2
from flask import Flask, Response, jsonify, request

# ───────────────────────────────────────────────
# 오토카 하드웨어 초기화
# ───────────────────────────────────────────────
# pop/Pilot.py가 module-level에서 np.array()를 사용하지만
# Pilot.py 자체에는 numpy import가 없어서 터미널 실행 시 NameError 발생.
# builtins에 np를 주입해서 해결.
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
    print("[AutoCar] 하드웨어 연결 성공")
except Exception as _e:
    print("[AutoCar] 시뮬레이션 모드: " + str(_e))
finally:
    for _attr in ('np', 'torchvision'):
        if hasattr(_builtins, _attr):
            delattr(_builtins, _attr)

# ───────────────────────────────────────────────
# 워치독: 핑 200ms 이상 없으면 자동 정지
# ───────────────────────────────────────────────
_last_ping = time.time()

def _watchdog():
    global _last_ping
    while True:
        time.sleep(0.05)
        if car and time.time() - _last_ping > 0.2:
            car.stop()

# ───────────────────────────────────────────────
# TTS 초기화
# ───────────────────────────────────────────────
HAS_GTTS = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    print("[TTS] gTTS 없음 -- pip install gTTS")

# ───────────────────────────────────────────────
# 카메라 스레드
# ───────────────────────────────────────────────
_camera_active = False
_latest_frame = None
_frame_lock = threading.Lock()


def _camera_loop():
    global _latest_frame
    cap = None
    opened = False
    while True:
        if not _camera_active:
            time.sleep(0.1)
            continue
        if not opened:
            try:
                from pop import Util
                gst = Util.gstrmer(width=640, height=480, fps=30, flip=0)
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            except Exception:
                cap = cv2.VideoCapture(0)
            opened = True
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        with _frame_lock:
            _latest_frame = buf.tobytes()


# ───────────────────────────────────────────────
# 멜로디 생성 (pyaudio 없이 WAV + aplay)
# ───────────────────────────────────────────────
MELODIES = {
    'star_wars': [
        (392.0, 0.35), (392.0, 0.35), (392.0, 0.35),
        (311.1, 0.25), (466.2, 0.10),
        (392.0, 0.35), (311.1, 0.25), (466.2, 0.10), (392.0, 0.70),
        (587.3, 0.35), (587.3, 0.35), (587.3, 0.35),
        (622.3, 0.25), (466.2, 0.10),
        (369.9, 0.35), (311.1, 0.25), (466.2, 0.10), (392.0, 0.70),
    ],
    'imperial': [
        (220.0, 0.40), (220.0, 0.40), (220.0, 0.40),
        (174.6, 0.30), (261.6, 0.10),
        (220.0, 0.35), (174.6, 0.30), (261.6, 0.10), (220.0, 0.70),
        (329.6, 0.40), (329.6, 0.40), (329.6, 0.40),
        (349.2, 0.30), (261.6, 0.10),
        (207.7, 0.35), (174.6, 0.30), (261.6, 0.10), (220.0, 0.70),
    ],
    'nokia': [
        (659.3, 0.15), (587.3, 0.15), (369.9, 0.30), (415.3, 0.30),
        (554.4, 0.15), (493.9, 0.15), (293.7, 0.30), (329.6, 0.30),
        (493.9, 0.15), (440.0, 0.15), (277.2, 0.30), (329.6, 0.30),
        (440.0, 0.60),
    ],
}


def _make_wav(notes, filename):
    """note 리스트를 WAV 파일로 저장"""
    rate = 44100
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for freq, dur in notes:
            n = int(rate * dur)
            for i in range(n):
                t = i / float(rate)
                env = math.exp(-2.5 * t / dur)
                val = int(32767 * 0.4 * math.sin(2 * math.pi * freq * t) * env)
                wf.writeframes(struct.pack('<h', val))
            # 음 사이 짧은 묵음
            for _ in range(int(rate * 0.03)):
                wf.writeframes(struct.pack('<h', 0))


def _play_melody_thread(name):
    notes = MELODIES.get(name, MELODIES['star_wars'])
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    try:
        _make_wav(notes, tmp)
        subprocess.call(['aplay', '-q', tmp])
    except Exception as e:
        print("[Melody] 오류: " + str(e))
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ───────────────────────────────────────────────
# Flask 앱
# ───────────────────────────────────────────────
app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오토카 대시보드</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background: #161b22;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #30363d;
  }
  header h1 { font-size: 1.2em; color: #58a6ff; letter-spacing: 1px; }
  #hwBadge {
    font-size: 0.75em; padding: 2px 8px; border-radius: 12px;
    background: #21262d; border: 1px solid #30363d;
  }
  #connBadge {
    margin-left: auto; font-size: 0.75em; padding: 2px 8px;
    border-radius: 12px; background: #1f2c1f;
    color: #2ecc71; border: 1px solid #2ecc71;
  }
  .body-grid {
    display: grid;
    grid-template-columns: 280px 1fr 220px;
    gap: 12px; padding: 12px; flex: 1; overflow: hidden;
  }
  .card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 12px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .card h2 {
    font-size: 0.82em; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #21262d; padding-bottom: 6px;
  }
  #camBox {
    background: #0d1117; border-radius: 8px;
    border: 1px solid #21262d; aspect-ratio: 4/3;
    display: flex; align-items: center; justify-content: center;
    color: #484f58; font-size: 0.9em; overflow: hidden;
  }
  #camImg { width: 100%; height: 100%; object-fit: cover; display: none; }
  .sl-row { display: flex; flex-direction: column; gap: 4px; }
  .sl-row label {
    display: flex; justify-content: space-between;
    font-size: 0.78em; color: #8b949e;
  }
  input[type=range] { width: 100%; accent-color: #58a6ff; cursor: pointer; }
  .btn {
    border: none; border-radius: 6px; padding: 8px 12px;
    font-size: 0.85em; font-weight: 600; cursor: pointer;
    transition: filter 0.15s, transform 0.1s; width: 100%;
  }
  .btn:hover  { filter: brightness(1.15); }
  .btn:active { transform: scale(0.97); }
  .btn-blue  { background: #1f6feb; color: #fff; }
  .btn-red   { background: #da3633; color: #fff; }
  .btn-dark  { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  .btn-green { background: #238636; color: #fff; }
  .btn-on    { background: #da3633 !important; }
  .joy-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; flex: 1;
  }
  #joyCanvas { cursor: grab; touch-action: none; border-radius: 50%; display: block; }
  #joyCanvas:active { cursor: grabbing; }
  #joyInfo { font-size: 0.8em; color: #484f58; font-family: monospace; }
  .tts-input {
    width: 100%; background: #0d1117; border: 1px solid #30363d;
    border-radius: 6px; color: #c9d1d9; font-size: 0.82em; padding: 6px 8px;
  }
  .tts-input:focus { outline: none; border-color: #58a6ff; }
  footer {
    background: #161b22; padding: 5px 16px;
    font-size: 0.72em; color: #484f58; border-top: 1px solid #21262d;
  }
</style>
</head>
<body>

<header>
  <h1>&#128663; 오토카 대시보드</h1>
  <span id="hwBadge">HW: 확인중...</span>
  <span id="connBadge">● 대기중</span>
</header>

<div class="body-grid">

  <!-- 왼쪽: 카메라 + Pan/Tilt -->
  <div class="card">
    <h2>&#128247; 카메라</h2>
    <div id="camBox">
      <span id="camOffLabel">카메라 꺼짐</span>
      <img id="camImg" alt="camera stream">
    </div>
    <button class="btn btn-blue" id="camBtn" onclick="toggleCamera()">카메라 ON</button>
    <div class="sl-row">
      <label>Pan 좌우 <span id="panVal">90&#176;</span></label>
      <input type="range" id="panSlider" min="0" max="180" value="90">
    </div>
    <div class="sl-row">
      <label>Tilt 상하 <span id="tiltVal">0&#176;</span></label>
      <input type="range" id="tiltSlider" min="0" max="90" value="0">
    </div>
  </div>

  <!-- 가운데: 조이스틱 -->
  <div class="card">
    <h2>&#128505; 조이스틱 조종</h2>
    <div class="joy-wrap">
      <canvas id="joyCanvas" width="260" height="260"></canvas>
      <div id="joyInfo">X: 0.000 &nbsp;|&nbsp; Y: 0.000</div>
      <button class="btn btn-red" onclick="emergencyStop()">&#9209; 긴급 정지</button>
    </div>
  </div>

  <!-- 오른쪽: TTS + 멜로디 -->
  <div class="card">
    <h2>&#128266; 음성 &amp; 멜로디</h2>
    <div class="sl-row">
      <label>TTS 문구</label>
      <input class="tts-input" type="text" id="ttsText"
             value="안녕하세요! 오토카가 출발합니다." placeholder="안내 문구 입력">
    </div>
    <button class="btn btn-green" onclick="playTTS()">&#128266; TTS 재생</button>
    <hr style="border-color:#21262d;">
    <label style="font-size:0.78em; color:#8b949e;">멜로디</label>
    <button class="btn btn-dark" onclick="playMelody('star_wars')">&#127925; 스타워즈 테마</button>
    <button class="btn btn-dark" style="margin-top:2px;" onclick="playMelody('imperial')">&#127925; 임페리얼 마치</button>
    <button class="btn btn-dark" style="margin-top:2px;" onclick="playMelody('nokia')">&#127925; 노키아 벨소리</button>
    <hr style="border-color:#21262d;">
    <div style="font-size:0.75em; color:#8b949e; line-height:1.8;">
      HW: <span id="hwInfo">-</span><br>
      TTS: <span id="ttsInfo">-</span>
    </div>
  </div>

</div>

<footer>조이스틱 드래그로 주행 &nbsp;|&nbsp; Pan/Tilt 슬라이더로 카메라 방향 &nbsp;|&nbsp; TTS/멜로디로 음성 출력</footer>

<script>
// ── API 함수 ──
async function api(path, body) {
  try {
    const r = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const badge = document.getElementById('connBadge');
    badge.textContent = '● 연결됨';
    badge.style.color = '#2ecc71';
    badge.style.borderColor = '#2ecc71';
    return await r.json();
  } catch(e) {
    const badge = document.getElementById('connBadge');
    badge.textContent = '● 오프라인';
    badge.style.color = '#f85149';
    badge.style.borderColor = '#f85149';
  }
}

// ── 상태 조회 ──
fetch('/status').then(r => r.json()).then(d => {
  const hw = document.getElementById('hwBadge');
  hw.textContent = 'HW: ' + (d.has_car ? '연결됨' : '시뮬레이션');
  hw.style.color = d.has_car ? '#2ecc71' : '#e3b341';
  document.getElementById('hwInfo').textContent = d.has_car ? '연결됨' : '시뮬레이션';
  document.getElementById('ttsInfo').textContent = d.has_tts ? '사용가능' : '미설치';
}).catch(function(){});

// ── 조이스틱 ──
const canvas = document.getElementById('joyCanvas');
const ctx    = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CX = W/2, CY = H/2;
const OUTER_R = W/2 - 10;
const STICK_R = 36;
const MAX_DIST = OUTER_R - STICK_R;

let jx = 0, jy = 0, dragging = false;

function drawJoystick(nx, ny) {
  ctx.clearRect(0, 0, W, H);
  // 바깥 원
  ctx.beginPath();
  ctx.arc(CX, CY, OUTER_R, 0, Math.PI*2);
  ctx.fillStyle = '#0d1117';
  ctx.fill();
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.stroke();
  // 격자
  ctx.setLineDash([4,6]);
  ctx.strokeStyle = 'rgba(88,166,255,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(CX, CY-OUTER_R); ctx.lineTo(CX, CY+OUTER_R); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX-OUTER_R, CY); ctx.lineTo(CX+OUTER_R, CY); ctx.stroke();
  ctx.setLineDash([]);
  // 안쪽 원
  ctx.beginPath();
  ctx.arc(CX, CY, OUTER_R*0.5, 0, Math.PI*2);
  ctx.strokeStyle = 'rgba(88,166,255,0.12)';
  ctx.stroke();
  // 스틱
  const sx = CX + nx*MAX_DIST;
  const sy = CY - ny*MAX_DIST;
  ctx.beginPath();
  ctx.arc(sx, sy, STICK_R, 0, Math.PI*2);
  ctx.fillStyle = dragging ? '#1f6feb' : '#388bfd';
  ctx.fill();
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(sx, sy, 5, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.fill();
}

function getRelPos(e) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return { x: src.clientX - rect.left, y: src.clientY - rect.top };
}

function onDown(e) { dragging = true; onMove(e); }

function onMove(e) {
  if (!dragging) return;
  e.preventDefault();
  const p = getRelPos(e);
  let dx = p.x - CX, dy = p.y - CY;
  const dist = Math.hypot(dx, dy);
  if (dist > MAX_DIST) { dx *= MAX_DIST/dist; dy *= MAX_DIST/dist; }
  jx = +(dx/MAX_DIST).toFixed(3);
  jy = +(-(dy/MAX_DIST)).toFixed(3);
  drawJoystick(jx, jy);
  document.getElementById('joyInfo').textContent =
    'X: ' + jx.toFixed(3) + '   |   Y: ' + jy.toFixed(3);
}

function onUp() {
  dragging = false; jx = 0; jy = 0;
  drawJoystick(0, 0);
  document.getElementById('joyInfo').textContent = 'X: 0.000   |   Y: 0.000';
  api('/stop', {});
}

canvas.addEventListener('mousedown',  onDown);
window.addEventListener('mousemove',  onMove);
window.addEventListener('mouseup',    onUp);
canvas.addEventListener('touchstart', onDown, {passive:false});
canvas.addEventListener('touchmove',  onMove, {passive:false});
canvas.addEventListener('touchend',   onUp);

drawJoystick(0, 0);

// 50ms마다 조이스틱 값 전송
setInterval(function() {
  if (dragging) api('/control', {x:jx, y:jy});
}, 50);

// 100ms마다 핑 (워치독용)
setInterval(function() { api('/ping', {}); }, 100);

// ── 긴급 정지 ──
function emergencyStop() {
  dragging = false; jx = 0; jy = 0;
  drawJoystick(0, 0);
  api('/stop', {});
}

// ── 카메라 ──
let camOn = false;
function toggleCamera() {
  camOn = !camOn;
  const btn   = document.getElementById('camBtn');
  const img   = document.getElementById('camImg');
  const label = document.getElementById('camOffLabel');
  if (camOn) {
    img.src = '/video?' + Date.now();
    img.style.display = 'block';
    label.style.display = 'none';
    btn.textContent = '카메라 OFF';
    btn.classList.add('btn-on');
  } else {
    img.src = '';
    img.style.display = 'none';
    label.style.display = '';
    btn.textContent = '카메라 ON';
    btn.classList.remove('btn-on');
  }
  api('/camera/toggle', {active: camOn});
}

// ── 슬라이더 ──
let panT = null, tiltT = null;
document.getElementById('panSlider').addEventListener('input', function() {
  document.getElementById('panVal').textContent = this.value + '°';
  clearTimeout(panT);
  const v = parseInt(this.value);
  panT = setTimeout(function(){ api('/camera/pan', {value:v}); }, 80);
});
document.getElementById('tiltSlider').addEventListener('input', function() {
  document.getElementById('tiltVal').textContent = this.value + '°';
  clearTimeout(tiltT);
  const v = parseInt(this.value);
  tiltT = setTimeout(function(){ api('/camera/tilt', {value:v}); }, 80);
});

// ── TTS ──
function playTTS() {
  const text = document.getElementById('ttsText').value.trim();
  if (!text) return;
  api('/tts', {text: text});
}

// ── 멜로디 ──
function playMelody(name) {
  api('/melody', {name: name});
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return _HTML


@app.route('/status')
def status():
    return jsonify({'has_car': HAS_CAR, 'has_tts': HAS_GTTS})


@app.route('/video')
def video_stream():
    def generate():
        while True:
            if not _camera_active:
                time.sleep(0.1)
                continue
            with _frame_lock:
                frame = _latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/toggle', methods=['POST'])
def camera_toggle():
    global _camera_active
    data = request.get_json(silent=True) or {}
    _camera_active = bool(data.get('active', not _camera_active))
    return jsonify({'active': _camera_active})


@app.route('/control', methods=['POST'])
def control():
    data = request.get_json(silent=True) or {}
    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    if car:
        car.steering = x
        if y > 0.05:
            speed = int(y * 99)
            car.setSpeed(speed)
            car.forward()
        elif y < -0.05:
            speed = int(-y * 99)
            car.setSpeed(speed)
            car.backward()
        else:
            car.stop()
    return jsonify({'status': 'ok', 'x': x, 'y': y})


@app.route('/stop', methods=['POST'])
def stop_car():
    if car:
        car.stop()
        car.steering = 0
    return jsonify({'status': 'stopped'})


@app.route('/ping', methods=['POST'])
def ping():
    global _last_ping
    _last_ping = time.time()
    return jsonify({'status': 'ok'})


@app.route('/camera/pan', methods=['POST'])
def cam_pan():
    data = request.get_json(silent=True) or {}
    value = float(data.get('value', 0))
    if car:
        car.camPan(value)
    return jsonify({'status': 'ok', 'pan': value})


@app.route('/camera/tilt', methods=['POST'])
def cam_tilt():
    data = request.get_json(silent=True) or {}
    value = float(data.get('value', 0))
    if car:
        car.camTilt(value)
    return jsonify({'status': 'ok', 'tilt': value})


@app.route('/tts', methods=['POST'])
def tts_play():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '안녕하세요!')

    def _do():
        try:
            mp3 = '/tmp/_tts.mp3'
            gTTS(text, lang='ko').save(mp3)
            # mpg123 → ffplay 순서로 시도
            if subprocess.call(['which', 'mpg123'], stdout=subprocess.DEVNULL) == 0:
                subprocess.call(['mpg123', '-q', mp3])
            else:
                subprocess.call(['ffplay', '-nodisp', '-autoexit',
                                 '-loglevel', 'quiet', mp3])
        except Exception as e:
            print("[TTS] 오류: " + str(e))

    if HAS_GTTS:
        threading.Thread(target=_do, daemon=True).start()
    return jsonify({'status': 'ok'})


@app.route('/melody', methods=['POST'])
def melody_play():
    data = request.get_json(silent=True) or {}
    name = data.get('name', 'star_wars')
    threading.Thread(target=_play_melody_thread, args=(name,), daemon=True).start()
    return jsonify({'status': 'ok', 'melody': name})


# ───────────────────────────────────────────────
# 메인 실행
# ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  오토카 대시보드 시작")
    print("  AutoCar HW : " + ("연결됨" if HAS_CAR else "시뮬레이션"))
    print("  gTTS       : " + ("사용가능" if HAS_GTTS else "미설치"))
    print("=" * 50)

    threading.Thread(target=_camera_loop, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()

    print("Flask 서버: http://0.0.0.0:5000")
    print("브라우저에서 http://192.168.0.47:5000 으로 접속하세요!")
    print("종료하려면 Ctrl+C")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
