# 오토카 시동/종료 런처
#
# phone_remote_control.py(카메라+YOLO+라이다, 무거움)를 폰 버튼으로 켜고 끄기 위한
# 아주 가벼운 별도 서버. YOLO/pycuda를 아예 안 불러오기 때문에 부팅하자마자
# 즉시 켜져서 항상 대기할 수 있음 - 이 런처를 tmux 세션으로 상시 띄워두면 됨.
#
# tmux 세션 이름 "autocar" 안에서 phone_remote_control.py를 시작/종료시킨다.
#
# 사용법: python3 launcher.py  (포트 5001)
#   폰 브라우저에서 http://<오토카IP>:5001 접속 -> 시동/종료 버튼
#   시동 후에는 http://<오토카IP>:5000/remote 로 실제 조종 화면 이동

import subprocess
from flask import Flask, jsonify, redirect, request

app = Flask(__name__)

TMUX_SESSION = "autocar"
RUN_DIR = "/home/soda/tensorrt_demos"
RUN_CMD = f"cd {RUN_DIR} && python3 -u phone_remote_control.py > phone_log.txt 2>&1"


def is_running():
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


@app.route('/')
def index():
    running = is_running()
    status_text = "실행 중" if running else "꺼짐"
    status_color = "#28a745" if running else "#888"
    return f"""
    <html><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: white; padding: 30px; }}
        h2 {{ color: #00ffcc; }}
        .status {{ font-size: 20px; font-weight: bold; color: {status_color}; margin: 20px 0; }}
        .btn {{ width: 200px; height: 60px; font-size: 18px; margin: 10px; border-radius: 12px; border: none; font-weight: bold; }}
        .start {{ background: #28a745; color: white; }}
        .stop {{ background: #dc3545; color: white; }}
        .goto {{ background: #007bff; color: white; }}
    </style></head>
    <body>
        <h2>오토카 시동 런처</h2>
        <div class="status">현재 상태: {status_text}</div>
        <button class="btn start" onclick="fetch('/start').then(()=>location.reload())">시동 걸기</button><br>
        <button class="btn stop" onclick="fetch('/stop').then(()=>location.reload())">종료</button><br>
        <button class="btn goto" onclick="location.href='/remote'">조종 화면으로 이동</button>
    </body></html>
    """


@app.route('/remote')
def go_remote():
    host_only = request.host.split(':')[0]
    return redirect(f'http://{host_only}:5000/remote')


@app.route('/start')
def start():
    if not is_running():
        subprocess.Popen(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-n", "phone_yolo", RUN_CMD])
        return jsonify({"result": "starting"})
    return jsonify({"result": "already_running"})


@app.route('/stop')
def stop():
    if is_running():
        # 먼저 차를 안전하게 세우고(가능하면), tmux 세션을 종료한다.
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:5000/api/kill", timeout=2)
        except Exception:
            pass
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION])
        return jsonify({"result": "stopped"})
    return jsonify({"result": "already_stopped"})


@app.route('/status')
def status():
    return jsonify({"running": is_running()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
