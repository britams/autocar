# 여러 파일(webcam_yolo.py, obstacle_avoidance.py 등)에서 공통으로 쓰는
# "음성으로 말하기" 기능을 한 곳에 모아둔 파일.
import subprocess
import threading
import time

last_tts_time = 0     # 마지막으로 음성을 낸 시각(초)
TTS_COOLDOWN_SEC = 3  # 같은 알림을 최소 몇 초 간격으로만 반복할지


def _play_tts(text):
    """실제로 음성을 만들어서 재생하는 부분. (시간이 좀 걸리는 작업)"""
    espeak = subprocess.Popen(
        ["espeak-ng", "-v", "ko", text, "--stdout"],
        stdout=subprocess.PIPE,
    )
    # WSL은 paplay(PulseAudio), 실제 Jetson은 aplay(ALSA)를 쓴다.
    # 둘 다 시도해보고, 먼저 성공하는 쪽을 그대로 쓰면 되지만
    # 여기서는 WSL 개발 환경 기준으로 paplay를 기본값으로 둔다.
    subprocess.run(["paplay"], stdin=espeak.stdout)


def speak(text):
    """text를 한국어 음성으로 재생한다. (메인 루프가 멈추지 않도록 별도 스레드에서 실행)"""
    global last_tts_time

    now = time.time()
    if now - last_tts_time < TTS_COOLDOWN_SEC:
        return
    last_tts_time = now

    threading.Thread(target=_play_tts, args=(text,), daemon=True).start()
