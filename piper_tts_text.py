# pip install piper-tts
# wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
# wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
# echo "Hello, this is Piper text to speech." | \
# =============================================================
# [원본 - 강사님 메모] (이 환경에서 안 되는 이유)
# =============================================================
# pip install piper-tts                 ← Python 3.7+ 필요 (현재 3.6.9)
# Piper 바이너리                         ← GLIBC 2.29+ 필요 (현재 2.27)
# piper-phonemize                       ← aarch64 + Python 3.6 휠 없음
#
# =============================================================
# [수정본] espeak-ng 사용
# =============================================================
# 설치:
#   sudo apt update
#   sudo apt install espeak-ng
#
# 터미널에서 바로 사용:
#   espeak-ng "Hello, this is a test" --stdout > output.wav
#   aplay output.wav

# Python 코드에서 사용하기
import subprocess

# 영어 TTS
text = "Hello, this is Piper text to speech."
subprocess.run(
    ["espeak-ng", text, "--stdout"],
    stdout=open("output.wav", "wb")
)

# 한국어 TTS
text_ko = "안녕하세요, 음성 합성 테스트입니다."
subprocess.run(
    ["espeak-ng", "-v", "ko", text_ko, "--stdout"],
    stdout=open("output_ko.wav", "wb")
)

# 재생
subprocess.run(["aplay", "output.wav"])