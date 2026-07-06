# pip install gTTS
import subprocess

from gtts import gTTS

text = "And the saddest thing. under the sun above"
filename = "en_tts.mp3"

tts= gTTS(text)
tts.save(filename)

from pathlib import Path


def convert_mp3_to_wav(mp3_path):
    mp3_path = Path(mp3_path)
    wav_path = mp3_path.with_suffix(".wav")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(mp3_path),
        "-ar", "44100",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(wav_path)
    ], check=True)

    return wav_path

audio_path = Path("en_tts.mp3")

if audio_path.suffix.lower() == ".mp3":
    print("mp3 파일입니다. wav로 변환합니다.")
    play_path = convert_mp3_to_wav(audio_path)

elif audio_path.suffix.lower() == ".wav":
    print("wav 파일입니다. 바로 재생합니다.")
    play_path = audio_path