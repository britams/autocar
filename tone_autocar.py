
import numpy as np
import pyaudio

volume = 0.5
fs = 48000
duration = 5.0
f = 440.0       # 라 음



p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
stream.write(volume * data)


class Tone:
    def __init__(self, volume=0.5, rate=48000, channels=1):
        self.volume = volume
        self.rate = rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate, output=True)

    def play(self, octave=3, note=1, duration=2):
        f = 2**(octave) * 55 * 2**(((note)-10)/12)
        sample = (np.sin(2 * np.pi * np.arange(self.rate*duration) * f/self.rate)).astype(np.float32)
        self.stream.write(self.volume * sample)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

tone = Tone()
for n in [1, 3, 5, 7, 8, 10, 12]:
    tone.play(3, n, 4)
tone.stop()


#클래스를 사용해서 재미있는 게임음악 멜로디를 출력하세요
# 스타워즈, 클래식, 기타 등등....(능동 버저 프로젝트를 참고해서 노래를 선택해서 플레이)
#논 블럭스타일로 클래스를 수정하세요.
# 논 블럭 스타일은 재생하면서 동시에 다른 작업도 가능한 방식
