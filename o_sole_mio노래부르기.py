import numpy as np
import pyaudio
import queue
import time

class Tone:
    def __init__(self, volume=0.5, rate=48000, channels=1):
        self.volume = volume
        self.rate = rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.note_queue = queue.Queue()
        self._current = None
        self._pos = 0
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def _make_samples(self, octave, note, duration):
        if note is None:
            return np.zeros(int(self.rate * duration), dtype=np.float32)
        f = 2 ** (octave) * 55 * 2 ** (((note) - 10) / 12)
        n = int(self.rate * duration)
        sample = np.sin(2 * np.pi * np.arange(n) * f / self.rate).astype(np.float32)
        fade = min(300, n // 8)
        if fade > 0:
            env = np.ones(n, dtype=np.float32)
            env[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            env[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
            sample *= env
        return self.volume * sample

    def _callback(self, in_data, frame_count, time_info, status):
        out = np.zeros(frame_count, dtype=np.float32)
        filled = 0
        while filled < frame_count:
            if self._current is None or self._pos >= len(self._current):
                if self.note_queue.empty():
                    break
                octave, note, duration = self.note_queue.get()
                self._current = self._make_samples(octave, note, duration)
                self._pos = 0
            take = min(len(self._current) - self._pos, frame_count - filled)
            out[filled:filled + take] = self._current[self._pos:self._pos + take]
            self._pos += take
            filled += take
        return (out.tobytes(), pyaudio.paContinue)

    def play(self, octave=3, note=1, duration=2):
        self.note_queue.put((octave, note, duration))

    def is_playing(self):
        return not self.note_queue.empty() or (
            self._current is not None and self._pos < len(self._current)
        )

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# note: C=1 D=3 E=5 F=6 G=8 A=10 Bb=11 B=12
C, D, E, F, G, A, Bb, B = 1, 3, 5, 6, 8, 10, 11, 12
REST = None
S = 0.2   # 8분음표
Q = 0.4   # 4분음표
Q3 = 0.6  # 점4분음표
H = 0.8   # 2분음표

# 오 솔레미오 정확한 멜로디
o_sole_mio = [
    # 솔솔 파#미 레미파# 솔솔솔
    (3, G, S), (3, G, S),
    (3, F, S), (3, E, S),
    (3, D, S), (3, E, S), (3, F, S),
    (3, G, Q), (3, G, S), (3, REST, S),

    # 솔라 솔파#미 레레레
    (3, G, S), (3, A, S),
    (3, G, S), (3, F, S), (3, E, S),
    (3, D, H), (3, REST, Q),

    # 미미 파미레 미파솔 라라라
    (3, E, S), (3, E, S),
    (3, F, S), (3, E, S), (3, D, S),
    (3, E, S), (3, F, S), (3, G, S),
    (3, A, Q), (3, A, S), (3, REST, S),

    # 라시 라솔파# 미미미
    (3, A, S), (3, B, S),
    (3, A, S), (3, G, S), (3, F, S),
    (3, E, H), (3, REST, Q),

    # 미파솔 라라 솔파미레
    (3, E, S), (3, F, S), (3, G, S),
    (4, A, S), (4, A, S),
    (4, G, S), (4, F, S), (4, E, S), (3, D, S),

    # 도레미 파파 미레도시
    (3, C, S), (3, D, S), (3, E, S),
    (3, F, S), (3, F, S),
    (3, E, S), (3, D, S), (3, C, S), (3, B, S),

    # 라 솔---
    (3, A, Q),
    (3, G, H), (3, REST, Q),
]

tone = Tone()
for octave, note, duration in o_sole_mio:
    tone.play(octave, note, duration)

count = 0
while tone.is_playing():
    print(f"재생 중... ({count})")
    count += 1
    time.sleep(0.5)

tone.stop()
print("재생 완료")