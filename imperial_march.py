import numpy as np
import pyaudio
import queue
import time


class Tone:
    """기존 Tone 클래스를 논블럭 방식(콜백 스트림)으로 수정한 버전.
    play()는 큐에 음표만 등록하고 바로 리턴되며,
    실제 재생은 백그라운드 콜백 스레드에서 처리된다."""

    def __init__(self, volume=0.5, rate=48000, channels=1):
        self.volume = volume
        self.rate = rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.note_queue = queue.Queue()
        self._current = None   # 현재 재생 중인 샘플 배열
        self._pos = 0           # 현재 샘플 내 재생 위치

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            output=True,
            stream_callback=self._callback,
        )

    def _make_samples(self, octave, note, duration):
        if note is None:  # 쉼표
            return np.zeros(int(self.rate * duration), dtype=np.float32)

        f = 2 ** (octave) * 55 * 2 ** (((note) - 10) / 12)
        n = int(self.rate * duration)
        sample = np.sin(2 * np.pi * np.arange(n) * f / self.rate).astype(np.float32)

        # 음표 경계에서 딸깍 소리(클릭 노이즈)가 나지 않도록 짧게 페이드 인/아웃
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

            remain_note = len(self._current) - self._pos
            remain_out = frame_count - filled
            take = min(remain_note, remain_out)

            out[filled:filled + take] = self._current[self._pos:self._pos + take]
            self._pos += take
            filled += take

        return (out.tobytes(), pyaudio.paContinue)

    def play(self, octave=3, note=1, duration=2):
        """기존과 동일한 시그니처. 큐에 넣기만 하므로 즉시 리턴(논블럭)."""
        self.note_queue.put((octave, note, duration))

    def is_playing(self):
        return not self.note_queue.empty() or (
            self._current is not None and self._pos < len(self._current)
        )

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# 스타워즈 - 제국 행진곡(Imperial March) 도입부
# (octave, note, duration) : note 번호는 기존 코드의 12음계 기준(10=라)
Q, E, S, H = 0.45, 0.225, 0.1, 0.9  # 4분음표/8분음표/짧은꾸밈음/2분음표 길이(초)
REST = (0, None, 0.1)

imperial_march = [
    (3, 10, Q), (3, 10, Q), (3, 10, Q),
    (3, 7, E), (3, 14, S),
    (3, 10, Q), (3, 7, E), (3, 14, S),
    (3, 10, H),

    (4, 5, Q), (4, 5, Q), (4, 5, Q),
    (4, 7, E), (3, 14, S),
    (3, 6, Q), (3, 7, E), (3, 14, S),
    (3, 10, H),
]


if __name__ == "__main__":
    tone = Tone()

    for octave, note, duration in imperial_march:
        tone.play(octave, note, duration)

    # 논블럭 확인용: 노래가 재생되는 동안 메인 스레드는 다른 작업을 계속할 수 있다
    count = 0
    while tone.is_playing():
        print(f"노래 재생 중... 다른 작업 수행 가능 ({count})")
        count += 1
        time.sleep(0.5)

    tone.stop()
    print("재생 완료")
