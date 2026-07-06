import wave

import numpy as np
import pyaudio

TIME = 5
data = []
CHUNK = 1024
RATE = 48000

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
w = wave.open("./out.wav", "wb")
w.setnchannels(1)
w.setsampwidth(p.get_sample_size(pyaudio.paInt16))
w.setframerate(RATE)
print("녹음 시작")
try:
    while True:
        w.writeframes(stream.read(CHUNK))
except KeyboardInterrupt:
    pass

w.close()
stream.stop_stream()
stream.close()
p.terminate()

# ---

import wave

import pyaudio

p = pyaudio.PyAudio()
with wave.open("out.wav", "rb") as w:
    data = w.readframes(w.getnframes())
    stream = p.open(format=p.get_format_from_width(2), channels=1, rate=48000, output=True)
    print("재생 시작")
    stream.write(data)     # 블럭킹
    stream.stop_stream()
    print("재생 끝")
    stream.close()


import time

from pop import SoundMeter

sm = SoundMeter()

def onSoundMeter(rms, inData):
    if(rms>600):
        print(rms)

sm.setCallback(onSoundMeter)

input("input something")

sm.stop()