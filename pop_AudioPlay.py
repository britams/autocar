import time

from pop import AudioPlay, AudioRecord

with AudioRecord("my_record.wav") as record:
    record.run()
    print("start Recording...")

    for _ in range(5):
        time.sleep(1)

    record.stop()
    print("Stop Recoding...")

with AudioPlay("my_record.wav", False, True) as play:
    play.run()
    print("start playing...")

    for _ in range(12):
        time.sleep(1)

    play.stop()
    print("Stop playing...")