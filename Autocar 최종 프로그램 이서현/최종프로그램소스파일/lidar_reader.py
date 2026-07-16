# 라이다 전용 프로세스 (독립 실행)
#
# _rplidar.so가 카메라/YOLO(GPU)와 같은 프로세스 안에서 동시에 활동하면
# 가끔 프로세스 전체가 죽는 문제가 있었음(원인 불명, 자원 경합으로 추정).
# 그래서 라이다는 아예 별도의 OS 프로세스로 완전히 분리한다.
# -> 여기서 죽어도 카메라/YOLO/폰조종 메인 프로그램은 영향을 받지 않음.
#
# [2026-07-15] 별도 프로세스로 분리해도, 그 프로세스 자체가 connect() 안에서
# 멈춰버리는(응답도 없고 에러도 안 나는) 경우가 실제로 있었음 -> 이 파일 안에서
# 다시 "감시자(main) - 실제 작업(worker)" 구조로 한 겹 더 분리해서, 연결이
# CONNECT_TIMEOUT_SEC 안에 안 되면 감시자가 강제로 죽이고 재시도하도록 함.
#
# 좌/우 근접 장애물 여부를 계속 계산해서 /tmp/lidar_state.json 파일에 기록만 한다.
# 메인 프로그램은 이 파일을 주기적으로 읽기만 하면 됨.

import json
import time
import multiprocessing

LIDAR_RANGE_MM = 1300
LIDAR_FRONT_DEG = 45
LIDAR_MIN_POINTS = 3

STATE_FILE = "/tmp/lidar_state.json"
CONNECT_TIMEOUT_SEC = 10  # 이 시간 안에 연결 성공(connected:true 기록)이 안 되면 강제 종료 후 재시도


def write_state(left, right, connected):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({"left": left, "right": right, "connected": connected, "ts": time.time()}, f)
    except Exception:
        pass


def worker():
    """실제 라이다 연결 + 읽기 루프. 여기서 멈추더라도 바깥의 감시자(main)가 강제 종료함."""
    try:
        from pop import Lidar
    except ImportError:
        from pop import LiDAR as Lidar

    lidar = Lidar.Rplidar()
    lidar.connect()
    lidar.startMotor()
    print("[라이다 프로세스] 연결 성공")

    while True:
        try:
            vectors = lidar.getVectors()
            left_points = 0
            right_points = 0
            if vectors is not None:
                for v in vectors:
                    if len(v) >= 2 and 50 < v[1] < LIDAR_RANGE_MM:
                        if 360 - LIDAR_FRONT_DEG <= v[0] <= 360:
                            left_points += 1
                        elif 0 <= v[0] <= LIDAR_FRONT_DEG:
                            right_points += 1
            write_state(left_points > LIDAR_MIN_POINTS, right_points > LIDAR_MIN_POINTS, True)
        except Exception:
            pass
        time.sleep(0.05)


def main():
    """감시자: worker를 자식 프로세스로 띄우고, 연결이 너무 오래 걸리거나 도중에
    죽으면 강제로 죽인 뒤 다시 새 프로세스로 재시도한다 (무한 재시도)."""
    write_state(False, False, False)

    while True:
        proc = multiprocessing.Process(target=worker, daemon=True)
        proc.start()

        deadline = time.time() + CONNECT_TIMEOUT_SEC
        connected_ok = False
        while time.time() < deadline and proc.is_alive():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                if data.get('connected'):
                    connected_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not connected_ok:
            print("[라이다 프로세스] 연결 타임아웃/실패 -> 강제 종료 후 재시도")
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join()
            write_state(False, False, False)
            time.sleep(1)
            continue

        # 연결 성공한 뒤 worker가 (예상치 못하게) 죽으면 여기로 빠져나오고, 다시 재시도한다.
        proc.join()
        print("[라이다 프로세스] worker 종료됨 -> 재시도")
        write_state(False, False, False)
        time.sleep(1)


if __name__ == '__main__':
    main()
