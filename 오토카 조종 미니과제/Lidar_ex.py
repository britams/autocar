# vector [각도. 거리. 데이터신뢰도]
# 얼마나 데이터를 빠르게 내보내는가?
# 라이다의 속도가 1초가 몇번 회전 하는가?
# 한번 회전 할때 몇개의 데이터가 나오는가?

import time

from pop import Cds as cds
from pop import LiDAR, Util

lidar=LiDAR.Rplidar()

lidar.connect()
lidar.startMotor()
lasttime=0
dtime = time.time()
while time.time()-dtime<30:
    if time.time()-lasttime>0.03:
        data=lidar.getMap(size=(300,300))
        Util.imshow("map", data, width=600, height=600)
        lasttime=time.time()
lidar.stopMotor()

# ----

import time

from IPython.display import clear_output
from pop import Cds as cds
from pop import LiDAR, Util

lidar=LiDAR.Rplidar()

lidar.connect()
lidar.startMotor()
while True:
    try:
        vectors = lidar.getVectors()
        for v in vectors:
            print(v[0],v[1],v[2])
            clear_output()
    except KeyboardInterrupt:
        pass

# ----

import time

import numpy as np
from pop import LiDAR


def analyze_lidar(duration=10.0, quality_min=0, distance_min=0):
    """
    RPLIDAR 데이터 속도 분석 코드

    duration     : 측정 시간, 초
    quality_min  : 최소 품질값 필터
    distance_min : 최소 거리값 필터, mm
    """

    lidar = LiDAR.Rplidar()

    total_points = 0
    total_calls = 0
    rotation_count = 0

    points_in_current_rotation = 0
    points_per_rotation = []

    last_angle = None

    try:
        lidar.connect()
        lidar.startMotor()

        print("LiDAR 측정 시작")
        print(f"측정 시간: {duration}초")
        print("-" * 50)

        start_time = time.perf_counter()
        last_report_time = start_time

        while True:
            now = time.perf_counter()
            elapsed = now - start_time

            if elapsed >= duration:
                break

            vectors = lidar.getVectors()
            total_calls += 1

            if vectors is None or len(vectors) == 0:
                continue

            for v in vectors:
                angle = float(v[0])
                distance = float(v[1])
                quality = float(v[2])

                # 유효 데이터 필터
                if distance <= distance_min:
                    continue

                if quality < quality_min:
                    continue

                total_points += 1
                points_in_current_rotation += 1

                # 각도가 큰 값에서 작은 값으로 넘어가면 한 바퀴 회전했다고 판단
                # 예: 359도 -> 0도
                if last_angle is not None:
                    if last_angle > 300 and angle < 60:
                        rotation_count += 1
                        points_per_rotation.append(points_in_current_rotation)
                        points_in_current_rotation = 0

                last_angle = angle

            # 중간 출력
            if now - last_report_time >= 1.0:
                current_elapsed = now - start_time

                calls_per_sec = total_calls / current_elapsed
                points_per_sec = total_points / current_elapsed
                rotations_per_sec = rotation_count / current_elapsed
                rpm = rotations_per_sec * 60

                if len(points_per_rotation) > 0:
                    avg_points_per_rotation = np.mean(points_per_rotation)
                else:
                    avg_points_per_rotation = 0

                print(
                    f"[{current_elapsed:5.1f}s] "
                    f"호출/s={calls_per_sec:7.1f}, "
                    f"포인트/s={points_per_sec:8.1f}, "
                    f"회전/s={rotations_per_sec:5.2f}Hz, "
                    f"RPM={rpm:6.1f}, "
                    f"1회전 포인트={avg_points_per_rotation:7.1f}"
                )

                last_report_time = now

        total_elapsed = time.perf_counter() - start_time

        print("\n" + "=" * 50)
        print("최종 결과")
        print("=" * 50)

        print(f"총 측정 시간               : {total_elapsed:.2f} 초")
        print(f"getVectors() 총 호출 횟수   : {total_calls}")
        print(f"총 유효 포인트 개수         : {total_points}")
        print(f"감지한 회전 수              : {rotation_count}")

        if total_elapsed > 0:
            print(f"초당 getVectors() 호출 횟수 : {total_calls / total_elapsed:.2f} 회/s")
            print(f"초당 포인트 개수            : {total_points / total_elapsed:.2f} points/s")
            print(f"초당 회전 수                : {rotation_count / total_elapsed:.2f} Hz")
            print(f"분당 회전 수                : {(rotation_count / total_elapsed) * 60:.2f} RPM")

        if len(points_per_rotation) > 0:
            print(f"1회전당 평균 포인트 수      : {np.mean(points_per_rotation):.2f}")
            print(f"1회전당 최소 포인트 수      : {np.min(points_per_rotation)}")
            print(f"1회전당 최대 포인트 수      : {np.max(points_per_rotation)}")
            print(f"1회전당 표준편차            : {np.std(points_per_rotation):.2f}")
        else:
            print("회전이 감지되지 않았습니다.")

    except KeyboardInterrupt:
        print("\n사용자 중지")

    finally:
        try:
            lidar.stopMotor()
            print("LiDAR motor stopped")
        except Exception as e:
            print("stopMotor 처리 중 오류:", e)


def main():
    analyze_lidar(
        duration=10.0,
        quality_min=0,
        distance_min=0
    )


if __name__ == "__main__":
    main()


# ==================================================
# 최종 결과
# ==================================================
# 총 측정 시간               : 10.04 초
# getVectors() 총 호출 횟수   : 116
# 총 유효 포인트 개수         : 30043
# 감지한 회전 수              : 115
# 초당 getVectors() 호출 횟수 : 11.56 회/s
# 초당 포인트 개수            : 2992.98 points/s
# 초당 회전 수                : 11.46 Hz
# 분당 회전 수                : 687.40 RPM
# 1회전당 평균 포인트 수      : 259.10
# 1회전당 최소 포인트 수      : 239
# 1회전당 최대 포인트 수      : 347
# 1회전당 표준편차            : 19.70
# LiDAR motor stopped