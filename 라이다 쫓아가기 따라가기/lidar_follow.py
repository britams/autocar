# -*- coding: utf-8 -*-
"""
라이다로 정면 장애물을 피해서 달리는 Auto car
- 장애물이 없으면 전진, 정면 800mm 안에 장애물이 잡히면 후진 (가까울수록 빠르게)
실행: python3 lidar_follow.py
종료: Ctrl+C
"""

import sys
import os
import time

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.getcwd())

FRONT_ANGLE_DEG = 45     # 정면으로 인정할 좌우 각도 범위
OBSTACLE_RANGE_MM = 800  # 이 거리 안이면 장애물로 인식
DEFAULT_SPEED = 30       # 장애물 없을 때 기본 전진 속도
MIN_SPEED = 20
MAX_SPEED = 80

# 이 차량은 배선이 반대라서 car.forward()가 실제로는 뒤로, car.backward()가 실제로는 앞으로 간다.
FORWARD_IS_REVERSED = True


def drive_forward(car, speed):
    (car.backward if FORWARD_IS_REVERSED else car.forward)(speed)


def drive_backward(car, speed):
    (car.forward if FORWARD_IS_REVERSED else car.backward)(speed)


def connect_lidar():
    from pop.LiDAR import Rplidar
    lidar = Rplidar()
    try:
        lidar.connect()
    except TypeError:
        for port in ('/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0'):
            try:
                lidar.connect(port)
                break
            except Exception:
                continue
        else:
            raise RuntimeError('라이다 포트를 찾지 못했습니다.')
    lidar.startMotor()
    return lidar


def connect_car():
    import builtins as _builtins
    import numpy as _np
    _builtins.np = _np
    try:
        from pop.Pilot import AutoCar
        return AutoCar()
    finally:
        delattr(_builtins, 'np')


def find_front_target(samples):
    """정면(±FRONT_ANGLE_DEG) 범위에서 가장 가까운 물체를 찾는다."""
    best = None
    for s in samples:
        distance = float(s.distance)
        if distance <= 0 or distance > OBSTACLE_RANGE_MM:
            continue

        angle = float(s.angle)
        angle_signed = angle if angle <= 180 else angle - 360
        if abs(angle_signed) > FRONT_ANGLE_DEG:
            continue

        if best is None or distance < best[1]:
            best = (angle_signed, distance)
    return best


def main():
    lidar = connect_lidar()
    print('[LiDAR] 연결 성공')
    car = connect_car()
    print('[AutoCar] 연결 성공')

    try:
        while True:
            t0 = time.time()
            samples = lidar.getSamples(filter_quality=True)
            t1 = time.time()
            target = find_front_target(samples)
            t2 = time.time()

            if target is None:
                car.steering = 0
                drive_forward(car, DEFAULT_SPEED)
                tag = '장애물 없음 -> 전진'
            else:
                angle_signed, distance = target
                car.steering = max(-1.0, min(1.0, angle_signed / FRONT_ANGLE_DEG))
                speed = MIN_SPEED + (1 - distance / OBSTACLE_RANGE_MM) * (MAX_SPEED - MIN_SPEED)
                drive_backward(car, speed)
                tag = '장애물 감지({:.0f}mm, {:.1f}도) -> 후진(속도 {:.0f})'.format(
                    distance, angle_signed, speed)

            print('{}  [샘플 {}개, getSamples {:.0f}ms, 계산 {:.0f}ms, 총 루프 {:.0f}ms]'.format(
                tag, len(samples), (t1 - t0) * 1000, (t2 - t1) * 1000, (time.time() - t0) * 1000))
    except KeyboardInterrupt:
        print('\n종료합니다.')
    finally:
        car.stop()
        car.steering = 0
        try:
            lidar.stopMotor()
        except Exception:
            pass


if __name__ == '__main__':
    main()
