# -*- coding: utf-8 -*-
"""
라이다 거리값 확인/캘리브레이션용 디버그 스크립트

사용법:
1. 라이다 정면(0도 방향)에 자로 정확한 거리(예: 50cm)에 물체(책, 벽 등)를 놓는다.
2. python3 lidar_debug.py 실행
3. 화면에 찍히는 "정면 거리" 숫자와 실제로 잰 거리를 비교한다.
   - 실제 500mm인데 화면에 2000mm 나오면 -> 4배 (RPLIDAR raw q2 단위 의심)
   - 실제 500mm인데 화면에 5000mm 나오면 -> 10배 (cm를 mm로 안 바꾼 것 의심)
   - 실제 500mm인데 화면에 50 나오면 -> 이미 cm 단위로 주는 것 의심
종료: Ctrl+C
"""

import sys
import os
import time

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.getcwd())

from pop.LiDAR import Rplidar

lidar = Rplidar()

try:
    lidar.connect()
except TypeError:
    connected = False
    for port in ('/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0'):
        try:
            lidar.connect(port)
            connected = True
            break
        except Exception:
            continue
    if not connected:
        raise RuntimeError('라이다 포트를 찾지 못했습니다.')

lidar.startMotor()
print('[LiDAR] 연결 성공, 1초 후 측정 시작...')
time.sleep(1)

try:
    while True:
        samples = lidar.getSamples(filter_quality=True)

        rows = []
        for s in samples:
            angle = float(s.angle)
            distance = float(s.distance)
            quality = float(s.quality)
            angle_signed = angle if angle <= 180 else angle - 360
            rows.append((angle_signed, distance, quality))

        if not rows:
            print('샘플 없음 (라이다에서 값을 못 받는 중)')
        else:
            rows.sort(key=lambda r: abs(r[0]))  # 0도(정면)에 가까운 순서로 정렬
            distances = [r[1] for r in rows]

            print('-' * 70)
            print('총 샘플 수: {}개  |  전체 거리 범위: {:.1f} ~ {:.1f}'.format(
                len(rows), min(distances), max(distances)))
            print('정면(0도)에 가장 가까운 샘플 5개  (각도, 원시distance값, 강도):')
            for angle_signed, distance, quality in rows[:5]:
                print('  각도 {:6.1f}도   raw distance = {:9.2f}   강도 {:.0f}'.format(
                    angle_signed, distance, quality))
            print('※ 위 raw distance 값과 실제 자로 잰 거리(mm)를 비교해서 알려주세요.')

        time.sleep(0.7)

except KeyboardInterrupt:
    print('\n종료합니다.')
finally:
    try:
        lidar.stopMotor()
    except Exception:
        pass
