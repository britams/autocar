import threading
import time

import numpy as np
from pop import LiDAR, Pilot


class FrontDistanceKeeper:
    def __init__(
        self,
        target_distance=700,      # 유지할 거리, mm
        tolerance=80,             # 허용 오차, mm
        front_range=10,           # 정면 기준 ±10도
        quality_min=10,
        min_valid_distance=100,
        max_valid_distance=3000,
        base_speed=25,
        max_speed=50,
        kp=0.04
    ):
        self.target_distance = target_distance
        self.tolerance = tolerance
        self.front_range = front_range
        self.quality_min = quality_min
        self.min_valid_distance = min_valid_distance
        self.max_valid_distance = max_valid_distance
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.kp = kp

        self.lidar = LiDAR.Rplidar()
        self.car = Pilot.AutoCar()

        # 0도 ~ 359도까지 1도 간격 dictionary
        self.angle_dict = {degree: [] for degree in range(360)}

        self.front_distance = None
        self.front_count = 0

        self.lock = threading.Lock()
        self.running = threading.Event()

    def connect(self):
        self.lidar.connect()
        self.lidar.startMotor()

    def start(self):
        self.running.set()

        self.update_thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.move_thread = threading.Thread(
            target=self.move,
            daemon=True
        )

        self.update_thread.start()
        self.move_thread.start()

    def stop(self):
        self.running.clear()

        try:
            self.car.stop()
        except Exception as e:
            print("car.stop() 오류:", e)

        try:
            self.lidar.stopMotor()
        except Exception as e:
            print("lidar.stopMotor() 오류:", e)

    def clear_angle_dict(self):
        for degree in range(360):
            self.angle_dict[degree].clear()

    def angle_to_degree(self, angle):
        """
        실수 각도를 1도 단위 정수 각도로 변환
        예:
        0.2  -> 0
        0.7  -> 1
        359.6 -> 0 으로 넘어갈 수 있으므로 % 360 처리
        """
        return int(round(angle)) % 360

    def get_front_degrees(self):
        """
        정면을 0도라고 가정한다.
        front_range=10이면
        350, 351, ..., 359, 0, 1, ..., 10
        """
        degrees = []

        for d in range(-self.front_range, self.front_range + 1):
            degrees.append(d % 360)

        return degrees

    def calc_front_average(self):
        """
        angle_dict에서 정면 ±10도 데이터만 모아서 평균 거리 계산
        """
        front_distances = []

        for degree in self.get_front_degrees():
            front_distances.extend(self.angle_dict[degree])

        if len(front_distances) == 0:
            return None, 0

        return float(np.mean(front_distances)), len(front_distances)

    def update(self):
        """
        LiDAR 데이터를 계속 읽어서
        1도 단위 dictionary에 저장하고,
        정면 ±10도 평균 거리를 계산한다.
        """
        while self.running.is_set():
            try:
                vectors = self.lidar.getVectors()

                # 이번 스캔 데이터를 새로 저장
                self.clear_angle_dict()

                for angle, distance, quality in vectors:
                    angle = float(angle)
                    distance = float(distance)
                    quality = float(quality)

                    if quality < self.quality_min:
                        continue

                    if distance < self.min_valid_distance:
                        continue

                    if distance > self.max_valid_distance:
                        continue

                    degree = self.angle_to_degree(angle)

                    # 1도 단위 dictionary에 거리 저장
                    self.angle_dict[degree].append(distance)

                front_avg, front_count = self.calc_front_average()

                with self.lock:
                    self.front_distance = front_avg
                    self.front_count = front_count

                time.sleep(0.01)

            except Exception as e:
                print("update 오류:", e)
                time.sleep(0.1)

    def calc_speed(self, error):
        speed = self.base_speed + abs(error) * self.kp
        speed = int(min(speed, self.max_speed))
        return speed

    def move(self):
        """
        정면 평균 거리 기준으로 자동차 이동
        """
        while self.running.is_set():
            with self.lock:
                distance = self.front_distance
                count = self.front_count

            if distance is None:
                self.car.stop()
                print("정면 데이터 없음 -> 정지")
                time.sleep(0.1)
                continue

            error = distance - self.target_distance
            speed = self.calc_speed(error)

            if abs(error) <= self.tolerance:
                self.car.stop()
                state = "STOP"

            elif error > 0:
                # 현재 거리가 목표보다 멀다 -> 전진
                self.car.forward(speed)
                state = "FORWARD"

            else:
                # 현재 거리가 목표보다 가깝다 -> 후진
                self.car.backward(speed)
                state = "BACKWARD"

            print(
                f"state={state:8s} | "
                f"front_avg={distance:7.1f} mm | "
                f"target={self.target_distance:4d} mm | "
                f"error={error:7.1f} | "
                f"speed={speed:3d} | "
                f"front_points={count}"
            )

            time.sleep(0.05)


def main():
    robot = FrontDistanceKeeper(
        target_distance=700,   # 70cm 유지
        tolerance=80,          # ±8cm 이내면 정지
        front_range=10,        # 정면 ±10도
        quality_min=10,
        min_valid_distance=100,
        max_valid_distance=3000,
        base_speed=40,
        max_speed=65,
        kp=0.04
    )

    try:
        robot.connect()
        robot.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n사용자 중지")

    finally:
        robot.stop()
        print("프로그램 종료")


if __name__ == "__main__":
    main()