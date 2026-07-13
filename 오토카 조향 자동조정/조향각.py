import json          # 보정 결과(테이블)를 파일로 저장/불러오기 위한 라이브러리
import os             # json 저장 폴더 경로 생성/확인을 위한 라이브러리
import time           # 대기(sleep) 시간 처리를 위한 라이브러리

import numpy as np    # 배열 연산(선형 구간 생성, 정렬, 보간 등)을 위한 라이브러리
from pop import Pilot  # 오토카(AutoCar)를 제어하기 위한 라이브러리

Car = Pilot.AutoCar()  # 오토카 객체 생성 (조향/전진/후진/센서읽기 제어에 사용)


# =========================
# 안전 설정값
# =========================

SPEED = 70          # 테스트 주행 속도 (기존 70보다 낮게 시작 권장 -> 안전하게 저속 보정)
MOVE_TIME = 0.35    # 한 번 측정할 때 앞으로 움직이는 시간(초)
BACK_TIME = 0.35    # 측정 후 원위치로 복귀할 때 뒤로 움직이는 시간(초)
STABLE_TIME = 0.15  # 정지 후 진동/흔들림이 가라앉을 때까지 대기하는 시간(초)

GYRO_COUNT = 8      # gyro 값을 평균낼 때 몇 번 샘플링할지 (get_gyro_avg에서 사용)
GYRO_DELAY = 0.03   # gyro 샘플링 사이의 간격(초)

# =========================
# json 저장 위치
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # 이 스크립트 파일이 있는 폴더 경로
JSON_DIR = os.path.join(BASE_DIR, "json 파일저장")            # json 파일들을 모아둘 하위 폴더
CALIBRATION_PATH = os.path.join(JSON_DIR, "steering_calibration.json")  # 조향 보정 결과 저장 경로


def car_forward(car, speed):
    # 오토카 라이브러리 버전에 따라 forward()가 속도 인자를 받을 수도, 안 받을 수도 있어
    # 두 경우 모두 대응하기 위한 안전한 전진 함수
    try:
        car.forward(speed)   # 속도(speed)를 지정해서 전진 시도
    except TypeError:
        car.forward()        # speed 인자를 지원하지 않는 버전이면 인자 없이 전진


def car_backward(car, speed):
    # car_forward와 동일한 이유로, 후진도 속도 인자 유무에 안전하게 대응
    try:
        car.backward(speed)  # 속도(speed)를 지정해서 후진 시도
    except TypeError:
        car.backward()       # speed 인자를 지원하지 않는 버전이면 인자 없이 후진


def get_gyro_avg(car, count=8, delay=0.03):
    # gyro 값을 한 번만 읽으면 순간 노이즈(잡음)에 흔들릴 수 있으므로
    # 여러 번 읽어 평균을 내서 값을 안정시키는 함수
    values = []  # 측정한 gyro 값들을 담을 리스트

    for _ in range(count):        # count번 반복해서 gyro 값을 샘플링
        z = car.getGyro("z")      # 자이로 z축(회전) 값 읽기
        values.append(float(z))   # 읽은 값을 리스트에 추가 (float로 형 변환)
        time.sleep(delay)         # 다음 샘플링 전까지 잠깐 대기

    return sum(values) / len(values)  # 샘플들의 평균값 반환


def move_forward_and_measure(car, steer, speed, move_time):
    """
    짧게 앞으로 이동하면서 gyro 평균을 측정
    """
    car.steering = steer          # 조향값을 steer로 설정 (이 각도로 주행)
    car_forward(car, speed)       # 지정한 속도로 전진 시작

    start_time = time.time()      # 이동을 시작한 시각 기록
    gyro_values = []              # 이동하는 동안 측정한 gyro 값들을 담을 리스트

    while time.time() - start_time < move_time:
        # move_time(초) 동안 계속 반복하면서 gyro 값을 측정
        z = car.getGyro("z")           # 현재 자이로 z축 값 읽기
        gyro_values.append(float(z))   # 측정값을 리스트에 추가
        time.sleep(GYRO_DELAY)         # 다음 측정까지 짧게 대기

    car.stop()               # 이동 시간이 다 되면 정지
    time.sleep(STABLE_TIME)  # 정지 후 흔들림이 가라앉을 때까지 대기

    if len(gyro_values) == 0:
        return 0.0            # 측정된 값이 하나도 없으면(예외 상황) 0.0 반환

    return sum(gyro_values) / len(gyro_values)  # 이동 중 측정한 gyro 값들의 평균 반환


def move_backward_return(car, steer, speed, back_time):
    """
    같은 steering 값으로 뒤로 복귀
    같은 steering을 유지해야 앞으로 간 곡선 경로를 되돌아오기 쉽다.
    """
    car.steering = steer       # 앞으로 갈 때와 동일한 조향값 유지 (같은 경로로 되돌아오기 위함)
    car_backward(car, speed)   # 지정한 속도로 후진 시작

    time.sleep(back_time)      # back_time(초) 동안 후진 유지

    car.stop()               # 후진 종료 후 정지
    time.sleep(STABLE_TIME)  # 정지 후 흔들림이 가라앉을 때까지 대기


def collect_steering_data(car):
    # 여러 조향값에 대해 실제로 주행시켜보고, 그때의 gyro 평균값을 모아 데이터셋을 만드는 함수
    dataset = {
        "steer": [],   # 테스트에 사용한 조향값들을 저장할 리스트
        "gyro": []     # 각 조향값에서 측정된 gyro 평균값을 저장할 리스트
    }

    car.setSpeed(SPEED)   # 오토카의 기본 속도를 SPEED로 설정

    steer_values = np.linspace(-1.0, 1.0, 11)
    # -1.0(최대 좌회전) ~ 1.0(최대 우회전) 구간을 11개로 균등하게 나눈 값 생성
    # 예: -1.0, -0.8, -0.6, ..., 0.8, 1.0

    try:
        for steer in steer_values:           # 생성된 조향값들을 하나씩 꺼내서 테스트
            steer = round(float(steer), 1)   # 부동소수점 오차 제거 (예: 0.1999998 -> 0.2)

            print("test steer:", steer)      # 현재 테스트 중인 조향값 출력 (진행 확인용)

            # 1. 앞으로 짧게 이동하면서 gyro 측정
            gyro_avg = move_forward_and_measure(
                car=car,
                steer=steer,
                speed=SPEED,
                move_time=MOVE_TIME
            )  # 이 조향값으로 전진했을 때의 평균 gyro 값을 측정

            # 2. 같은 steering으로 뒤로 복귀
            move_backward_return(
                car=car,
                steer=steer,
                speed=SPEED,
                back_time=BACK_TIME
            )  # 다음 측정을 위해 제자리로 복귀

            dataset["steer"].append(steer)       # 사용한 조향값 기록
            dataset["gyro"].append(gyro_avg)     # 측정된 gyro 평균값 기록

            print({
                "steer": steer,
                "gyro": gyro_avg
            })  # 이번 측정 결과 출력 (진행 확인용)

            time.sleep(0.3)   # 다음 조향값 테스트 전에 잠깐 대기 (안정성 확보)

    finally:
        car.stop()   # 반복 도중 오류가 나더라도 반드시 차량을 정지시킴 (안전장치)

    return dataset   # 수집된 {steer, gyro} 데이터셋 반환


class SteeringCalibrator:
    # 수집된 (steer, gyro) 데이터를 바탕으로
    # "원하는 steer 값 -> 실제로 넣어야 할 보정된 steer 값"을 계산해주는 클래스
    # (AI 학습 모델 대신, 데이터를 정렬하고 보간(interpolate)하는 방식으로 보정값을 구함)

    def __init__(self):
        self.steer_table = []   # 정렬된 steer 값들을 저장할 리스트
        self.gyro_table = []    # 정렬된 steer에 대응하는 gyro 값들을 저장할 리스트

        self.base_gyro = 0.0        # steer=0 근처일 때의 기준 gyro 값
        self.zero_correction = 0.0  # 실제로 차량이 직진하게 만드는 보정된 steer 값 (중앙값 보정)

        self.usable_gyro = 0.0  # 좌/우 양쪽에서 공통으로 표현 가능한 최대 회전량(gyro 변화량)

        self.left_steer = []   # 중앙 보정값보다 왼쪽(음수 방향) 구간의 steer 값들
        self.left_delta = []   # 그 구간에서의 gyro 변화량(기준 대비 차이)

        self.right_steer = []  # 중앙 보정값보다 오른쪽(양수 방향) 구간의 steer 값들
        self.right_delta = []  # 그 구간에서의 gyro 변화량(기준 대비 차이)

    def fit(self, steer_list, gyro_list):
        # 수집된 데이터를 바탕으로 보정 테이블을 계산(학습에 해당하는 과정)
        steer_arr = np.array(steer_list, dtype=float)  # steer 리스트를 numpy 배열로 변환
        gyro_arr = np.array(gyro_list, dtype=float)     # gyro 리스트를 numpy 배열로 변환

        order = np.argsort(steer_arr)   # steer 값 기준으로 오름차순 정렬했을 때의 인덱스 순서
        steer_arr = steer_arr[order]    # steer 배열을 오름차순으로 정렬
        gyro_arr = gyro_arr[order]      # gyro 배열도 steer와 같은 순서로 재배열 (짝을 맞추기 위함)

        self.steer_table = steer_arr.tolist()  # 정렬된 steer 값들을 리스트로 저장 (나중에 저장/조회용)
        self.gyro_table = gyro_arr.tolist()    # 정렬된 gyro 값들을 리스트로 저장

        # steer=0에 가장 가까운 값의 gyro를 직진 기준으로 사용
        zero_idx = np.argmin(np.abs(steer_arr))  # steer 값이 0에 가장 가까운 데이터의 인덱스
        self.base_gyro = float(gyro_arr[zero_idx])  # 그 지점의 gyro 값을 "직진 기준값"으로 설정

        # 기준 gyro 대비 변화량
        delta_arr = gyro_arr - self.base_gyro
        # 각 지점의 gyro 값에서 기준값(base_gyro)을 빼서, 회전량 변화(delta)만 남김

        # 실제로 가장 직진에 가까운 steering 값
        zero_correction_idx = np.argmin(np.abs(delta_arr))
        # delta(회전량 변화)가 0에 가장 가까운 지점 = 실제로 차가 거의 직진했던 지점
        self.zero_correction = float(steer_arr[zero_correction_idx])
        # 그 지점의 steer 값을 "진짜 중앙(직진) 조향값"으로 채택 -> 이것이 조향 오차 보정값

        left_mask = steer_arr < self.zero_correction   # 보정된 중앙값보다 작은(왼쪽) 데이터 위치 표시
        right_mask = steer_arr > self.zero_correction  # 보정된 중앙값보다 큰(오른쪽) 데이터 위치 표시

        left_steer = steer_arr[left_mask]   # 왼쪽 구간의 steer 값들만 추출
        left_delta = delta_arr[left_mask]   # 왼쪽 구간의 gyro 변화량들만 추출

        right_steer = steer_arr[right_mask]  # 오른쪽 구간의 steer 값들만 추출
        right_delta = delta_arr[right_mask]  # 오른쪽 구간의 gyro 변화량들만 추출

        if len(left_steer) < 2 or len(right_steer) < 2:
            # 보간(interpolation)을 하려면 양쪽 구간에 최소 2개 이상의 데이터가 필요
            raise ValueError("왼쪽/오른쪽 조향 데이터가 부족합니다.")

        self.left_steer = left_steer.tolist()   # 왼쪽 구간 steer 값들을 클래스 변수에 저장
        self.left_delta = left_delta.tolist()   # 왼쪽 구간 gyro 변화량들을 클래스 변수에 저장

        self.right_steer = right_steer.tolist()  # 오른쪽 구간 steer 값들을 클래스 변수에 저장
        self.right_delta = right_delta.tolist()  # 오른쪽 구간 gyro 변화량들을 클래스 변수에 저장

        left_max = max(abs(v) for v in self.left_delta)   # 왼쪽에서 낼 수 있는 최대 회전량(절댓값)
        right_max = max(abs(v) for v in self.right_delta)  # 오른쪽에서 낼 수 있는 최대 회전량(절댓값)

        # 좌우 모두 표현 가능한 공통 회전량
        self.usable_gyro = float(min(left_max, right_max))
        # 좌/우 최대 회전량 중 더 작은 쪽에 맞춰야 양쪽 모두 대칭적으로 조향값을 매핑할 수 있음

        print("===== Calibration Result =====")
        print("base_gyro       :", self.base_gyro)        # steer=0 부근의 기준 gyro 값
        print("zero_correction :", self.zero_correction)  # 실제 직진을 위한 보정된 중앙 steer 값
        print("left_max_delta  :", left_max)               # 왼쪽 최대 회전량
        print("right_max_delta :", right_max)              # 오른쪽 최대 회전량
        print("usable_gyro     :", self.usable_gyro)       # 좌우 공통으로 쓸 수 있는 회전량

    def steering(self, desired_steer):
        """
        사용자가 원하는 steer -1.0 ~ 1.0을 넣으면
        실제 Car.steering에 넣을 보정값을 반환한다.
        """

        desired_steer = float(desired_steer)                    # 입력값을 float로 변환
        desired_steer = max(-1.0, min(1.0, desired_steer))      # 값을 -1.0~1.0 범위로 제한(clip)

        if abs(desired_steer) < 1e-6:
            # 원하는 값이 사실상 0(직진)이면, 계산된 보정 중앙값을 그대로 반환
            return self.zero_correction

        target_delta_abs = abs(desired_steer) * self.usable_gyro
        # 원하는 조향 강도(0~1 비율)를 실제 사용 가능한 회전량(usable_gyro)에 곱해서
        # "이 정도 세기로 돌고 싶다"는 목표 회전량(절댓값)을 구함

        if desired_steer > 0:
            # 오른쪽으로 돌고 싶은 경우 -> 오른쪽 구간 데이터에서 보정값을 찾음
            corrected = self._inverse_interpolate(
                target_delta_abs,
                self.right_steer,
                self.right_delta
            )
        else:
            # 왼쪽으로 돌고 싶은 경우 -> 왼쪽 구간 데이터에서 보정값을 찾음
            corrected = self._inverse_interpolate(
                target_delta_abs,
                self.left_steer,
                self.left_delta
            )

        return max(-1.0, min(1.0, corrected))  # 최종 보정값도 -1.0~1.0 범위로 제한해서 반환

    def _inverse_interpolate(self, target_delta_abs, steer_list, delta_list):
        # "목표 회전량(target_delta_abs)"에 대응하는 steer 값을
        # 수집된 데이터 사이를 보간(interpolate)해서 역으로 추정하는 함수
        steer_arr = np.array(steer_list, dtype=float)          # steer 값 배열
        delta_abs_arr = np.abs(np.array(delta_list, dtype=float))  # 회전량 변화의 절댓값 배열

        order = np.argsort(delta_abs_arr)      # 회전량(절댓값) 기준 오름차순 정렬 인덱스
        delta_abs_arr = delta_abs_arr[order]   # 회전량 배열을 오름차순으로 정렬 (보간을 위해 필요)
        steer_arr = steer_arr[order]           # steer 배열도 같은 순서로 재배열

        unique_delta = []   # 중복 제거된 회전량 값들을 담을 리스트
        unique_steer = []   # 그에 대응하는 steer 값들을 담을 리스트

        for d, s in zip(delta_abs_arr, steer_arr):
            # 회전량이 이전 값과 거의 같으면(중복이면) 건너뛰고, 다르면 추가
            # (np.interp는 x값이 같은 지점이 여러 개 있으면 오작동할 수 있어 중복 제거가 필요)
            if len(unique_delta) == 0 or abs(d - unique_delta[-1]) > 1e-6:
                unique_delta.append(d)
                unique_steer.append(s)

        if len(unique_delta) < 2:
            # 보간할 수 있는 서로 다른 지점이 2개 미만이면 보간이 불가능하므로
            # 목표값에 가장 가까운 지점의 steer 값을 그대로 사용
            idx = np.argmin(np.abs(delta_abs_arr - target_delta_abs))
            return float(steer_arr[idx])

        return float(np.interp(
            target_delta_abs,   # 찾고자 하는 목표 회전량
            unique_delta,       # 알려진 회전량들 (x축, 오름차순)
            unique_steer        # 그에 대응하는 steer 값들 (y축)
        ))  # 목표 회전량 사이를 선형 보간하여 대응하는 steer 값을 추정해서 반환

    def save(self, path=CALIBRATION_PATH):
        # 계산된 보정 테이블을 JSON 파일로 저장 (다음에 다시 측정하지 않고 불러와 재사용 가능)
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 저장할 폴더("json 파일저장")가 없으면 생성

        data = {
            "steer_table": self.steer_table,        # 정렬된 steer 값들
            "gyro_table": self.gyro_table,          # 정렬된 gyro 값들
            "base_gyro": self.base_gyro,            # 기준 gyro 값
            "zero_correction": self.zero_correction,  # 보정된 중앙 steer 값
            "usable_gyro": self.usable_gyro          # 좌우 공통 사용 가능 회전량
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)   # data를 보기 좋게 들여쓰기(indent=2)해서 JSON으로 저장

    def load(self, path=CALIBRATION_PATH):
        # 저장해둔 JSON 파일을 불러와서 다시 보정 테이블을 복원
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)   # JSON 파일을 읽어서 파이썬 딕셔너리로 변환

        self.fit(data["steer_table"], data["gyro_table"])
        # 저장해뒀던 steer/gyro 데이터로 다시 fit()을 실행해 보정값들을 재계산


def main():
    calib = SteeringCalibrator()   # 보정 계산기 객체 생성

    if os.path.exists(CALIBRATION_PATH):
        # 이미 보정을 마치고 저장해둔 json 파일이 있으면, 다시 주행하며 측정하지 않고 그대로 불러온다
        print("기존 보정 파일을 불러옵니다:", CALIBRATION_PATH)
        calib.load(CALIBRATION_PATH)
    else:
        # json 파일이 없을 때만 새로 데이터를 수집하고 계산한다
        print("보정 파일이 없어 새로 측정합니다.")
        dataset = collect_steering_data(Car)   # 여러 조향값으로 실제 주행시켜 (steer, gyro) 데이터 수집

        print("===== Raw Dataset =====")
        print(dataset)   # 수집된 원본 데이터셋 출력 (확인용)

        calib.fit(dataset["steer"], dataset["gyro"])  # 수집한 데이터로 보정 테이블 계산
        calib.save()                                 # 계산된 보정 테이블을 JSON 파일로 저장

    print("===== Calibrated Steering Table =====")

    for desired in np.linspace(-1.0, 1.0, 11):
        # -1.0 ~ 1.0 사이 11개 지점에 대해 보정값이 어떻게 나오는지 확인
        desired = round(float(desired), 1)      # 부동소수점 오차 제거
        corrected = calib.steering(desired)     # 원하는 steer 값에 대한 보정값 계산

        print({
            "desired": desired,      # 원래 원하던 조향값
            "corrected": corrected   # 실제로 넣어야 할 보정된 조향값
        })

    # 테스트 주행
    try:
        for desired in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            # 대표적인 조향값 5개로 실제 보정이 잘 되는지 테스트 주행
            corrected = calib.steering(desired)   # 보정된 조향값 계산

            print("run:", {
                "desired": desired,
                "corrected": corrected
            })   # 어떤 값으로 테스트하는지 출력

            Car.steering = corrected     # 보정된 조향값 적용
            car_forward(Car, SPEED)      # 전진 시작
            time.sleep(0.4)              # 0.4초 동안 전진 유지
            Car.stop()                   # 정지
            time.sleep(0.2)              # 잠깐 안정화 대기

            Car.steering = corrected     # 복귀할 때도 동일한 보정 조향값 유지
            car_backward(Car, SPEED)     # 후진 시작 (원위치 복귀)
            time.sleep(0.4)              # 0.4초 동안 후진 유지
            Car.stop()                   # 정지
            time.sleep(0.5)              # 다음 테스트 전 안정화 대기

    finally:
        Car.stop()   # 중간에 오류가 나더라도 마지막엔 반드시 차량을 정지시킴 (안전장치)


if __name__ == "__main__":
    main()   # 이 파일을 직접 실행했을 때만 main() 함수 실행 (다른 파일에서 import될 땐 실행 안 됨)
