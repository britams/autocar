import time

import numpy as np
from pop import AI, Pilot

# ============================================================
# AutoCar 객체 생성
# ============================================================

Car = Pilot.AutoCar()


# ============================================================
# 실행 옵션
# ============================================================

# False: 새로 학습
# True : 저장된 모델 불러오기
RESTORE_MODE = True

# 체크포인트 이름은 고정해야 한다.
# time.time()을 붙이면 실행할 때마다 다른 모델 이름이 되어 restore가 안 된다.
CKPT_NAME = "dnn_yaw_attitude_control"


# ============================================================
# 기본 설정
# ============================================================

SPEED = 60
RUN_TIME = 5.0
CONTROL_DT = 0.05

# getGyro("z")가 raw 값이면 MPU6050 기준 1/131 정도
# 이미 deg/s라면 1.0으로 변경
GYRO_SCALE = 1.0 / 131.0

BIAS_CALIB_TIME = 2.0

STEER_MIN = -1.0
STEER_MAX = 1.0

GYRO_DEADBAND = 0.3
YAW_DEADBAND = 0.5

# DNN 입력 정규화 기준
MAX_YAW_ERROR = 30.0
MAX_GYRO_DPS = 120.0
MAX_SPEED = 100.0

# 교사 제어기 게인
K_YAW = 0.035
K_GYRO = 0.015

# 조향 방향 보정값
# 방향이 반대로 움직이면 -1.0과 1.0을 바꿔서 테스트
STEER_SIGN = -1.0

# DNN 출력 전체 조절
AI_GAIN = 1.0

# 학습 설정
TRAIN_TIMES = 1000
PRINT_EVERY = 100


# ============================================================
# 기본 유틸 함수
# ============================================================

def clip(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))


def car_forward(car, speed):
    try:
        car.forward(speed)
    except TypeError:
        car.forward()


def car_backward(car, speed):
    try:
        car.backward(speed)
    except TypeError:
        car.backward()


def get_gyro_z_raw(car):
    return float(car.getGyro("z"))


def get_gyro_z_dps(car, gyro_bias_raw):
    raw = get_gyro_z_raw(car)

    corrected_raw = raw - gyro_bias_raw
    gyro_dps = corrected_raw * GYRO_SCALE

    if abs(gyro_dps) < GYRO_DEADBAND:
        gyro_dps = 0.0

    return raw, corrected_raw, gyro_dps


def model_output_to_float(value):
    arr = np.array(value)
    return float(arr.reshape(-1)[0])


# ============================================================
# DNN 입력 정규화
# ============================================================

def make_dnn_input(yaw_error, gyro_dps, speed):
    """
    DNN 입력 3개:
        1. yaw_error 정규화
        2. gyro_dps 정규화
        3. speed 정규화

    반환:
        [x1, x2, x3]
    """

    yaw_error = clip(yaw_error, -MAX_YAW_ERROR, MAX_YAW_ERROR)
    gyro_dps = clip(gyro_dps, -MAX_GYRO_DPS, MAX_GYRO_DPS)
    speed = clip(speed, 0.0, MAX_SPEED)

    x1 = yaw_error / MAX_YAW_ERROR
    x2 = gyro_dps / MAX_GYRO_DPS
    x3 = speed / MAX_SPEED

    return [x1, x2, x3]


def make_dnn_batch_input(yaw_error, gyro_dps, speed):
    """
    Keras predict 입력용 batch 데이터 생성.

    단일 데이터라도 shape이 (3,)이면 안 되고,
    반드시 shape이 (1, 3)이어야 한다.
    """

    x = make_dnn_input(
        yaw_error=yaw_error,
        gyro_dps=gyro_dps,
        speed=speed
    )

    x = np.array([x], dtype=np.float32)

    return x


# ============================================================
# 교사 제어기
# ============================================================

def teacher_controller(yaw_error, gyro_dps, speed):
    """
    DNN이 학습할 목표 steering 값을 만드는 교사 제어기.
    """

    if abs(yaw_error) < YAW_DEADBAND:
        yaw_error = 0.0

    base_cmd = (
        K_YAW * yaw_error
        - K_GYRO * gyro_dps
    )

    steer = STEER_SIGN * base_cmd

    # 속도가 빠르면 조향을 조금 줄임
    speed_scale = 1.0 - 0.3 * (speed / MAX_SPEED)
    steer = steer * speed_scale

    steer = clip(steer, STEER_MIN, STEER_MAX)

    return steer


# ============================================================
# DNN 학습 데이터 생성
# ============================================================

def make_training_data():
    X_data = []
    Y_data = []

    yaw_values = np.linspace(-30.0, 30.0, 25)
    gyro_values = np.linspace(-120.0, 120.0, 25)
    speed_values = [40.0, 60.0, 70.0]

    for speed in speed_values:
        for yaw_error in yaw_values:
            for gyro_dps in gyro_values:
                x = make_dnn_input(
                    yaw_error=yaw_error,
                    gyro_dps=gyro_dps,
                    speed=speed
                )

                y = teacher_controller(
                    yaw_error=yaw_error,
                    gyro_dps=gyro_dps,
                    speed=speed
                )

                X_data.append(x)
                Y_data.append([y])

    return X_data, Y_data


# ============================================================
# DNN 모델 생성
# ============================================================

def create_dnn_controller(restore):
    """
    restore=False:
        새 모델 생성

    restore=True:
        CKPT_NAME에 해당하는 저장 모델 복원
    """

    dnn = AI.DNN(
        input_size=3,
        hidden_size=10,
        output_size=1,
        layer_level=5,
        restore=restore,
        ckpt_name=CKPT_NAME,
        softmax=False
    )

    return dnn


# ============================================================
# DNN 학습 또는 복원
# ============================================================

def prepare_dnn_controller(restore):
    """
    RESTORE_MODE 값에 따라 학습 또는 복원을 선택한다.
    """

    if restore:
        print("===== 저장된 DNN 모델 불러오기 =====")
        print("ckpt_name:", CKPT_NAME)

        dnn = create_dnn_controller(restore=True)

        print("===== DNN 모델 불러오기 완료 =====")

        return dnn

    else:
        print("===== 새 DNN 모델 학습 시작 =====")
        print("ckpt_name:", CKPT_NAME)

        X_data, Y_data = make_training_data()

        print("===== DNN 학습 데이터 =====")
        print("X sample:", X_data[:5])
        print("Y sample:", Y_data[:5])
        print("data count:", len(X_data))

        dnn = create_dnn_controller(restore=False)

        dnn.X_data = X_data
        dnn.Y_data = Y_data

        dnn.train(times=TRAIN_TIMES, print_every=PRINT_EVERY)

        print("===== 새 DNN 모델 학습 완료 =====")

        return dnn


# ============================================================
# DNN 조향값 예측
# ============================================================

def dnn_predict_steering(dnn, yaw_error, gyro_dps, speed):
    """
    DNN 조향 예측 함수.

    pop.AI.DNN의 dnn.run(x)를 사용하지 않고,
    내부 Keras 모델인 dnn.model.predict()를 직접 사용한다.

    입력 shape:
        (1, 3)

    예:
        [[yaw_error_norm, gyro_dps_norm, speed_norm]]
    """

    x = make_dnn_batch_input(
        yaw_error=yaw_error,
        gyro_dps=gyro_dps,
        speed=speed
    )

    value = dnn.model.predict(x, verbose=0)

    steer = model_output_to_float(value)

    steer = steer * AI_GAIN
    steer = clip(steer, STEER_MIN, STEER_MAX)

    return steer


# ============================================================
# gyro bias 측정
# ============================================================

def calibrate_gyro_bias(car, sec=2.0):
    print("===== gyro bias 측정 시작 =====")
    print("차량을 움직이지 말고 기다리세요.")

    values = []
    start = time.time()

    while time.time() - start < sec:
        z = get_gyro_z_raw(car)
        values.append(z)
        time.sleep(0.02)

    bias = sum(values) / len(values)

    print({
        "gyro_bias_raw": bias,
        "sample_count": len(values)
    })

    print("===== gyro bias 측정 완료 =====")

    return bias


# ============================================================
# DNN 기반 초기 방향 유지 자세제어
# ============================================================

def hold_initial_heading_with_dnn(car, dnn, speed=60, run_time=5.0):
    gyro_bias_raw = calibrate_gyro_bias(
        car=car,
        sec=BIAS_CALIB_TIME
    )

    yaw_angle = 0.0
    target_yaw = 0.0

    print("===== DNN 초기 방향 유지 주행 시작 =====")
    print("target_yaw:", target_yaw)

    car.steering = 0.0
    car_forward(car, speed)

    start_time = time.time()
    last_time = start_time

    log = {
        "time": [],
        "gyro_raw": [],
        "gyro_dps": [],
        "yaw_angle": [],
        "yaw_error": [],
        "steer_cmd": []
    }

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            if dt <= 0:
                dt = CONTROL_DT

            gyro_raw, gyro_corrected_raw, gyro_dps = get_gyro_z_dps(
                car=car,
                gyro_bias_raw=gyro_bias_raw
            )

            # gyro 적분으로 yaw 추정
            yaw_angle += gyro_dps * dt

            # 시작 방향을 유지하므로 목표 yaw는 0
            yaw_error = target_yaw - yaw_angle

            # DNN이 자세 오차와 회전 속도를 보고 조향값 출력
            steer_cmd = dnn_predict_steering(
                dnn=dnn,
                yaw_error=yaw_error,
                gyro_dps=gyro_dps,
                speed=speed
            )

            car.steering = steer_cmd

            t = now - start_time

            log["time"].append(t)
            log["gyro_raw"].append(gyro_raw)
            log["gyro_dps"].append(gyro_dps)
            log["yaw_angle"].append(yaw_angle)
            log["yaw_error"].append(yaw_error)
            log["steer_cmd"].append(steer_cmd)

            print({
                "time": round(t, 2),
                "gyro_raw": round(gyro_raw, 3),
                "gyro_dps": round(gyro_dps, 3),
                "yaw_angle_deg": round(yaw_angle, 3),
                "yaw_error_deg": round(yaw_error, 3),
                "steer_cmd": round(steer_cmd, 3)
            })

            time.sleep(CONTROL_DT)

            if t >= run_time:
                break

    finally:
        car.stop()
        print("===== 주행 종료 =====")

    return log


# ============================================================
# 전진 후 후진 복귀 테스트
# ============================================================

def hold_initial_heading_forward_backward_with_dnn(car, dnn):
    log = hold_initial_heading_with_dnn(
        car=car,
        dnn=dnn,
        speed=SPEED,
        run_time=RUN_TIME
    )

    time.sleep(0.5)

    print("===== 후진 복귀 =====")

    car.steering = 0.0
    car_backward(car, SPEED)
    time.sleep(RUN_TIME)
    car.stop()

    return log


# ============================================================
# DNN 예측 확인
# ============================================================

def print_dnn_test(dnn):
    print("===== DNN 예측 테스트 =====")

    test_cases = [
        [-20.0, 0.0, SPEED],
        [-10.0, 0.0, SPEED],
        [-5.0, 0.0, SPEED],
        [0.0, 0.0, SPEED],
        [5.0, 0.0, SPEED],
        [10.0, 0.0, SPEED],
        [20.0, 0.0, SPEED],
        [0.0, -60.0, SPEED],
        [0.0, 60.0, SPEED],
    ]

    for yaw_error, gyro_dps, speed in test_cases:
        teacher = teacher_controller(
            yaw_error=yaw_error,
            gyro_dps=gyro_dps,
            speed=speed
        )

        start = time.time()

        pred = dnn_predict_steering(
            dnn=dnn,
            yaw_error=yaw_error,
            gyro_dps=gyro_dps,
            speed=speed
        )

        infer_ms = (time.time() - start) * 1000.0

        print({
            "yaw_error": yaw_error,
            "gyro_dps": gyro_dps,
            "speed": speed,
            "teacher": round(teacher, 3),
            "dnn_pred": round(pred, 3),
            "infer_ms": round(infer_ms, 3)
        })


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("===== 실행 설정 =====")
    print("RESTORE_MODE:", RESTORE_MODE)
    print("CKPT_NAME:", CKPT_NAME)

    dnn = prepare_dnn_controller(restore=RESTORE_MODE)

    print_dnn_test(dnn)

    hold_initial_heading_forward_backward_with_dnn(Car, dnn)