`_cat == 6`이면 이 파일에서는 **AutoCar Prime + NX**로 판단한 상태입니다. 감지 로직상 `PWM(1, 0x5c)` 연결이 성공하면 `_cat = 6`으로 설정됩니다. 즉 **NX 계열 Jetson에서 I2C bus 1의 PWM 보드 주소 0x5c가 잡힌 상태**로 보면 됩니다.

## 1. `_cat == 6`의 장치 종류

코드 주석 기준:

```python
6 : AutoCar Prime + NX
```

따라서 현재 보드는 **AutoCar Prime + NX**입니다.

## 2. 연결 방식 요약

| 장치          | 클래스             |                       연결 방식 |        bus | 주소/채널                           |
| ----------- | --------------- | --------------------------: | ---------: | ------------------------------- |
| PWM 컨트롤러    | `PWM`           |                         I2C |          1 | `0x5c`, `0x5e`                  |
| 조향 서보       | `Wheel`         |                PWM over I2C |          1 | 주소 `0x5c`, 채널 `15`              |
| 카메라 팬/틸트 서보 | `CameraPod`     |                PWM over I2C |          1 | 주소 `0x5c`, 채널 `14`, `13`        |
| 주행 모터       | `Driving`       |                PWM over I2C |          1 | 주소 `0x5e`, 채널 `0~3`             |
| IMU / 6축 센서 | `axis6`         |                         I2C |          8 | 주소 `0x68`                       |
| 조이스틱 UI     | `joystick`      |                   WebSocket |       네트워크 | 기본 포트 `8885`                    |
| 카메라         | `Camera`        | 별도 `pop.__init__.Camera` 사용 | 보통 CSI/USB | 이 파일에서는 직접 정의 안 함               |
| LiDAR 감지    | `has_lidar`     |                   USB 장치 검색 |        USB | `10c4:ea60` 포함 여부               |
| 객체 인식       | `Object_Follow` |           카메라 + YOLOv4 tiny |      소프트웨어 | `_cat == 6`이면 YOLOv4 tiny 경로 사용 |

## 3. 핵심 클래스 구성

전체 구조를 간단히 보면 이렇게 되어 있습니다.

```text
PWM
 ├─ I2C PWM 제어 기본 클래스
 │
Driving
 ├─ 주행 모터 제어
 ├─ _cat == 6이면 I2C bus 1, addr 0x5e 사용
 │
Wheel
 ├─ 조향 서보 제어
 ├─ _cat == 6이면 I2C bus 1, addr 0x5c 사용
 │
CameraPod
 ├─ 카메라 팬/틸트 서보 제어
 ├─ _cat == 6이면 I2C bus 1, addr 0x5c 사용
 │
axis6
 ├─ 자이로/가속도 센서
 ├─ _cat == 6이면 I2C bus 8, addr 0x68 사용
 │
AutoCar(axis6)
 ├─ Wheel 포함
 ├─ CameraPod 포함
 ├─ Driving 포함
 ├─ joystick 포함
 │
Data_Collector
 ├─ 카메라 데이터 수집
 ├─ 조이스틱으로 주행하면서 이미지 저장
 │
Collision_Avoid
 ├─ 충돌 회피 학습/추론
 │
Track_Follow
 ├─ 라인/트랙 주행 학습/추론
 │
Object_Follow
 ├─ 객체 인식/추적
 └─ _cat == 6이면 YOLOv4 tiny 사용
```

## 4. `_cat == 6`에서 `AutoCar()`를 만들면 내부적으로 생기는 것

Jupyter에서 보통 이렇게 만들게 됩니다.

```python
from pop import Pilot

car = Pilot.AutoCar()
```

또는:

```python
car = Pilot.get_Control()
```

`get_Control()`은 `_cat`이 `0, 1, 3, 5, 6`이면 `AutoCar()`를 반환합니다. 따라서 `_cat == 6`에서는 `SerBot`이 아니라 `AutoCar` 계열로 동작합니다.

`AutoCar.__init__()` 내부에서는 `_cat == 6`일 때 다음처럼 구성됩니다.

```python
bus = 1
self.wheel = Wheel(bus, 0x5c, 50)
self.campod = CameraPod(bus, 0x5c, 50)
self.drv = Driving(1, 0x5e, 200)
```

즉 구조는 다음과 같습니다.

```text
car
 ├─ car.wheel   → 조향 서보, I2C bus 1, addr 0x5c, 50Hz
 ├─ car.campod  → 카메라 팬/틸트, I2C bus 1, addr 0x5c, 50Hz
 ├─ car.drv     → 좌우 모터 주행, I2C bus 1, addr 0x5e, 200Hz
 └─ axis6 기능  → IMU, I2C bus 8, addr 0x68
```

## 5. 사용 가능한 주요 메서드

### 주행 제어

```python
car.forward(30)     # 전진
car.backward(30)    # 후진
car.stop()          # 정지
car.setSpeed(40)    # 현재 주행 상태에서 속도 변경
car.getSpeed()      # 현재 속도 확인
```

속도 범위는 코드상 `MIN_SPEED = 20`, `MAX_SPEED = 99`입니다.

### 조향 제어

```python
car.turnLeft()
car.turnRight()
car.turnCenter()
```

또는 연속값으로:

```python
car.steering = -1.0   # 최대 왼쪽
car.steering = 0.0    # 중앙
car.steering = 1.0    # 최대 오른쪽
```

주의할 점은 `_cat == 6`에서는 조향 값이 내부에서 한 번 반전됩니다.

```python
if _cat==3 or _cat==6:
    value *= -1
```

그래서 실제 좌우 방향이 예상과 반대면 이 부분 때문입니다.

### 카메라 팬/틸트

```python
car.camPan(0)
car.camTilt(30)

car.cam2Front()
car.cam2Back()
car.cam2Left(30)
car.cam2Right(30)
```

`_cat == 6`에서는 `CameraPod`의 `panCenterAngle`이 `188`, `tiltCenterAngle`이 `13`으로 설정됩니다.

### IMU / 6축 센서

`AutoCar`가 `axis6`를 상속하므로 바로 호출 가능합니다.

```python
car.getGyro()
car.getGyro("x")
car.getAccel()
car.getAccel("z")
```

`_cat == 6`에서는 IMU가 `smbus.SMBus(8)`, 주소 `0x68`로 연결됩니다.

## 6. 실제 연결 확인용 Jupyter 코드

아래 코드를 Jupyter Lab에서 실행하면 현재 구성 상태를 확인하기 좋습니다.

```python
from pop import Pilot

cat_names = {
    0: "AutoCar",
    1: "AutoCar Racing",
    2: "SerBot",
    3: "AutoCar Prime",
    4: "SerBot Prime X",
    5: "AutoCar Prime X",
    6: "AutoCar Prime + NX",
}

print("cat:", Pilot._cat)
print("type:", cat_names.get(Pilot._cat, "Unknown"))
print("has_lidar:", Pilot.has_lidar)

car = Pilot.get_Control()

print("control class:", type(car))
print("wheel:", type(car.wheel))
print("campod:", type(car.campod))
print("drv:", type(car.drv))

print("gyro:", car.getGyro())
print("accel:", car.getAccel())
```

## 7. I2C 장치 확인 명령어

젯슨 터미널 또는 Jupyter에서 `!`를 붙여 실행합니다.

```bash
i2cdetect -y 1
i2cdetect -y 8
```

Jupyter에서는:

```python
!i2cdetect -y 1
!i2cdetect -y 8
```

기대되는 값은 대략 다음입니다.

```text
bus 1: 0x5c, 0x5e
bus 8: 0x68
```

정리하면, **현재 6번 상태에서는 CAN 방식이 아니라 I2C PWM 방식으로 주행/조향/카메라 팬틸트를 제어하고, IMU만 I2C bus 8의 0x68을 사용**하는 구조입니다.