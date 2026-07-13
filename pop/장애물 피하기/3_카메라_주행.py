# 오토카(Jetson) Jupyter Notebook 환경에서 실행하는 코드입니다.
# 2_딥러닝_학습.py 에서 저장한 모델을 불러와서, 카메라만 이용해 장애물 회피 주행을 합니다.
# (학습은 하지 않습니다. 학습은 2_딥러닝_학습.py 에서 이미 끝났다고 가정합니다)

from pop import Pilot

MODEL_PATH = "collision_avoid_model.pth"

# 1) 카메라 켜기
cam = Pilot.Camera(width=300, height=300)
cam.camera.restart()

# 2) 신경망 준비 후, 저장된 모델 불러오기 (학습 없이 바로 사용)
CA = Pilot.Collision_Avoid(cam)
CA.load_model(path=MODEL_PATH)

# 3) (선택) 카메라 화면 + 모델 판단 결과 확인
CA.show()

# 4) 실제 차량 주행 준비
Car = Pilot.AutoCar()
Car.setSpeed(50)


def drive(value):
    # value: 현재 화면이 장애물일 확률 (0~1)
    if value <= 0.5:
        # 장애물 없음 -> 직진
        Car.steering = 0
        Car.forward()
    else:
        # 장애물 있음 -> 우측으로 후진해서 회피
        Car.steering = 1
        Car.backward()


# 5) 무한 반복: 카메라를 계속 보면서 장애물 여부에 따라 주행
#    이 셀은 계속 "Busy" 상태로 남아있는 게 정상입니다. 멈추려면 Jupyter 툴바의 정지(■) 버튼을 누르세요.
while True:
    CA.run(callback=drive)
