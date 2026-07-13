# 오토카(Jetson) Jupyter Notebook 환경에서 실행하는 코드입니다.
# 1_데이터_수집.py 로 모은 사진(collision_dataset)으로 신경망을 학습시키고,
# 학습된 모델을 파일로 저장합니다. (주행은 3_카메라_주행.py 에서 합니다)

from pop import Pilot

MODEL_PATH = "collision_avoid_model.pth"

# 1) 카메라 켜기
cam = Pilot.Camera(width=300, height=300)

# Camera 생성 직후 캡처가 멈춘 채로 시작되는 경우가 있어 예방 차원에서 한 번 재시작합니다.
cam.camera.restart()

# 2) 장애물 인식 신경망(Collision_Avoid) 만들기
CA = Pilot.Collision_Avoid(cam)

# 3) 앞서 모은 collision_dataset 폴더의 사진들을 불러오기
CA.load_datasets()

# 4) 학습 시작 (10번 반복 학습)
#    학습이 진행될 때마다 자동으로 모델이 저장됩니다. (끄려면 autosave=False)
CA.train(times=10)

# ---- 여기까지 실행 후, accuracy(정확도)가 1에 가까우면 잘 학습된 것입니다 ----

# 5) 학습된 모델을 파일로 저장 (3_카메라_주행.py, 4_라이다_카메라_통합_주행.py 에서 이 파일을 불러와서 씁니다)
CA.save_model(path=MODEL_PATH)
print(f"모델 저장 완료: {MODEL_PATH}")

# 6) (선택) 카메라 화면 + 모델 판단 결과 확인
CA.show()

# 참고: 나중에 사진을 더 모아서 이어서 학습하고 싶으면, CA.train() 전에 아래 줄을 추가하세요.
# CA.load_model(path=MODEL_PATH)
