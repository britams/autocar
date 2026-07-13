# 오토카(Jetson) Jupyter Notebook 환경에서 실행하는 코드입니다.
# 이 코드로 "장애물 있음(blocked)" / "장애물 없음(free)" 사진 데이터를 모읍니다.

from pop import Pilot
import os
import glob
import ipywidgets as widgets
from IPython.display import display

# 1) 카메라를 켭니다 (300x300 크기)
cam = Pilot.Camera(width=300, height=300)

# Camera 객체를 만들 때 내부적으로 nvargus-daemon을 재시작하는데,
# 이 과정에서 카메라 캡처가 끊긴 채로(화면이 멈춘 채로) 시작되는 경우가 있습니다.
# 캡처 스레드를 한 번 더 재시작해서 이 문제를 예방합니다.
cam.camera.restart()

# 2) 데이터 수집기를 만듭니다. Collision_Avoid(장애물 회피)용 데이터임을 지정합니다.
dataCollector = Pilot.Data_Collector(Pilot.Collision_Avoid, camera=cam)

# 2-1) 조이스틱으로 실제 차량을 움직이며 데이터를 모으려면
#      AutoCar 객체를 만들어서 dataCollector.ac 에 연결해야 합니다.
#      (이 연결이 없으면 조이스틱 조작 시 AttributeError가 발생합니다.)
Car = Pilot.AutoCar()
dataCollector.ac = Car

# 3) 수집 GUI를 표시합니다.
#    - 조이스틱으로 차를 움직이면서
#    - 화면에 장애물이 없으면 "add free" 버튼
#    - 화면에 장애물이 있으면 "add blocked" 버튼을 눌러 사진을 저장합니다.
#    각 300장 이상, 다양한 장소(실내/실외/타일 등)에서 모을수록 좋습니다.
dataCollector.show()

# 사진은 현재 폴더의 collision_dataset/free, collision_dataset/blocked 에 저장됩니다.


# 4) 되돌리기(실수로 잘못 찍었을 때, 가장 최근 사진 1장 삭제) 버튼
def undo_last(_):
    free_files = glob.glob(os.path.join(dataCollector.free_dir, "*.jpg"))
    blocked_files = glob.glob(os.path.join(dataCollector.blocked_dir, "*.jpg"))
    candidates = [(os.path.getmtime(f), f, "free") for f in free_files] + \
                 [(os.path.getmtime(f), f, "blocked") for f in blocked_files]

    if not candidates:
        print("삭제할 사진이 없습니다.")
        return

    candidates.sort(reverse=True)  # 가장 최근에 저장된 사진 찾기
    _, path, kind = candidates[0]
    os.remove(path)

    if kind == "free":
        dataCollector.free_count.value -= 1
    else:
        dataCollector.blocked_count.value -= 1

    print(f"삭제됨 ({kind}): {path}")


undo_button = widgets.Button(description="되돌리기 (마지막 사진 삭제)", button_style="warning")
undo_button.on_click(undo_last)
display(undo_button)
