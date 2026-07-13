# ============================================================
# line_tracing.py를 실제로 돌리기 전에, 선 인식이 잘 되는지
# 사진 한 장으로 미리 확인해보는 보정용 스크립트.
#
# [방식] 채도(Saturation) 기준으로 구분.
#   실제 트랙 사진을 보니: 트랙 바닥은 빨간색(채도 높음), 선은 흰색(채도 낮음).
#   역광(햇빛)으로 밝기가 다 날아가도, 색이 진하냐 옅으냐(채도)는
#   상대적으로 안정적이라 이 방식이 더 잘 맞을 가능성이 높음.
#
# 사용법:
#   1) 오토카를 실제 트랙 위(선이 잘 보이는 위치)에 놓는다.
#   2) 이 스크립트를 실행한다.
#   3) 생성된 calib_original.jpg / calib_mask.jpg / calib_overlay.jpg 를
#      다운로드해서 눈으로 확인한다.
# ============================================================

import cv2
import numpy as np

from pop import Util

# 채도(S)가 낮으면 흰색/회색 계열 (=선일 가능성 높음).
# 명도(V)는 어느 정도 밝기만 있으면 됨 (역광 영향을 덜 받기 위해 기준을 낮춤).
MAX_SATURATION = 70
MIN_VALUE = 170

# 화면 52%~100% 지점 사용.
# (0.4로 하면 선은 잘 잡히지만, 그 위 배경(나무/건물 있는 지평선)까지
#  같이 밝게 잡혀서 오인식됨 -> 지평선 위쪽은 아예 잘라냄)
ROI_TOP_RATIO = 0.52


def main():
    cam = Util.gstrmer(width=640, height=480, fps=30, flip=0)
    cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("사진을 못 찍었습니다.")
        return

    cv2.imwrite("calib_original.jpg", frame)

    height, width = frame.shape[:2]
    roi_top = int(height * ROI_TOP_RATIO)
    roi = frame[roi_top:height, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, MIN_VALUE])
    upper = np.array([180, MAX_SATURATION, 255])
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imwrite("calib_mask.jpg", mask)

    overlay = frame.copy()
    overlay_roi = overlay[roi_top:height, :]
    overlay_roi[mask > 0] = [0, 255, 0]

    cv2.imwrite("calib_overlay.jpg", overlay)

    white_pixel_count = int(np.sum(mask > 0))
    total_pixel_count = mask.shape[0] * mask.shape[1]
    ratio = white_pixel_count / total_pixel_count * 100

    print("calib_original.jpg / calib_mask.jpg / calib_overlay.jpg 생성 완료")
    print(f"인식된 선 픽셀 비율: {ratio:.1f}%  (5~20% 정도가 적당함)")
    if ratio < 1:
        print("-> 거의 인식이 안 됐습니다. MAX_SATURATION을 높이거나 MIN_VALUE를 낮춰보세요.")
    elif ratio > 40:
        print("-> 너무 많이 인식됐습니다. MAX_SATURATION을 낮추거나 MIN_VALUE를 높여보세요.")


if __name__ == "__main__":
    main()
