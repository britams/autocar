# -*- coding: utf-8 -*-
"""
track_vision.py
──────────────────────────────────────────────────────────────
카메라 영상만 보고 "오토카가 얼마나 움직였는지" 대략 추정하는 파일입니다.
이런 방식을 "Visual Odometry(영상 기반 주행거리 추정)"이라고 부릅니다.

왜 필요한가?
  track_odometry.py는 "모터에 준 속도값 + 조향값"으로 위치를 계산합니다.
  그런데 만약 바퀴가 미끄러지거나(헛돌거나), 모터 속도 보정값이 조금
  안 맞으면 실제 위치와 계산된 위치가 조금씩 어긋날 수 있습니다.

  카메라 영상으로도 "따로" 위치를 계산해두면, 두 값을 서로 비교해서
  "둘이 비슷하게 나오는지" 확인할 수 있고, 나중에 두 값을 섞어서
  (평균 내거나 서로 보정하는 방식으로) 더 정확한 위치를 만드는 데도
  쓸 수 있습니다. 그래서 이 파일은 track_odometry.py를 "대체"하는 게
  아니라 "보완"하는 역할입니다.

  주의! 이 파일은 "라인트레이싱"(카메라로 바닥의 선을 찾아 따라가는 것)
  용도가 아닙니다. 오로지 "오토카가 어느 쪽으로 얼마나 움직였는지"만
  계산합니다.

원리 (아주 쉽게 설명)
  카메라로 사진을 계속 찍으면, 오토카가 움직일 때마다 사진 속 물체들의
  위치도 화면 안에서 조금씩 움직입니다. 예를 들어 오토카가 오른쪽으로
  움직이면, 화면 속 물체들은 반대로 왼쪽으로 흘러가는 것처럼 보입니다.
  이 파일은 "이전 사진"과 "지금 사진"을 비교해서 화면 속 점들이 얼마나
  움직였는지(=옵티컬 플로우, optical flow)를 계산하고, 그 움직임의
  방향을 반대로 뒤집어서 "오토카가 움직인 방향"으로 사용합니다.

  단점(정직하게 말씀드리면): 카메라 하나로는 실제 "몇 미터"를 움직였는지
  정확히 알아낼 수 없습니다(전문 용어로 "스케일 문제"). 그래서 아래
  METERS_PER_PIXEL 값으로 대략 맞춰서 추정할 뿐이며, track_odometry.py
  보다 정밀하지 않습니다. "정확한 값"이 아니라 "비교/보완용 참고 값"으로
  써주세요.
──────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np


class VisualOdometry:
    """
    카메라 프레임을 계속 넣어주면, 화면 속 점들의 움직임을 보고
    (x, y) 위치를 대략 추정해주는 클래스.

    ── 사용법 ──
        vo = VisualOdometry()
        ...(카메라에서 새 프레임을 받을 때마다)...
        vo.update(frame)   # frame: OpenCV로 읽은 BGR 이미지 한 장
        x, y = vo.x, vo.y
    """

    # 한 번에 몇 개의 "추적하기 좋은 점(특징점)"을 화면에서 찾을지.
    # - 값이 클수록: 점이 많아서 평균 계산이 더 안정적이지만 계산이 느려짐.
    # - 값이 작을수록: 계산은 빠르지만 점이 너무 적으면 결과가 흔들릴 수 있음.
    MAX_FEATURES = 200

    # 화면 속 점이 "픽셀 1칸"만큼 움직였을 때, 실제로는 몇 미터
    # 움직인 것으로 볼지 정하는 보정값입니다. 카메라 화각/설치 높이/
    # 렌즈에 따라 실제 값이 달라서, 아래 방법으로 직접 보정해야 합니다.
    #   1) 오토카를 정확히 1m 직진시킨다.
    #   2) 그때 이 값으로 계산된 이동 거리(vo.y 변화량)를 확인한다.
    #   3) METERS_PER_PIXEL = 1m / (계산된 거리와 지금 값의 비율)
    #      만큼 곱하거나 나눠서 다시 맞춘다.
    # - 값이 클수록: 조금만 움직여도 이동 거리를 크게 계산합니다.
    # - 값이 작을수록: 많이 움직여야 이동 거리가 조금 늘어납니다.
    METERS_PER_PIXEL = 0.002

    # 추적 중이던 점이 이 개수보다 적게 남으면, 화면에서 새로 특징점을
    # 다시 찾습니다. (점들이 화면 밖으로 나가거나 추적을 놓치면 점점
    # 줄어들기 때문에, 너무 적어지기 전에 새로 채워 넣어야 계속 정확하게
    # 움직임을 잴 수 있습니다.)
    MIN_FEATURES_BEFORE_REFRESH = 30

    def __init__(self):
        self.x = 0.0   # 카메라 영상만으로 추정한 x 좌표 (미터, 참고용)
        self.y = 0.0   # 카메라 영상만으로 추정한 y 좌표 (미터, 참고용)
        self._prev_gray = None      # 직전 프레임(흑백)
        self._prev_points = None    # 직전 프레임에서 추적하던 점들

    def reset(self):
        """위치를 (0, 0)으로 초기화하고, 다음 프레임부터 새로 추적을 시작합니다."""
        self.x = 0.0
        self.y = 0.0
        self._prev_gray = None
        self._prev_points = None

    def _find_features(self, gray):
        """화면에서 "추적하기 좋은 점(모서리/코너)"들을 새로 찾습니다."""
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.MAX_FEATURES, qualityLevel=0.01, minDistance=7
        )

    def update(self, frame):
        """
        새 카메라 프레임을 받아서 위치(x, y)를 갱신합니다.
        frame: OpenCV로 읽은 컬러(BGR) 이미지 1장.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 처음 호출되는 경우 - 비교할 "이전 프레임"이 아직 없으므로
        # 이번 프레임에서 특징점만 찾아두고 다음 호출을 기다립니다.
        if self._prev_gray is None or self._prev_points is None or len(self._prev_points) < 5:
            self._prev_gray = gray
            self._prev_points = self._find_features(gray)
            return

        # 이전 프레임의 점들이 이번 프레임에서 어디로 이동했는지 추적
        # (Lucas-Kanade 옵티컬 플로우 방법)
        new_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None
        )

        # status가 1인 점만 "추적에 성공한 점"입니다. (화면 밖으로
        # 나갔거나 흐릿해져서 놓친 점은 status가 0이라 걸러냅니다.)
        found = status.flatten() == 1
        good_new = new_points[found]
        good_old = self._prev_points[found]

        if len(good_new) >= 5:
            # 각 점이 (이전 → 지금) 사이에 이동한 픽셀 거리(dx, dy)
            flow = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)

            # 평균이 아니라 "중앙값(median)"을 쓰는 이유: 몇몇 점이 잘못
            # 추적되어 엉뚱하게 튀는 값이 나와도, 중앙값은 그런 이상치에
            # 잘 흔들리지 않습니다 (안정적인 대표값).
            dx_px, dy_px = np.median(flow, axis=0)

            # 카메라가 오토카 앞쪽을 보고 있다고 가정한 부호 규칙:
            #  - 오토카가 오른쪽으로 이동하면 화면 속 물체는 왼쪽으로
            #    흘러가 보입니다(dx_px가 음수) → 그래서 부호를 반대로
            #    뒤집어서(-dx_px) "오토카의 오른쪽 이동"으로 사용합니다.
            #  - 오토카가 앞으로(전진) 이동하면 화면 속 물체는 아래쪽으로
            #    흘러가 보입니다(dy_px가 양수) → 마찬가지로 반대로
            #    뒤집어서(-dy_px) "오토카의 전진"으로 사용합니다.
            self.x += -dx_px * self.METERS_PER_PIXEL
            self.y += -dy_px * self.METERS_PER_PIXEL

        # 다음 번 비교를 위해 이번 프레임을 "이전 프레임"으로 저장
        self._prev_gray = gray
        if len(good_new) < self.MIN_FEATURES_BEFORE_REFRESH:
            self._prev_points = self._find_features(gray)
        else:
            self._prev_points = good_new.reshape(-1, 1, 2)
