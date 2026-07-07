# -*- coding: utf-8 -*-
"""
track_vision.py
──────────────────────────────────────────────────────────────
카메라 영상으로 오토카의 위치를 추정하는 VisualOdometry를 담은 파일입니다.

  - track_odometry.py 는 "모터 속도 + 조향각"으로 위치를 추정했다면,
    여기서는 "카메라 영상이 프레임마다 얼마나 움직였는지"를 보고
    위치를 추정합니다. 예를 들어 바닥의 무늬가 화면에서 왼쪽으로
    움직였다면, 실제로는 오토카가 오른쪽으로 이동한 것입니다.
  - 이런 방식을 "Visual Odometry(영상 기반 주행거리 추정)"이라고
    부릅니다.
──────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np


class VisualOdometry:
    """
    카메라 영상만 보고 오토카의 (x, y) 이동 거리를 추정하는 클래스.

    ── 원리 ──
        이전 프레임에서 추적하기 좋은 점들(구석, 무늬 등)을 몇 개
        골라둔 다음, 다음 프레임에서 그 점들이 어디로 이동했는지
        추적합니다(Optical Flow, 광학 흐름). 점들이 평균적으로
        얼마나 움직였는지를 보면 카메라(오토카)가 반대 방향으로 그만큼
        움직였다는 걸 알 수 있습니다.

    ── 사용법 ──
        vo = VisualOdometry()
        ...(카메라에서 새 프레임을 받을 때마다)...
        vo.update(frame_bgr)
        x, y = vo.x, vo.y
    """

    # ────────────────────────────────────────────────────────
    # 조정 가능한 값 - 실제 카메라/바닥 상황에 맞춰 조정하세요.
    # ────────────────────────────────────────────────────────

    # 화면에서 "픽셀 1개가 실제로 몇 미터인지"를 나타내는 보정값입니다.
    # - 이 값은 카메라가 바닥에서 얼마나 높이/각도로 달려있는지에 따라
    #   완전히 달라지므로, 반드시 실제로 보정해야 합니다.
    # - 보정 방법: 오토카를 정확히 1m 앞으로 이동시킨 뒤, 그 동안
    #   VisualOdometry가 측정한 이동 픽셀 수를 확인하고
    #   PIXELS_TO_METERS = 1m / (측정된 픽셀 수) 로 계산해서 넣습니다.
    # - 값이 클수록: 같은 픽셀 이동량에도 "더 많이 움직였다"고 계산합니다.
    PIXELS_TO_METERS = 1.0 / 400.0

    # 추적할 특징점(코너)을 최대 몇 개까지 찾을지.
    # - 값이 클수록: 더 정확하지만 계산이 느려집니다(반응속도 저하).
    # - 값이 작을수록: 더 빠르지만 바닥에 무늬가 적으면 추정이 불안정
    #   해질 수 있습니다.
    MAX_FEATURES = 60

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self._prev_gray = None
        self._prev_points = None

        # goodFeaturesToTrack(추적하기 좋은 점 찾기)에 쓰는 설정값
        self._feature_params = dict(
            maxCorners=self.MAX_FEATURES,
            qualityLevel=0.2,
            minDistance=7,
            blockSize=7,
        )
        # calcOpticalFlowPyrLK(점 추적)에 쓰는 설정값
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self._prev_gray = None
        self._prev_points = None

    def update(self, frame_bgr):
        """
        새 카메라 프레임(BGR 컬러 이미지)을 넣으면, 이전 프레임과
        비교해서 이동한 거리만큼 x, y를 갱신합니다.
        영상이 흔들리거나 추적할 무늬가 없으면 이번 프레임은 건너뜁니다
        (에러를 내지 않고 그냥 이전 위치를 유지합니다).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
            return

        if self._prev_points is None or len(self._prev_points) < 4:
            # 추적할 점이 너무 적으면(예: 바닥이 단색이라 무늬가 없음)
            # 새로 점을 다시 찾아보고, 이번 프레임은 이동량 계산을 건너뜀
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
            return

        new_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if new_points is None:
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
            return

        good_new = new_points[status == 1]
        good_old = self._prev_points[status == 1]

        if len(good_new) >= 4:
            # 추적된 점들이 평균적으로 이동한 픽셀 거리(화면 기준 dx, dy)
            diff = good_new - good_old
            dx_px = float(np.mean(diff[:, 0]))
            dy_px = float(np.mean(diff[:, 1]))

            # 화면에서 오른쪽/아래로 움직인 것처럼 보이면, 실제 오토카는
            # 반대 방향(왼쪽/앞)으로 움직인 것이므로 부호를 반대로 뒤집습니다.
            self.x += -dx_px * self.PIXELS_TO_METERS
            self.y += -dy_px * self.PIXELS_TO_METERS

        # 다음 프레임을 위해 추적점을 다시 계산 (누적 오차 방지)
        self._prev_gray = gray
        self._prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
