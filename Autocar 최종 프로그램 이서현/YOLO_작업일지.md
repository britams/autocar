# 자율주행 작업 일지

> 오토카 프로젝트에서 "자율주행 담당"(라이다+YOLO+OpenCV 전부)으로 진행한 내용을 정리한 문서입니다.
> 초보자도 이해할 수 있게 최대한 쉽고 자세하게 적었습니다.
> 앞으로 작업이 진행될 때마다 이 문서를 계속 업데이트합니다.

## 센서/기술 역할 분담

전부 내 담당.

| | 역할 |
|---|---|
| **라이다** | 360도 전방위, 정확한 거리 측정 → 정지/회피 여부 결정(안전 메인) |
| **YOLO** | 정면 카메라로 물체 종류 판단(사람/자동차 등) → TTS 안내에 사용(보조) |
| **OpenCV** | 트랙 선 색상 검출 → 두 선 사이 중앙 유지(라인 트레이싱) |

## ✅ 진행 상황 요약

팀 프로젝트에서 **자율주행(라이다+YOLO+OpenCV) 전체가 담당 범위**이며, 아래 내용까지 진행되었습니다.

- 오토카(Jetson)에서 YOLOv3-tiny 실시간 객체 탐지 성공 (신뢰도 최대 0.98)
- TensorRT 변환으로 **FPS 6.0 → 30.1**, 5배 성능 개선
- 이 결과(박스 위치, 클래스, 신뢰도)를 다른 팀원(조향/속도 제어 담당)에게 넘겨서 자율주행 로직에 연결하면 됨
- 참고용으로 "탐지 결과를 실제 조향/속도 명령으로 바꾸는" 예시 코드(`jetson_obstacle_avoidance.py`)까지 만들어서 **"정지 후 생각"(stop-then-think) 방식**으로 설계함 — 다만 모터의 실제 반응 속도·정밀 튜닝은 담당 범위 밖이라 팀의 하드웨어 제어 담당자 몫으로 넘김
- 카메라는 화각(FOV)이 제한적이라 정면 좁은 범위만 보이므로, 360도 전체를 보는 **라이다와 함께 쓰는 것을 권장** (`sensor_fusion.py`에 설계 방식 정리해둠, 실측/실기 테스트는 라이다 담당자 몫)

---

## 0. 이 프로젝트에서 YOLO를 왜 쓰는가

- **YOLO(You Only Look Once)**: 카메라로 찍은 이미지 안에서 "무엇이(사람/자동차 등)" "어디에" 있는지 실시간으로 찾아주는 AI 모델(객체 탐지 모델)
- 우리 오토카에서 쓰는 이유:
  1. **장애물 감지**: 앞에 사람/물체가 있으면 인식해서 정지 또는 회피
  2. **음성 알림**: 장애물이 있으면 TTS(음성)로 알려주기
- on-device AI 프로젝트라, Jetson 보드 위에서 직접 돌아가야 하고, 성능이 낮은 장비라 **가벼운(경량) 모델**을 써야 함

---

## 1. 개발 환경 구조 이해하기

| 환경 | 역할 |
|---|---|
| **WSL(우분투, 이 컴퓨터/노트북)** | 코드를 미리 짜고, 웹캠으로 기본 테스트하는 "연습장" |
| **오토카(Jetson Nano)** | 실제로 로봇에 들어가서 돌아가는 "진짜 실행 환경" |

**중요한 차이점:**
- WSL: Python 3.10, 최신 라이브러리(ultralytics YOLOv8) 사용 가능
- Jetson: **Python 3.6.9**, **OpenCV 4.3.0** (2020년에 나온 구버전, JetPack 4.3) → 최신 라이브러리 대부분 설치가 안 됨

---

## 2. WSL에서 한 것 (1단계: 기본 개념 익히기 + 연습)

### 2-1. 가상환경 만들기
- 위치: `Autocar 최종 프로그램/.venv`
- 왜 필요한가: 프로젝트마다 필요한 파이썬 패키지 버전이 다를 수 있어서, 서로 안 꼬이게 독립된 공간을 만드는 것

### 2-2. YOLOv8n(nano)으로 정지 이미지 탐지 테스트
- 파일: `test_yolo.py`
- `ultralytics` 라이브러리의 `yolov8n.pt`(가장 가벼운 버전) 사용
- 결과: 버스 사진에서 사람 3명, 버스 1대 정확히 탐지 성공

### 2-3. 웹캠 실시간 탐지
- 파일: `webcam_yolo.py`
- USB 웹캠을 `usbipd-win`으로 WSL에 연결해서 실시간 탐지 테스트
- **겪은 문제와 해결:**
  - 화면이 초록색으로만 나옴 → 카메라 영상 포맷(YUYV)을 OpenCV가 잘못 해석하는 문제 → `MJPG` 포맷으로 강제 지정해서 해결
  - Qt 창이 안 뜸 → `QT_QPA_PLATFORM=xcb` 환경변수로 해결
  - TTS 말하는 동안 화면이 멈춤 → TTS를 별도 스레드(`threading`)에서 실행하도록 수정

### 2-4. TTS(음성 알림) 연결
- 파일: `tts_util.py` (공용 모듈, 여러 파일에서 재사용)
- `espeak-ng`(오프라인 음성 합성) + `paplay`(WSL 재생) 조합
- 사람이 감지되면 "전방에 장애물이 있습니다" 음성 안내
- 3초 쿨다운을 둬서 너무 자주 말하지 않도록 함

### 2-5. 라이다 센서 퓨전 (설계만, WSL 테스트 불가)
- 파일: `sensor_fusion.py`
- 라이다 드라이버(`pop/LiDAR/_rplidar.so`)가 **ARM(Jetson) 전용**이라 WSL(x86_64)에서는 아예 실행 불가
- YOLO가 찾은 물체의 화면 속 위치(픽셀) → 각도로 변환 → 그 각도의 라이다 거리값과 매칭하는 로직만 미리 작성해둠
- 실측 필요한 값: 카메라 화각(FOV), 라이다 0도 기준 방향

### 2-6. 장애물 회피 로직 (설계만, WSL 테스트 불가)
- 파일: `obstacle_avoidance.py`
- `pop.Pilot.AutoCar()`(실제 오토카 제어 클래스) 기반으로 작성
- 로직: 화면에서 가장 큰(=가장 가까운) 물체를 찾아서
  - 아주 가까우면 → 정지 + 음성 경고
  - 어느 정도 가까우면 → 물체 반대쪽으로 핸들 꺾어서 회피
  - 멀면 → 그냥 직진

### 2-7. 카메라는 정면 고정, 나머지 방향은 라이다 담당
- `pop/CAN.py`의 `Car` 클래스에 `camPan()`/`camTilt()`(카메라 팬/틸트 서보 제어) 함수가 존재하긴 하지만, YOLO 코드에서는 **이 기능을 쓰지 않고 카메라가 정면 고정이라고 가정**하고 작성함
- 이유: 카메라가 움직이면 "화면 속 좌우 위치 → 실제 방향" 계산 로직이 팬 각도까지 고려해야 해서 훨씬 복잡해짐
- 역할 분담: **카메라(YOLO)는 정면만 담당, 측면/후방 등 나머지 방향은 라이다가 담당**하는 구조로 가는 게 가장 현실적
- **카메라와 라이다는 서로 상호보완 관계:**
  - 카메라(YOLO): 정면만 보지만 "이게 사람인지 자동차인지" **종류**를 구분할 수 있음 (라이다는 종류를 모름, "뭔가 있다"만 앎)
  - 라이다: 360도 전체를 보고 **정확한 거리**를 알지만, "그게 뭔지"는 모름
  - 둘을 합치면(`sensor_fusion.py`) → "정면 30cm 앞에 사람이 있다" 같은 완성된 정보가 나옴

### 2-8. 최종 역할 분담 방향 정리
여러 시행착오(특히 구버전 Jetson과 최신 도구 간 호환성 문제) 끝에, 아래처럼 역할을 나누는 것으로 방향을 정함:

| 기능 | 담당 | 이유 |
|---|---|---|
| **라인 트레이싱**(트랙 두 선 사이 가운데 유지) | **OpenCV 색상 검출** | 선은 색이 뚜렷해서 딥러닝 없이도 빠르고 정확함 |
| **장애물 회피(안전)** | **라이다(메인)** | 360도 전방위, 정확한 거리, Jetson 버전 호환성 문제 없음 |
| **장애물 종류 인식** | **YOLO(보조)** | "뭔가 있다"가 아니라 "사람이다/자동차다"처럼 종류를 구분해서 TTS 안내에 활용 |

즉 라이다가 "멈출지 말지"를 결정하는 **안전 담당**이고, YOLO는 그 위에 "무엇을 봤는지" 설명을 얹어주는 **보조 담당**으로 구성.

### 2-9. 라인 트레이싱 코드 작성 (OpenCV 색상 검출 방식)
- 파일: `line_tracing.py`, `line_calibration.py`
- **원리**: HSV 색공간에서 흰색/노란색 범위를 필터링해서 선 픽셀만 골라내고, 화면을 좌/우로 나눠 각각 선의 위치를 찾은 뒤 그 정가운데를 트랙 중심으로 계산 → 화면 중심과의 차이만큼 조향
- `line_calibration.py`: 실제 트랙 사진 한 장을 찍어서 색상 필터가 잘 맞는지 미리 확인하는 보정용 스크립트 (`calib_original.jpg`, `calib_mask.jpg`, `calib_overlay.jpg` 생성)
- ⏳ **아직 실기 테스트 안 함** — 실제 트랙 색상에 맞게 `LOWER_WHITE`/`UPPER_WHITE`/`LOWER_YELLOW`/`UPPER_YELLOW` 값 보정 필요
- 카메라는 팬/틸트 없이 **정면 고정**이라고 가정하고 설계함 (실제로 `pop/CAN.py`에 `camPan`/`camTilt` 기능이 존재하지만 이 코드에서는 사용하지 않음)

### 2-10. 실제 트랙 현장 색상 보정 시도 (미완료, 다음에 이어서)

운동장 트랙에서 `line_calibration.py`로 여러 번 시도함. 겪은 문제와 시행착오:

| 시도 | 조건 | 인식 비율 | 문제 |
|---|---|---|---|
| 1차 | 카메라가 해를 정면으로 바라봄, HSV 흰색/노란색 고정값 | 64.6% | 역광으로 화면 절반이 하얗게 날아가서(과다노출) 하늘과 선을 구분 못함 |
| 2차 | Otsu 자동 임계값으로 변경 | 65.4% | 여전히 역광 문제로 큰 차이 없음 |
| 3차 | 채도(S) 기준으로 변경, ROI를 화면 아래 25%로 좁힘 | 93.9% | ROI가 선이 없는 순수 트랙 바닥만 보고 있었음 (선은 화면 중간쯤에 위치) |
| 4차 | ROI를 화면 40%~100%로 넓힘 | 33.9~39.8% | 지평선(나무/건물 배경)까지 밝게 잡혀서 오인식 |
| 5차 | 오토카를 해를 등지게 회전, ROI를 52%~100%로 좁힘 | 26.1% | 여전히 배경 일부 포함 |
| 6차 | 카메라 각도를 손으로 아래로 내려 지평선 자체를 화면에서 제거 | **2.7%** | 역광 문제는 해결됐으나 이번엔 기준이 너무 엄격해서 선을 거의 못 잡음 |
| 7차 | 채도/명도 기준 완화 (S<70, V>170) | 64.6% | 이번엔 너무 헐렁해져서 트랙 바닥 전체가 다 잡힘 |
| 8차 | `color_sample.py`로 선/트랙의 정확한 HSV 수치 직접 측정 | - | 측정해보니 선(S=52~74)과 트랙(S=65)의 채도 차이가 거의 없어서, 지금 조명(핑크빛 도는 색감)에서는 색상만으로 깔끔하게 구분하기 어려움을 확인 |

**결론 및 다음 시도 방향:**
- 역광(태양 정면)은 카메라 각도를 낮춰서(지평선 미포함) 어느 정도 해결 가능함을 확인
- 다만 이 조명 조건(맑은 날 정오 근처, 카메라 색감이 핑크빛으로 치우침) 자체가 흰 선과 빨간 트랙의 색 대비를 약하게 만들어서, HSV 고정값 튜닝만으로는 한계가 있음
- **다음에 시도해볼 것**: (1) 흐린 날 또는 저녁 무렵 등 역광이 덜한 조건에서 재시도, (2) `color_sample.py`로 정확한 좌표에서 다시 측정, (3) 색상 대신 엣지(경계선) 검출 방식도 고려, (4) 카메라 자체의 화이트밸런스 설정 확인

### 2-11. 라인 트레이싱 + 장애물 회피 통합 (핵심 설계: "목표 지점을 옆으로 잠깐 옮기기")
- 파일: `line_tracing_with_avoidance.py`
- **요구사항**: 장애물의 종류는 중요하지 않지만, **회피 후 원래 라인으로 자연스럽게 복귀하는 것이 중요함**
- **설계 아이디어**:
  - 회피할 때 "라인 추적을 끄고 blind하게 핸들을 꺾는" 방식이 아니라
  - 라인 추적 계산 로직은 **항상 똑같이 계속 돌아가고**, 장애물이 있을 때만 "목표로 삼는 중심 위치"를 옆으로 잠깐(`AVOID_OFFSET_PIXELS`만큼, `AVOID_DURATION_SEC`초 동안) 옮김
  - 회피 시간이 끝나면 목표 지점이 자동으로 화면 정중앙으로 돌아가므로, **같은 제어 로직이 자연스럽게 원래 라인 중심으로 복귀시켜줌**
- YOLO(TensorRT)는 "장애물이 왼쪽/오른쪽 어디에 있는지"만 판단하는 데 사용 (종류는 안 씀)
- ⏳ **아직 실기 테스트 안 함**

### 2-12. 사람 따라가기 (YOLO person 추적)
- 파일: `jetson_follow_person.py`
- 라인 트레이싱은 색상 보정 문제로 잠시 보류하고, 대신 색상 보정이 필요 없는 **사람 추적 기능**을 먼저 만듦
- 원리: YOLO로 "person" 클래스만 찾아서, 가장 큰(가까운) 사람을 기준으로
  - 화면 중앙에 오도록 조향
  - 박스 크기(거리 추정)로 전진/정지 결정 (멀면 전진, 적당하면 서행, 가까우면 정지)
- 사람이 안 보이면 안전하게 정지
- ⏳ **작성만 완료, 실기 테스트는 다음에 (오늘은 더위로 인해 현장 종료)**

---

## 3. Jetson(실제 오토카)에서 한 것 (2단계: 실기 테스트)

### 3-1. 접속 방식
- USB로 연결 후 SSH로 오토카(`soda@192.168.55.1`)에 터미널 접속

### 3-2. 버그 발견 및 수정: `pop/Pilot.py`
- `from pop import Pilot` 시도 시 `NameError: name 'np' is not defined` 에러
- 원인: 파일 맨 위에 `import numpy as np`가 빠져 있었음 (라이브러리 자체의 버그)
- 해결: `import subprocess as sp` 다음 줄에 `import numpy as np` 추가

### 3-3. YOLOv4-tiny 시도 → 실패 → YOLOv3-tiny로 전환
- 처음엔 `pop/model/yolov4-tiny/`에 있던 `yolov4-tiny.weights`를 OpenCV DNN으로 돌리려 함
- `.cfg`(모델 구조 파일)가 없어서 다운로드 받아 추가
- **문제**: 탐지된 물체 수가 항상 0, 신뢰도도 항상 0
- **원인**: OpenCV 4.3.0은 YOLOv4 계열의 최신 구조(CSP 그룹 연산)를 지원하지 않음 (OpenCV 4.4부터 정식 지원)
- **해결**: 더 단순한 구조라 OpenCV 4.3에서도 확실히 동작하는 **YOLOv3-tiny**로 교체
  - 신뢰도 0.98까지 나오며 정상 작동 확인

### 3-4. 실시간 카메라 연동 (CPU 버전)
- 파일: `jetson_yolo_camera.py`
- CSI 카메라(`Util.gstrmer`) + YOLOv3-tiny(OpenCV DNN) 연결
- 결과: **FPS 6.0** (CPU로만 연산해서 느림)

### 3-5. CUDA 백엔드 시도 → 실패
- `cv2.dnn`에 CUDA 백엔드를 지정했지만 "DNN module was not built with CUDA backend" 에러
- 원인: 이 Jetson에 설치된 OpenCV는 DNN 모듈이 CUDA 지원 없이 빌드되어 있음 (별도로 다시 컴파일해야 하는 부분)

### 3-6. TensorRT 변환 (최종 성공!)
Jetson 전용 고속 실행 엔진(TensorRT)으로 변환하는 작업. 여러 단계의 버전/설정 문제를 하나씩 해결함.

**전체 흐름:**
```
YOLOv3-tiny(다크넷 형식) → ONNX(중간 변환 파일) → TensorRT 엔진(.trt)
```

**겪은 문제와 해결 순서:**

| # | 문제 | 원인 | 해결 |
|---|---|---|---|
| 1 | `onnx==1.4.1` 설치 실패 (`Unknown generator option: dllexport_decl`) | onnx 버전과 protobuf 버전이 안 맞음 | `onnx==1.9.0` + 최신 protobuf로 재설치 |
| 2 | `pycuda` 설치 스크립트에서 `nvcc not found` | CUDA 툴킷 경로가 PATH에 없음 | `~/.bashrc`에 `/usr/local/cuda/bin` 경로 추가 |
| 3 | `pycuda` 설치 스크립트가 `basename` 에러로 중단 | 스크립트 내부 로직 버그 | `pip3 install pycuda`로 직접 설치 |
| 4 | `yolo_to_onnx.py` 실행 시 "file not found" | 파일 이름에 해상도(`-416`)가 안 붙어 있었음 | `yolov3-tiny-416.cfg`/`.weights`로 이름 맞춰 복사 |
| 5 | 플러그인 빌드 시 `libyolo_layer.so` 없음 | `plugins` 폴더에서 빌드(`make`)를 안 함 | `cd plugins && make` 실행 |
| 6 | `make` 실행 시 `unterminated call to function 'shell'` | Makefile 안 정규식에 `#` 문자가 있어서, make가 그 뒤를 "주석"으로 착각 | `grep -Po`(정규식) 대신 `awk`로 대체 |
| 7 | `onnx_to_tensorrt.py` 실행 시 `build_serialized_network` 없음 | 설치된 TensorRT(7.1.3)가 그 함수를 지원하는 8버전보다 낮음 | TensorRT 7 API인 `build_engine`으로 코드 수정 |

**최종 결과:**
- `yolov3-tiny-416.trt` 엔진 파일 생성 성공
- 실시간 카메라 테스트 결과: **FPS 30.1** (카메라 최대 속도까지 도달, 기존 6.0 대비 5배 향상)

---

## 4. 지금까지 만든 파일 목록

### WSL (`Autocar 최종 프로그램/`)
| 파일 | 설명 | 테스트 상태 |
|---|---|---|
| `test_yolo.py` | 정지 이미지 YOLO 테스트 | ✅ 완료 |
| `webcam_yolo.py` | 웹캠 실시간 탐지 + TTS | ✅ 완료 |
| `tts_util.py` | 공용 음성 알림 모듈 | ✅ 완료 |
| `cam_debug.py` | 카메라 포맷 디버깅용 | ✅ 완료 |
| `sensor_fusion.py` | 카메라+라이다 방향/거리 계산 | ⏳ 설계만, Jetson 실기 테스트 필요 |
| `obstacle_avoidance.py` | 장애물 회피 조향 로직 | ⏳ 설계만, Jetson 실기 테스트 필요 (현재 ultralytics 기반이라 Jetson용으로 재작성 필요) |
| `jetson_yolo_test.py` | Jetson용 정지 이미지 테스트 (OpenCV DNN) | ✅ 완료 |
| `jetson_yolo_camera.py` | Jetson용 실시간 카메라 탐지 (OpenCV DNN, CPU) | ✅ 완료, FPS 6.0 |
| `jetson_obstacle_avoidance.py` | Jetson용 장애물 회피 (TensorRT, "정지 후 생각" 방식) | ✅ 실행됨, 물리적 반응속도는 하드웨어 담당 범위 |
| `line_tracing.py` | 라인 트레이싱 (OpenCV 색상 검출) | ⏳ 색상값 미확정 (아래 2-11 참고) |
| `line_calibration.py` | 라인 색상 보정용 (사진 찍어서 마스크 확인) | 🔄 현장 테스트 중, 계속 사용 예정 |
| `color_sample.py` | 선/트랙 바닥의 정확한 HSV 값을 찍어서 비교하는 진단용 | ✅ 작성 완료 |
| `line_tracing_with_avoidance.py` | 라인 트레이싱 + 장애물 회피 통합본 (목표 오프셋 방식) | ⏳ 실기 테스트 필요 |
| `jetson_follow_person.py` | YOLO로 사람 인식해서 따라가는 기능 (TensorRT) | ⏳ 작성 완료, 실기 테스트 필요 |

### Jetson (`~/tensorrt_demos/`, `~/pop/model/`)
| 파일/폴더 | 설명 |
|---|---|
| `~/pop/model/yolov3-tiny/` | YOLOv3-tiny 원본 가중치(cfg, weights, names) |
| `~/tensorrt_demos/` | TensorRT 변환 도구 (오픈소스 clone) |
| `~/tensorrt_demos/yolo/yolov3-tiny-416.onnx` | ONNX 변환 결과 |
| `~/tensorrt_demos/yolo/yolov3-tiny-416.trt` | **최종 TensorRT 엔진 (완성본)** |
| `~/tensorrt_demos/headless_test.py` | TensorRT 엔진 속도 테스트용 (화면 없이 콘솔 출력) |

---

## 5. 겪었던 문제와 해결법 모음 (트러블슈팅 총정리)

여기저기 흩어져 있던 문제/해결을 한 곳에 모았습니다. 나중에 비슷한 문제가 또 생기면 여기부터 찾아보면 됩니다.

### WSL(연습 환경)에서 겪은 문제

| 문제 | 증상 | 원인 | 해결 |
|---|---|---|---|
| 가상환경 생성 실패 | `ensurepip is not available` | `python3-venv` 패키지가 시스템에 없음 | `sudo apt install python3.10-venv` |
| 가상환경 이름 바꾼 후 안 됨 | `ModuleNotFoundError: No module named 'ultralytics'` | `.venv` 폴더만 `mv`로 이름 바꿨더니, 내부 `activate` 스크립트에 옛날 경로(`yolo_env`)가 그대로 남아있었음 | 가상환경을 통째로 삭제하고 새 이름으로 재생성 |
| 웹캠 화면이 초록색으로만 나옴 | 카메라는 열리는데 영상이 이상함 | usbipd로 넘어온 웹캠의 기본 영상 포맷(YUYV)을 OpenCV가 잘못 해석 | `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))`로 포맷 강제 지정 |
| 영상 창이 안 뜨고 에러남 | `Could not find the Qt platform plugin "wayland"` | WSLg가 기본으로 wayland를 쓰는데, OpenCV의 Qt 플러그인은 xcb만 지원 | 실행 시 `QT_QPA_PLATFORM=xcb` 환경변수 지정 |
| 웹캠 권한 없음 | `/dev/video0` 접근 불가 | 현재 계정이 `video` 그룹에 속해있지 않음 | `sudo usermod -aG video $USER` 후 재로그인 (즉시 반영하려면 `sg video -c "명령어"`) |
| 소리가 안 남 | `aplay` 실행 시 `Unknown PCM default` | WSL은 ALSA 사운드카드가 없고 PulseAudio(WSLg)만 있음 | `aplay` 대신 `paplay` 사용 (+ `pulseaudio-utils` 설치) |
| 소리 여전히 안 남 | 명령어는 성공(exit 0)하는데 소리가 안 들림 | 노트북 자체 스피커 음소거 상태였음 | 노트북 볼륨 확인 (당연하지만 은근히 놓치기 쉬움) |
| 카메라 화면과 `q` 키가 멈춤 | 사람 감지 시 화면이 뚝뚝 끊기고 `q`도 안 먹힘 | TTS 음성 재생(`subprocess.run`)이 끝날 때까지 메인 루프 전체가 멈춰버림(동기 처리) | TTS 재생을 `threading.Thread`로 분리해서 화면 루프와 동시에 돌아가게 함 |

### Jetson(실제 오토카)에서 겪은 문제

| 문제 | 증상 | 원인 | 해결 |
|---|---|---|---|
| `pop.Pilot` import 실패 | `NameError: name 'np' is not defined` | `pop/Pilot.py` 파일에 `import numpy as np`가 아예 빠져있던 라이브러리 자체 버그 | 파일 상단에 `import numpy as np` 추가 |
| YOLOv4-tiny 탐지 결과 항상 0개 | 신뢰도(confidence)가 항상 0으로 나옴 | 이 Jetson의 OpenCV(4.3.0)가 YOLOv4 계열의 최신 구조(CSP group 연산)를 지원 안 함 (OpenCV 4.4부터 지원) | 더 단순한 구조인 **YOLOv3-tiny**로 교체 |
| CUDA로 설정해도 그대로 CPU 속도 | `DNN module was not built with CUDA backend; switching to CPU` | 이 Jetson에 설치된 OpenCV는 DNN 모듈이 CUDA 지원 없이 빌드됨 | (OpenCV DNN 경로는 포기하고) TensorRT 변환으로 우회 |
| `onnx==1.4.1` 설치 실패 | `Unknown generator option: dllexport_decl` | onnx 1.4.1과 새로 설치된 protobuf 버전이 서로 안 맞음 | `onnx==1.9.0`(README에 명시된 정확한 버전) + 최신 protobuf로 재설치 |
| `pycuda` 설치 스크립트 실패 (1) | `ERROR: nvcc not found` | CUDA 툴킷(`/usr/local/cuda/bin`)이 PATH에 없음 | `~/.bashrc`에 PATH, LD_LIBRARY_PATH 추가 |
| `pycuda` 설치 스크립트 실패 (2) | `basename: extra operand` 후 스크립트 중단 | 설치 스크립트 내부의 버전 감지 로직 버그(`set -e`라 바로 중단됨) | 스크립트 대신 `pip3 install pycuda`로 직접 설치 |
| `yolo_to_onnx.py` 실행 실패 | `ERROR: file (yolov3-tiny-416.cfg) not found!` | 변환 도구는 파일 이름에 해상도(`-416`)가 붙어있길 기대하는데, 원본 파일 이름엔 없었음 | `yolov3-tiny-416.cfg`/`.weights`로 이름 맞춰서 복사 |
| 플러그인 로드 실패 | `failed to load ../plugins/libyolo_layer.so` | `plugins` 폴더의 C++ 코드를 아직 빌드(`make`)하지 않음 | `cd plugins && make` 실행 |
| `make` 자체가 실패 | `unterminated call to function 'shell': missing ')'` | Makefile 안에 있는 정규식(`#define ...`)의 `#` 문자를, make가 "주석 시작"으로 잘못 해석해서 그 뒤 내용을 통째로 무시함 | `grep -Po` 정규식 대신 `#`이 필요 없는 `awk` 방식으로 해당 줄을 수정 |
| `onnx_to_tensorrt.py` 실행 실패 | `'tensorrt.tensorrt.Builder' object has no attribute 'build_serialized_network'` | 이 함수는 TensorRT 8 이상 API인데, 설치된 TensorRT는 7.1.3 버전이라 없음 | 코드를 TensorRT 7 방식인 `builder.build_engine(network, config)`로 수정 |
| `trt_yolo.py` 실행 실패 | `failed to load ./plugins/libyolo_layer.so` | 상대 경로(`./plugins`) 문제로, 엉뚱한 위치(홈 디렉터리)에서 실행함 | `tensorrt_demos` 폴더 안으로 `cd`한 뒤 실행 |
| GUI 창을 못 띄움 | `trt_yolo.py`가 화면(cv2.imshow)을 띄우려 함 | SSH 터미널 접속이라 디스플레이(화면 출력 장치)가 없음 | 화면 없이 콘솔에 결과와 FPS만 출력하는 스크립트(`headless_test.py`)를 따로 작성 |
| 실제 차량 반응속도가 너무 느림 | 판단(콘솔 로그)은 빠른데 실제 조향/정지가 굼뜸 | 매 프레임(초당 30번)마다 같은 조향/속도값을 계속 다시 명령해서, `Pilot.AutoCar`의 점진적 가속/조향(ramping) 로직이 계속 리셋되는 것으로 추정 | 1차: 상태가 바뀔 때만 명령 전송하도록 수정 → 그래도 동일 → 2차: "정지 후 생각"(멈춰서 방향 판단 후 짧게 이동) 방식으로 설계 자체를 변경 (`jetson_obstacle_avoidance.py`) |

**패턴으로 보면:** Jetson 쪽 문제들은 대부분 "**2020년에 나온 구버전 소프트웨어(JetPack 4.3)** vs **최근 만들어진 변환 도구/라이브러리**" 사이의 버전 불일치에서 비롯됐습니다. 앞으로도 새 라이브러리를 설치할 때는 항상 "이 Jetson이 몇 년도 소프트웨어인지"를 염두에 두고, 버전을 낮춰서 맞추는 방식으로 접근하면 됩니다.

---

## 6. 다음에 할 일 (TODO)

- [x] ~~`obstacle_avoidance.py`를 TensorRT 기반으로 다시 작성~~ → `jetson_obstacle_avoidance.py` 완료, 실행됨
- [x] ~~라인 트레이싱 코드 작성~~ → `line_tracing.py`, `line_tracing_with_avoidance.py` 완료 (실기 테스트 전)
- [x] ~~사람 따라가기 코드 작성~~ → `jetson_follow_person.py` 완료 (실기 테스트 전)
- [ ] **`jetson_follow_person.py` 실기 테스트** (색상 보정 불필요라 바로 테스트 가능) ← 다음 우선순위
- [ ] `line_calibration.py`로 실제 트랙 색상 보정 재시도 (오늘 시도했으나 강한 역광으로 미완료, 흐린 날/저녁 + 차양막 준비해서 재시도 필요 — 2-10 참고)
- [ ] `line_tracing.py` 단독으로 먼저 라인 추적 테스트 (장애물 없이)
- [ ] `line_tracing_with_avoidance.py`로 장애물 회피 + 라인 복귀 통합 테스트
- [ ] 라이다를 메인 안전 센서로 실제 연결 및 테스트 (`sensor_fusion.py` 기반)
- [ ] `sensor_fusion.py`를 Jetson에서 실측하며 카메라 화각(FOV), 라이다 각도 오프셋 보정
- [ ] 모든 기능(카메라+YOLO+라이다+라인트레이싱+TTS)을 하나의 최종 실행 프로그램으로 통합
- [ ] daemon(systemctl) 등록해서 전원 켜면 자동 실행되게 설정

---

*이 문서는 진행 상황에 따라 계속 업데이트됩니다.*
