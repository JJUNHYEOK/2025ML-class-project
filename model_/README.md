# 🔥 Wildfire Detection using YOLOv8 🚒

YOLOv8을 기반으로 산불(화재) + 연기 이미지를 감지하는 모델을 개발한다.

---

## 진행 현황

### 1. 📊 데이터 준비
- `wildfire-detection` 깃허브 레포지토리의 `training-recipes` 기반으로 프로젝트 구조 구성
- 학습용 이미지/라벨 데이터 준비 완료
- 학습 데이터 경로: `/home/jjh/ML/D-Fire`
  <tr><td>
    
  | Category | # Images |
  | ------------- | ------------- |
  | Only fire  | 1,164  |
  | Only smoke  | 5,867  |
  | Fire and smoke  | 4,658  |
  | None  | 9,838  |

  </td><td>
---

### 2. 📝 모델 학습
- 모델: YOLOv8s (`ultralytics` 사용)
- 학습 스크립트: `wildfire_training.py`
- 주요 설정:
  - Epochs: 200
  - Batch Size: 64
  - Image Size: 640
  - Early Stopping: 20 epochs
  - Optimizer: Adam
- 학습 환경: 연구실 리눅스 서버, Python 가상환경 (venv)
- 학습 실행: `tmux`를 통해 장시간 학습 안정적으로 실행
- 학습 모델 : 차후 HuggingFace를 통해 업로드 예정


**1차 튜닝**
- 모델 : 최초의 학습을 거친 best.pt
- 주요 설정 : 기존 데이터셋에서 background 클래스가 더욱 차이나게 드러날 수 있도록 이미지 전처리 수행(색상 반전)
![preprocessed dataset example](pic/preprocessed/train_batch1.jpg)
- 이외의 설정은 동일하게 진행함.


**2차 튜닝**
- 모델 : best.pt
- 주요 설정 : hyperparameter 조정
- 세부사항:
  - hsv_h = 0.005 (Hue Augmentation, 모델이 특정 색조에 과도하게 의존하는 것 방지)
  - hsv_s = 0.6 (Saturation Augmentation, 채도를 무작위 증강시켜 이미지 선명도 증가)
  - hsv_v = 0.3 (Value Augmentation, 명도를 무작위 증강시켜 이미지 강인성 증가)
  - fliplr = 0.7 (Horizontal Flip Probability, 기존 이미지셋을 0.7의 확률로 좌우 반전시켜 이미지 민감도 하향 조정)
- 이외의 설정은 동일



---


### 3. 모델 예측 및 평가
- label이 존재하는 데이터셋과 비교
- 전체적으로 균형잡힌 화재/연기 탐지 및 식별 가능
- 기존에 우려했던 석양, 그림자에 대한 오탐 사례가 다소 식별됨.

**Initial Ver**

![predict dataset으로 예측한 결과](pic/initial/val_batch0_pred.jpg)

---

![label dataset의 실제 결과](pic/initial/val_batch0_labels.jpg)

- 자원 투입 및 분배 시점에 대한 기준 : 확률값 or 화재 및 연기 객체 인식 개수 ? --> 논의해봐야함.


**Modified Hyperparameter Ver**

![predict dataset으로 예측한 결과](pic/hyp_fix/val_batch0_pred.jpg)


---


**원본 혼동 행렬(Confusion Matrix) 해석**
- 본 모델 예측 및 평가를 통해 두 개의 혼동 행렬이 생성되었는데, 원본과 정규화된 형태로 구성되어있다.

**1. Initial Ver**

![원본 혼동 행렬](matrix/initial/confusion_matrix.png)

- 이 행렬은 각 클래스에 대해 모델이 얼마나 정확하게 예측했는지 실제 개수를 보여준다.
- 총 샘플 수 : 3116


**2. Preprocessed Ver**

![원본 혼동 행렬](matrix/preprocesssed/confusion_matrix.png)

- False Negative가 대부분, 정상적인 튜닝이 아님을 확인 가능


**3. Modified Hyperparameter Ver**

![원본 혼동 행렬](matrix/hyp_fix/confusion_matrix.png)


---

**클래스별 성능 분석**

**1. Initial Ver**

- **True `smoke` (실제 연기)**
- 올바른 예측 : 780개 / `fire`로 잘못 예측 : 2개 / `background`로 잘못 예측 : 157개
- 실제 `smoke` 샘플 수 : 939개

- **True `fire` (실제 화재)**
- 올바른 예측 : 899개 / `smoke`로 잘못 예측 : 321개 / `background`로 잘못 예측 : 321개
- 실제 `fire`샘플 수 : 1604개

- **True `background` (실제 배경)**
- 올바른 예측 : -개 / `smoke`로 잘못 예측 : 0개 / `background`로 잘못 예측 : 573개

**주요 관찰**

1. `smoke` 예측 : 실제 `smoke`를 `smoke`로 잘 예측하지만, `background`로 잘못 예측하는 경우도 꽤 존재.

2. `fire` 예측 : 실제 `fire`를 가장 잘 예측. 하지만 `smoke`로 384개, `background`로 321개를 잘못 예측하는 경우 존재. 특히 `fire`를 `smoke`로 잘못 예측하는 것은 오경보에 따른 자원 낭비로 이어질 수 있다.


**2. Modified Hyperparameter Ver**

- **True `smoke` (실제 연기)**
- 올바른 예측 : 767개 / `fire`로 잘못 예측 : 2개 / `background`로 잘못 예측 : 453개
- 실제 `smoke` 샘플 수 : 939개

- **True `fire` (실제 화재)**
- 올바른 예측 : 737개 / `smoke`로 잘못 예측 : 2개 / `background`로 잘못 예측 : 200개
- 실제 `fire`샘플 수 : 1604개

---

**정규화 혼동 행렬(Confusion Matrix_normalized) 해석**
- 본 모델 예측 및 평가를 통해 두 개의 혼동 행렬이 생성되었는데, 원본과 정규화된 형태로 구성되어있다.

**1. Initial Ver**

![정규화 혼동 행렬](matrix/initial/confusion_matrix_normalized.png)

**2. Modified Hyperparameter Ver**

![정규화 혼동 행렬](matrix/hyp_fix/confusion_matrix_normalized.png)

- 이 행렬은 각 행의 값이 1이 되도록 정규화한 값인 재현율(Recall)을 보여준다.




**재현율 분석**

**1. Initial Ver**

- **True `smoke` (실제 연기)**
- 모델이 `smoke`로 예측 : 0.83(83%)
- 모델이 `fire`로 예측 : 0에 수렴
- 모델이 `background`로 예측 : 0.17 -> 실제 `smoke` 샘플 중 17%를 `background`로 잘못 예측

- **True `fire` (실제 화재)**
- 모델이 `smoke`로 예측 : 0.26(26%)
- 모델이 `fire`로 예측 : 0.73(73%)
- 모델이 `background`로 예측 : 0.20 -> 실제 `smoke` 샘플 중 20%를 `background`로 잘못 예측

-**True `background` (실제 배경)**
- 모델이 `smoke`로 예측 : 0에 수렴
- 모델이 `fire`로 예측 : 0.60(60%)
- 모델이 `background`로 예측 : 0.40


**2. Modified Hyperparameter Ver**

- **True `smoke` (실제 연기)**
- 모델이 `smoke`로 예측 : 0.63(63%)
- 모델이 `fire`로 예측 : 0에 수렴
- 모델이 `background`로 예측 : 0.37 -> 실제 `smoke` 샘플 중 37%를 `background`로 잘못 예측

- **True `fire` (실제 화재)**
- 모델이 `smoke`로 예측 : 0에 수렴
- 모델이 `fire`로 예측 : 0.78(78%)
- 모델이 `background`로 예측 : 0.35 -> 실제 `smoke` 샘플 중 35%를 `background`로 잘못 예측

-**True `background` (실제 배경)**
- 모델이 `smoke`로 예측 : 0.65(65%)
- 모델이 `fire`로 예측 : 0.35(35%)

---


**종합적 해석**

**1. Initial Ver**

1. **`smoke` 클래스 예측 성능**
- 83%의 높은 재현율로 잘 찾아냄
- 실제 `smoke`의 17%를 miss

2. **`fire` 클래스 예측 성능**
- 73%의 재현율. 준수하나 개선의 여지 있음.
- 가장 큰 문제는 실제 `fire`의 26%를 `smoke`로 잘못 예측함.

3. **`background` 클래스 예측 성능**
- 40%의 재현율, 비교적 낮은 성능
- 실제 `background`의 60%를 `fire`로 오인함.


따라서, `background` 클래스에서 개선의 여지가 있음. 차후 모델 개선에서 가장 신경써야 할 부분으로 보임.

**2. Modified Hyperparameter Ver**

- 개선된 부분:
  - `fire` 클래스에 대한 True Positive(TP) 증가
  - `background`가 `fire`로 오인되는 확률 감소

- 저해된 부분:
  - 각 클래스마다의 본래 예측 성능 감소


`background` 클래스 예측 성능 개선에 포커싱을 해서 그런지, fire와 smoke 클래스에 대해 다소 예측 성능이 저하되는 문제 발생.














---




### 차후 과제

- 모델과 최적화 함수 및 시각화 서비스와의 파이프라인 구성
- 모델 성능 향상이 필요한가 ? -> 필요하다면 학습전략 구축
- 각종 정량지표 이용 결과 해석 및 보완점 탐색
- 모델 튜닝을 더 진행할 것인가? -> 하지 않는다면 어떤 모델을 사용할 것인지?


