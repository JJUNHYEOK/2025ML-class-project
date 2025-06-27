# **2025 GNU ML Project - 인공지능 기반 산불 대응 솔루션**
![Image](https://gijun.notion.site/image/attachment%3A777fa5bd-03a7-4c8d-a01f-9899cb00eff8%3AChatGPT_Image_2025%EB%85%84_5%EC%9B%94_24%EC%9D%BC_%EC%98%A4%EC%A0%84_10_28_12.png?table=block&id=1fde7e8d-fc4a-8001-b258-f03de2dd3bf7&spaceId=beb5f5c9-5ecd-4100-95bc-79cc3c53ea0e&width=1420&userId=&cache=v2)

![Image](title.png)
**25.06.27~


## **👨‍🏫 프로젝트 소개**
Python 기반 컴퓨터 비전과 최적화 알고리즘을 활용한 실시간 산불감지·자원배치 및 시각화 시스템 개발  
**Artificial Intelligence-Based Wildfire Response Solution (AIWRS)**

![Image](https://gijun.notion.site/image/attachment%3A29265e95-2cc3-4a6e-a159-de5ab6fff609%3A%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2025-05-22_152427.png?table=block&id=1fbe7e8d-fc4a-8050-a07b-d0616ffcaff1&spaceId=beb5f5c9-5ecd-4100-95bc-79cc3c53ea0e&width=1420&userId=&cache=v2)
![Image](https://gijun.notion.site/image/attachment%3A440094bc-5812-4661-ba0b-0d8e8e8e470e%3A%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2025-05-22_152432.png?table=block&id=1fbe7e8d-fc4a-80fe-bcbc-fd2abd458059&spaceId=beb5f5c9-5ecd-4100-95bc-79cc3c53ea0e&width=1420&userId=&cache=v2)
![Image](https://gijun.notion.site/image/attachment%3A579e725a-2fab-4fb2-b1ba-6f3f1543edf0%3A%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2025-05-22_152436.png?table=block&id=1fbe7e8d-fc4a-8019-a2c9-db1050b2393d&spaceId=beb5f5c9-5ecd-4100-95bc-79cc3c53ea0e&width=1420&userId=&cache=v2)

## **⏲️ 개발 기간**

- 2025.05.12 ~ 2025.06.09
- 05.12 : 주제발표
- 05.26 : 중간발표
- 06.09 : 최종발표

## **🧑‍🤝‍🧑 개발 인원**

- 문기준 : 자원 분배 최적화 모델 - 최적화 모델 개발
- 고성윤 : 자원 분배 최적화 모델 - 지적정보 활용 개발
- 정준혁 : 객체인식 모델 학습 및 튜닝
- 손영준 : GUI 어플리케이션 개발
- 이현주 : 모델 성능 검증 및 테스팅

## **💻 개발환경**
- Version : Python 3.10
- IDE : Visual Studio Code, Colab (for ML)

## **⚙️ 기술 스택**
- OpenCV
- YOLO
- Keras
- SciPy
- PyQt5
- Streamlit

## **📝 프로젝트 아키텍쳐**
전체 시스템 구성

- 데이터 수집·전처리
- AI 추론 엔진
- 결과 분석·후처리
- 데스크톱 애플리케이션 (PyQt5)


1. 데이터 수집·전처리 계층

- **Frame Grabber (OpenCV)**
    - 카메라·동영상 스트림 캡처
    - 프레임 단위로 추출 및 타임스탬프 부여
- **Preprocessing (SciPy + OpenCV)**
    - 노이즈 제거 (Gaussian, Median filter)
    - 색상 변환(Grayscale, HSV)
    - 해상도 조정·정규화


2. AI 추론 엔진

- **YOLO 모델**
    - Keras(TensorFlow 백엔드)로 구현한 YOLOv5/v7
    - 사전 학습된 가중치 로딩 및 사용자 데이터로 파인튜닝
- **Inference Service**
    - 전처리된 프레임을 배치 단위로 모델에 전달
    - 객체 감지 결과(Bounding box, Confidence) 반환


3. 결과 분석·후처리 계층

- **Postprocessing (SciPy)**
    - 비최대 억제(NMS)
    - Confidence 임곗값 필터링
    - 객체 추적(추가 필요 시 칼만 필터 등 적용)
- **메트릭·로그 수집**
    - 추론 지연시간, 프레임 처리율 등 퍼포먼스 지표 수집
    - 감지된 객체 통계 집계 (시간·위치별 카운트)
- **수학적 최적화 모델**
    - 수식 번역


4. 데스크톱 애플리케이션 (PyQt5)

- **UI 레이아웃**
    - 실시간 영상 뷰어
    - 감지 박스·레이블 오버레이
    - 퍼포먼스 차트(Graph) 및 로그 뷰
- **컨트롤 패널**
    - 모델 파라미터(임곗값, NMS) 실시간 조정
    - 스트림 소스 변경, 녹화 시작·정지 버튼
- **모듈 인터페이스**
    - 내부 스레드로 인퍼런스 엔진 호출
    - 결과를 Qt Signal/Slot 으로 화면 갱신


## **📌 주요 기능**
- 산불 감지
- 산불 예측
- 자원 분석
- 자원 분배
- 어플리케이션 시각화

## **사용 방법**
1. requirements.txt 설치
```bash
pip install -r requirements.txt
```


2. WSQ000301.csv 포함
```
datasets/WSQ000301.csv
```

**공공데이터이므로 공공데이터 포털에서 다운 받는다.*
[공공산림데이터](https://www.bigdata-forest.kr/product/WSQ000301).


3. run.py의 BOOL_DEBUG를 False로 수정한다.  
**만약 경로 문제가 발생하는 경우 BOOL_DEBUG를 True로 변경하십시오.**


4. 실행
```bash
python run.py
```
