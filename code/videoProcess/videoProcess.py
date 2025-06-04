import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from torchvision.models import mobilenet_v3_small
import torch
from ultralytics import YOLO

MAX_PREV_FRAMES = 5  # 저장할 이전 프레임 수 제한
prev_frames = []  # 이전 프레임들을 저장할 리스트

# YOLO 모델 로드
#fire_model = YOLO('model_/best.pt')  # 실제 인식 모델을 가져오셈

# 배경 제거기 객체를 전역 변수로 생성
bg_subtractor_obj = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=50, 
    detectShadows=True
)

def frame_difference(frame, prev_frames):
    if not prev_frames:
        return np.zeros_like(frame)
    return cv2.absdiff(frame, prev_frames[-1])

def fire_color_detection_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 화재 색상 범위 정의 (빨간색 계열)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    return cv2.bitwise_or(mask1, mask2)

def fire_color_detection_ycrcb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # 화재 색상 범위 정의 (YCrCb 공간)
    lower_fire = np.array([0, 133, 77])
    upper_fire = np.array([255, 173, 127])
    return cv2.inRange(ycrcb, lower_fire, upper_fire)

def glcm_analysis(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # GLCM 계산
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    # GLCM 특징 추출
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # 1차원 특징을 2차원 마스크로 변환
    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    texture_mask = np.zeros_like(gray, dtype=np.float32)
    texture_mask[gray > 128] = np.mean(features)  # 임계값 기반으로 마스크 생성
    return texture_mask

def combine_evidences(evidences):
    weights = [0.3, 0.3, 0.3, 0.05, 0.05]  # temporal_diff, hsv_mask, ycrcb_mask, texture_features, bg_mask
    combined = np.zeros_like(evidences[0], dtype=np.float32)
    
    for evidence, weight in zip(evidences, weights):
        if isinstance(evidence, np.ndarray):
            # 2차원 마스크를 3차원으로 확장
            if len(evidence.shape) == 2:
                evidence = np.stack([evidence] * 3, axis=-1)
            combined += weight * evidence.astype(np.float32)
    
    return np.clip(combined, 0, 255).astype(np.uint8)

def bg_subtractor(frame):
    return bg_subtractor_obj.apply(frame)

# 멀티스케일 접근법
def preprocessing(frame):
    # 이전 프레임 관리
    global prev_frames
    prev_frames.append(frame)
    if len(prev_frames) > MAX_PREV_FRAMES:
        prev_frames.pop(0)
    
    #시간적 일관성 확인
    temporal_diff = frame_difference(frame, prev_frames)
    
    #배경 추정
    bg_mask = bg_subtractor(frame)
    
    #다중 색상공간 활용
    hsv_mask = fire_color_detection_hsv(frame)
    ycrcb_mask = fire_color_detection_ycrcb(frame)
    
    #텍스처 분석 추가
    texture_features = glcm_analysis(frame)
    
    # 모든 마스크를 3채널로 변환
    temporal_diff = cv2.cvtColor(temporal_diff, cv2.COLOR_GRAY2BGR) if len(temporal_diff.shape) == 2 else temporal_diff
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) if len(bg_mask.shape) == 2 else bg_mask
    hsv_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR) if len(hsv_mask.shape) == 2 else hsv_mask
    ycrcb_mask = cv2.cvtColor(ycrcb_mask, cv2.COLOR_GRAY2BGR) if len(ycrcb_mask.shape) == 2 else ycrcb_mask
    texture_features = cv2.cvtColor(texture_features.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(texture_features.shape) == 2 else texture_features
    
    return combine_evidences([temporal_diff, hsv_mask, ycrcb_mask, texture_features, bg_mask])

# 간단한 이진 분류기 (화재/비화재)
confidence_classifier = mobilenet_v3_small(pretrained=True)
confidence_classifier.classifier[3] = torch.nn.Linear(1024, 2)  # 출력층을 2개 클래스로 수정

def visualize_fire_detection(image, detection_result):
    # 원본 이미지 복사
    vis_image = image.copy()
    
    # 화재 감지 결과 시각화
    for result in detection_result:
        boxes = result.boxes
        for box in boxes:
            # 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 신뢰도 점수
            conf = float(box.conf[0])
            
            if conf > 0.2:
                # 빨간색 박스 그리기
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 텍스트 표시
                label = f'Fire: {conf:.2f}'
                cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 화재 영역 강조
                fire_region = vis_image[y1:y2, x1:x2]
                # 빨간색 채널 강화
                #fire_region[:, :, 2] = np.clip(fire_region[:, :, 2] * 1.5, 0, 255)
                vis_image[y1:y2, x1:x2] = fire_region
    
    return vis_image

# 메인 실행 코드
#image = cv2.imread('code/videoProcess/image.jpg')
#image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# 화재 감지 수행
#detection_result = fire_model(image)

# 전처리 수행
#processed = preprocessing(image)

# 화재 감지 결과 시각화
#visualized = visualize_fire_detection(image, detection_result)

# 결과 표시
#cv2.imshow('Original', image)
#cv2.imshow('Processed', processed)
#cv2.imshow('Fire Detection', visualized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

