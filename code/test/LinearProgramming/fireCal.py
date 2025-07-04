import pandas as pd
from sklearn.impute import SimpleImputer
import pulp
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import random
from typing import Dict, List, Tuple
from sklearn.impute import KNNImputer
import math

# 1. 머신러닝 모델 로드 (예시)
from sklearn.ensemble import RandomForestRegressor
import joblib

# 모델 파일 경로 설정
MODEL_PATH = 'fire_demand_model.joblib'  # 실제 모델 파일 경로로 변경

def load_ml_model(model_path: str):
    """학습된 머신러닝 모델 로드"""
    try:
        model = joblib.load(model_path)
        print("머신러닝 모델 로드 성공")
        return model
    except Exception as e:
        print(f"머신러닝 모델 로드 실패: {e}")
        return None

# 2. 데이터 로드 및 전처리 (기존 코드와 동일)
def load_and_preprocess_data():
    # 데이터 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # 가능한 모든 경로 시도
    possible_paths = [
        os.path.join(project_root, 'datasets', 'WSQ000301.csv'),  # 프로젝트 루트 기준
        os.path.join(current_dir, 'datasets', 'WSQ000301.csv'),   # 현재 디렉토리 기준
        os.path.join(current_dir, '..', 'datasets', 'WSQ000301.csv'),  # 상위 디렉토리 기준
        os.path.join(current_dir, '..', '..', 'datasets', 'WSQ000301.csv'),  # 상위 상위 디렉토리 기준
        'datasets/WSQ000301.csv',  # 상대 경로
        '../datasets/WSQ000301.csv',  # 상대 경로
        '../../datasets/WSQ000301.csv'  # 상대 경로
    ]
    
    # 각 경로 시도
    for path in possible_paths:
        print(f"시도 중인 경로: {path}")
        if os.path.exists(path):
            print(f"데이터 파일을 다음 경로에서 찾았습니다: {path}")
            try:
                # CSV 데이터 로드
                df = pd.read_csv(path, encoding='UTF-8')
                
                # 데이터프레임 복사본 생성
                df = df.copy()
                
                # 실제 존재하는 컬럼 선택
                features = df[['WDSP', 'FRTP_CD', 'DNST_CD', 'HMDT', 'DMCLS_CD']].copy()
                target = df[['POTFR_RSRC_INPT_QNTT', 'FRFR_DMG_AREA']].copy()

                # 컬럼 이름 변경
                features.columns = ['wind_speed', 'fuel_type', 'slope', 'humidity', 'damage_class']
                target.columns = ['required_resources', 'damage_area']

                # 수치형 데이터와 범주형 데이터 분리
                numeric_features = features[['wind_speed', 'humidity']].copy()
                categorical_features = features[['fuel_type', 'damage_class']].copy()

                # 결측치 처리 전 데이터 타입 변환
                # 수치형 데이터 변환
                for col in numeric_features.columns:
                    numeric_features[col] = pd.to_numeric(numeric_features[col], errors='coerce')
                
                # 범주형 데이터 변환
                for col in categorical_features.columns:
                    categorical_features[col] = categorical_features[col].astype(str)

                # 결측치 처리
                # 수치형 데이터: KNN imputation 사용
                numeric_imputer = KNNImputer(n_neighbors=5)
                numeric_imputed = pd.DataFrame(
                    numeric_imputer.fit_transform(numeric_features),
                    columns=numeric_features.columns
                )

                # 범주형 데이터: 최빈값으로 채우기
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                categorical_imputed = pd.DataFrame(
                    categorical_imputer.fit_transform(categorical_features),
                    columns=categorical_features.columns
                )

                # 범주형 데이터 원-핫 인코딩
                categorical_encoded = pd.get_dummies(categorical_imputed, prefix=['fuel', 'damage'])

                # 타겟 데이터 처리
                # required_resources: KNN imputation 사용
                target['required_resources'] = pd.to_numeric(target['required_resources'], errors='coerce')
                target_imputer = KNNImputer(n_neighbors=5)
                target_imputed = pd.DataFrame(
                    target_imputer.fit_transform(target[['required_resources']]),
                    columns=['required_resources']
                )
                
                # damage_area: 0으로 채우기 (피해 면적이 없는 경우)
                target['damage_area'] = pd.to_numeric(target['damage_area'], errors='coerce')
                target['damage_area'] = target['damage_area'].fillna(0)

                # 데이터 정제 및 병합
                numeric_imputed.reset_index(drop=True, inplace=True)
                categorical_encoded.reset_index(drop=True, inplace=True)
                target_imputed.reset_index(drop=True, inplace=True)
                target.reset_index(drop=True, inplace=True)

                # slope 데이터 처리 (기본값 0으로 설정)
                slope_data = pd.DataFrame({'slope': [0] * len(numeric_imputed)})
                
                # 모든 특성 결합
                features_processed = pd.concat([numeric_imputed, slope_data, categorical_encoded], axis=1)
                target_processed = pd.concat([target_imputed, target['damage_area']], axis=1)
                
                # 데이터 정제 (NaN 값 제거)
                combined_df = pd.concat([features_processed, target_processed], axis=1)
                combined_df_cleaned = combined_df.dropna()
                
                features_processed = combined_df_cleaned.drop(['required_resources', 'damage_area'], axis=1)
                target_processed = combined_df_cleaned[['required_resources', 'damage_area']]

                return features_processed, target_processed
                
            except Exception as e:
                print(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
                continue
    
    print("데이터 파일을 찾을 수 없습니다. 다음 경로들을 확인해주세요:")
    for path in possible_paths:
        print(f"- {path}")
    return None, None

# 3. 시나리오 생성 (기존 방식 유지)
def generate_scenarios_from_data(features_processed, target_processed, n_scenarios=5):
    """
    실제 산불 데이터를 기반으로 K-means를 사용하여 시나리오를 생성
    """
    from sklearn.cluster import KMeans
    
    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_processed)
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=min(5, len(features_processed)), random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # 클러스터별 특성 계산
    cluster_stats = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_features = features_processed[cluster_mask]
        cluster_targets = target_processed[cluster_mask]
        
        # 기본 통계 계산
        stats = {
            'wind_speed': float(cluster_features['wind_speed'].mean()),
            'slope': float(cluster_features['slope'].mean() * 45),
            'humidity': float(cluster_features['humidity'].mean() * 20 + 50),
            'required_resources': float(cluster_targets['required_resources'].mean()),
            'damage_area': float(cluster_targets['damage_area'].mean()),
            'probability': len(cluster_features) / len(features_processed)
        }
        
        # 연료 유형 계산
        fuel_cols = [col for col in cluster_features.columns if col.startswith('fuel_')]
        if fuel_cols:
            fuel_counts = cluster_features[fuel_cols].sum()
            max_fuel_col = fuel_counts.idxmax()
            try:
                # 컬럼 이름에서 숫자만 추출
                fuel_type = int(''.join(filter(str.isdigit, max_fuel_col)))
                stats['fuel_type'] = fuel_type
            except ValueError:
                stats['fuel_type'] = 1  # 기본값
        else:
            stats['fuel_type'] = 1  # 기본값
            
        # 피해 등급 계산
        damage_cols = [col for col in cluster_features.columns if col.startswith('damage_')]
        if damage_cols:
            damage_counts = cluster_features[damage_cols].sum()
            max_damage_col = damage_counts.idxmax()
            try:
                # 컬럼 이름에서 숫자만 추출
                damage_class = int(''.join(filter(str.isdigit, max_damage_col)))
                stats['damage_class'] = damage_class
            except ValueError:
                stats['damage_class'] = 1  # 기본값
        else:
            stats['damage_class'] = 1  # 기본값
        
        # 값 범위 조정
        stats['wind_speed'] = max(0, min(20, stats['wind_speed'] * 5))
        stats['slope'] = max(0, min(30, stats['slope']))
        stats['humidity'] = max(20, min(80, stats['humidity']))
        stats['required_resources'] = max(1, min(10, int(stats['required_resources'])))
        stats['damage_area'] = max(0, min(1000, stats['damage_area']))
        
        cluster_stats[cluster_id] = stats
    
    # 시나리오 생성
    scenarios = []
    for cluster_id, stats in cluster_stats.items():
        scenario = FireScenario(
            scenario_id=len(scenarios),
            probability=stats['probability'],
            cluster_stats=stats
        )
        scenarios.append(scenario)
    
    return scenarios

# 4. FireScenario 클래스 수정
class FireScenario:
    def __init__(self, scenario_id: int, probability: float, cluster_stats: Dict):
        self.id = scenario_id
        self.probability = probability
        self.cluster_stats = cluster_stats
        # 기준 소방서 위치 설정 (진주소방서)
        self.base_station = {
            'name': '기준 소방서',
            'latitude': 35.18035823746264,  # 위도
            'longitude': 128.11851962302458  # 경도
        }
        # 머신러닝 모델 로드
        self.model = load_ml_model(MODEL_PATH)
        self.sites = self._generate_sites_from_stats()
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 지점 간의 거리를 계산 (Haversine 공식 사용)"""
        R = 6371  # 지구의 반경 (km)
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _generate_sites_from_stats(self) -> Dict[str, Dict]:
        """클러스터 통계를 기반으로 현실적인 화재 발생 지점 생성"""
        num_sites = max(1, min(3, int(self.cluster_stats['required_resources'] / 3)))  # 1-3개 지점
        sites = {}
        
        for i in range(num_sites):
            site_id = f'site{i+1}'
            
            # 화재 발생 지점의 위도/경도 생성 (경상남도 내에서 랜덤하게)
            site_lat = random.uniform(35.28450887192325, 35.10468233527785)
            site_lon = random.uniform(128.01212832039607, 128.18678592428446)
            
            # 기준 소방서로부터의 실제 거리 계산
            base_distance = self._calculate_distance(
                self.base_station['latitude'],
                self.base_station['longitude'],
                site_lat,
                site_lon
            )
            
            # 모든 트럭 타입에 대한 거리 정보 생성
            distances = {}
            for truck_type in ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']:
                # 트럭 타입별로 거리 변동 추가 (실제 거리에서 ±10% 변동)
                distance_variation = base_distance * random.uniform(-0.1, 0.1)
                distances[truck_type] = max(2, min(30, base_distance + distance_variation))

            # **[수정됨] 머신러닝 모델을 사용하여 수요 예측**
            if self.model:
                # 모델 입력을 위한 데이터 준비
                input_data = pd.DataFrame([{
                    'wind_speed': self.cluster_stats['wind_speed'],
                    'fuel_type': self.cluster_stats['fuel_type'],
                    'slope': self.cluster_stats['slope'],
                    'humidity': self.cluster_stats['humidity'],
                    'damage_class': self.cluster_stats['damage_class'],
                    'latitude': site_lat,
                    'longitude': site_lon
                }])
                # 필요에 따라 입력 데이터 스케일링
                #input_data = scaler.transform(input_data)  # scaler는 모델 학습 시 사용한 스케일러

                # 수요 예측
                predicted_demand = self.model.predict(input_data)[0]
                demand = max(1, min(5, int(predicted_demand)))  # 1-5명으로 제한
            else:
                # 모델이 없을 경우 기본값 사용
                demand = 3  # 기본값

            sites[site_id] = {
                'demand': demand,
                'distance': distances,
                'risk_factors': {
                    'wind_speed': self.cluster_stats['wind_speed'],
                    'fuel_type': self.cluster_stats['fuel_type'],
                    'slope': self.cluster_stats['slope'],
                    'humidity': self.cluster_stats['humidity'],
                    'damage_class': self.cluster_stats['damage_class']
                },
                'latitude': site_lat,
                'longitude': site_lon
            }
        return sites

# 5. 나머지 클래스 및 main 함수 (기존 코드와 동일)
class ResourceAllocator:
    def __init__(self):
        """
        진주소방서
        소방차량 계 : 52대 (진압가능차량: 17대, 특수차량: 5대, 구급차: 12대)
        소방공무원 계 : 338명
        """
        self.truck_types = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
        self.firefighter_types = ['FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6']
        
        # 자원 특성 정의
        """
        펌프차량 특성
        2800 ~ 3000L 의 용량
        4~6km/L 연비
        36.8km/h 속도
        4인 탑승정원
        1회 출동시 30만원 소요 (국민 부담금 X - 필수인프라)
        """
        self.truck_capabilities = {
            'FT1': {'capacity': 2800, 'personnel': 4, 'fuel_efficiency': 4, 'speed': 36.8, 'cost': 300000},
            'FT2': {'capacity': 3000, 'personnel': 4, 'fuel_efficiency': 5, 'speed': 36.8, 'cost': 300000},
            'FT3': {'capacity': 2900, 'personnel': 4, 'fuel_efficiency': 6, 'speed': 40, 'cost': 300000},
            'FT4': {'capacity': 2800, 'personnel': 3, 'fuel_efficiency': 5, 'speed': 36, 'cost': 300000},
            'FT5': {'capacity': 2700, 'personnel': 2, 'fuel_efficiency': 4, 'speed': 32, 'cost': 300000},
            'FT6': {'capacity': 3000, 'personnel': 2, 'fuel_efficiency': 4, 'speed': 42.3, 'cost': 300000}
        }
        
        self.firefighter_capabilities = {
            'FF1': {'max_deployments': 3, 'cost': 2000},
            'FF2': {'max_deployments': 3, 'cost': 2000},
            'FF3': {'max_deployments': 3, 'cost': 2000},
            'FF4': {'max_deployments': 3, 'cost': 2000},
            'FF5': {'max_deployments': 3, 'cost': 2000},
            'FF6': {'max_deployments': 3, 'cost': 2000}
        }

        # 자원 배치 상태 초기화
        self.truck_deployments = {truck_type: 0 for truck_type in self.truck_types}
        self.firefighter_deployments = {ff_type: 0 for ff_type in self.firefighter_types}

    def set_resource_deployment(self, resource_type: str, resource_id: str, quantity: int):
        """자원 배치 상태를 설정"""
        if resource_type == 'truck' and resource_id in self.truck_types:
            self.truck_deployments[resource_id] = min(quantity, 2)  # 최대 2대
        elif resource_type == 'firefighter' and resource_id in self.firefighter_types:
            self.firefighter_deployments[resource_id] = min(quantity, 3)  # 최대 3명

    def get_available_resources(self):
        """현재 사용 가능한 자원 현황 반환"""
        return {
            'trucks': {truck_type: 2 - self.truck_deployments[truck_type] 
                      for truck_type in self.truck_types},
            'firefighters': {ff_type: 3 - self.firefighter_deployments[ff_type] 
                           for ff_type in self.firefighter_types}
        }

    def get_deployed_resources(self):
        """현재 배치된 자원 현황 반환"""
        return {
            'trucks': self.truck_deployments.copy(),
            'firefighters': self.firefighter_deployments.copy()
        }

    def optimize_single_scenario(self, scenario: FireScenario) -> Tuple[List[Dict], float]:
        """단일 시나리오에 대한 최적화 수행"""
        model = pulp.LpProblem(f"Fire_Resource_Allocation_Scenario_{scenario.id}", pulp.LpMinimize)
        
        # 결정변수 정의 (정수형으로 변경)
        x = pulp.LpVariable.dicts("truck_assign", 
                                [(i,n) for i in self.truck_types for n in scenario.sites.keys()], 
                                cat='Integer', lowBound=0, upBound=2)  # 최대 2대까지 배치 가능
        y = pulp.LpVariable.dicts("ff_assign", 
                                [(j,n) for j in self.firefighter_types for n in scenario.sites.keys()], 
                                cat='Integer', lowBound=0, upBound=3)  # 최대 3명까지 배치 가능

        # 이진 변수 선언 (트럭 타입 i가 지점 n에 배치되면 1)
        z = pulp.LpVariable.dicts("truck_location", 
                                [(i,n) for i in self.truck_types for n in scenario.sites.keys()], 
                                cat='Binary')

        # 목적함수
        model += pulp.lpSum(
            self.truck_capabilities[i]['cost'] * x[(i,n)] +
            self.firefighter_capabilities[j]['cost'] * y[(j,n)] +
            (scenario.sites[n]['distance'][i]/self.truck_capabilities[i]['fuel_efficiency']) * 1000 * x[(i,n)]
            for i in self.truck_types 
            for j in self.firefighter_types 
            for n in scenario.sites.keys()
        )

        # 제약조건
        # 소방차 최대 배치 수 (현재 배치 상태 고려)
        for i in self.truck_types:
            model += pulp.lpSum(x[(i,n)] for n in scenario.sites.keys()) <= \
                    (2 - self.truck_deployments[i])  # 현재 배치된 수량을 고려
            # 지점별 배치 제한 (1개 지점만 선택)
            model += pulp.lpSum(x[(i,n)] for n in scenario.sites.keys()) <= 1

            # x와 z 변수 연결
            for n in scenario.sites.keys():
                model += x[(i,n)] <= 2 * z[(i,n)]  # z=0 → x=0, z=1 → x≤2
                model += x[(i,n)] >= z[(i,n)]      # z=1 → x≥1 (최소 1대 배치)
        
        # 대원별 최대 출동 횟수 (현재 배치 상태 고려)
        for j in self.firefighter_types:
            model += pulp.lpSum(y[(j,n)] for n in scenario.sites.keys()) <= \
                    (self.firefighter_capabilities[j]['max_deployments'] - self.firefighter_deployments[j])

        # 지점별 인원 수요 충족 (완화된 제약)
        for n in scenario.sites.keys():
            model += pulp.lpSum(self.truck_capabilities[i]['personnel'] * x[(i,n)] 
                              for i in self.truck_types) + \
                    pulp.lpSum(y[(j,n)] for j in self.firefighter_types) >= \
                    max(1, scenario.sites[n]['demand'] - 1)  # 수요를 1명까지 부족하게 허용

        # 대원은 배치된 소방차가 있어야 이동 가능 (완화된 제약)
        for j in self.firefighter_types:
            for n in scenario.sites.keys():
                # 소방차 1대당 최대 3명의 대원 배치 가능하도록 제약조건 수정
                model += pulp.lpSum(x[(i,n)] for i in self.truck_types) * 3 >= y[(j,n)]

        # 시간 제약 (완화된 제약)
        for i in self.truck_types:
            for n in scenario.sites.keys():
                travel_time = scenario.sites[n]['distance'][i] / self.truck_capabilities[i]['speed']
                model += travel_time * x[(i,n)] <= 5.0  # 5시간으로 완화

        # 최적화 실행
        model.solve()
        
        # 결과 수집
        results = []
        if model.status == pulp.LpStatusOptimal:
            for i in self.truck_types:
                for n in scenario.sites.keys():
                    if pulp.value(x[(i,n)]) > 0.5:
                        # 자원 배치 상태 업데이트
                        self.truck_deployments[i] += int(pulp.value(x[(i,n)]))
                        results.append({
                            'scenario': scenario.id,
                            'base_station': scenario.base_station,
                            'type': i,
                            'location': n,
                            'resource_type': 'truck',
                            'quantity': int(pulp.value(x[(i,n)])),
                            'distance': scenario.sites[n]['distance'][i],
                            'latitude': scenario.sites[n]['latitude'],
                            'longitude': scenario.sites[n]['longitude']
                        })
            
            for j in self.firefighter_types:
                for n in scenario.sites.keys():
                    if pulp.value(y[(j,n)]) > 0.5:
                        # 자원 배치 상태 업데이트
                        self.firefighter_deployments[j] += int(pulp.value(y[(j,n)]))
                        results.append({
                            'scenario': scenario.id,
                            'type': j,
                            'location': n,
                            'resource_type': 'firefighter',
                            'quantity': int(pulp.value(y[(j,n)])),
                            'distance': scenario.sites[n]['distance'][j],
                            'latitude': scenario.sites[n]['latitude'],
                            'longitude': scenario.sites[n]['longitude']
                        })
            
            return results, pulp.value(model.objective)
        return [], float('inf')

class RiskCalculator:
    def __init__(self):
        self.risk_factors = {
            'wind_speed': {
                'weight': 0.3,
                'thresholds': [(0, 0.2), (5, 0.4), (10, 0.6), (15, 0.8), (20, 1.0)]
            },
            'humidity': {
                'weight': 0.2,
                'thresholds': [(80, 0.2), (60, 0.4), (40, 0.6), (20, 0.8), (0, 1.0)]
            },
            'fuel_type': {
                'weight': 0.2,
                'values': {
                    1: 0.2,  # 낮은 연료량
                    2: 0.4,
                    3: 0.6,
                    4: 0.8,
                    5: 1.0   # 높은 연료량
                }
            },
            'slope': {
                'weight': 0.15,
                'thresholds': [(0, 0.2), (10, 0.4), (20, 0.6), (30, 0.8), (45, 1.0)]
            },
            'damage_class': {
                'weight': 0.15,
                'values': {
                    1: 0.2,  # 낮은 피해 등급
                    2: 0.4,
                    3: 0.6,
                    4: 0.8,
                    5: 1.0   # 높은 피해 등급
                }
            }
        }

    def calculate_risk_score(self, risk_factors: Dict) -> float:
        """위험 요소들을 기반으로 위험도 점수 계산 (0-100)"""
        total_score = 0
        total_weight = 0

        for factor, value in risk_factors.items():
            if factor in self.risk_factors:
                factor_info = self.risk_factors[factor]
                weight = factor_info['weight']
                
                if 'thresholds' in factor_info:
                    # 연속형 변수 처리 (풍속, 습도, 경사도)
                    score = self._calculate_continuous_score(value, factor_info['thresholds'])
                else:
                    # 이산형 변수 처리 (연료 유형, 피해 등급)
                    score = factor_info['values'].get(value, 0.5)
                
                total_score += score * weight
                total_weight += weight

        # 가중 평균 계산 및 100점 만점으로 변환
        final_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
        return round(final_score, 1)

    def _calculate_continuous_score(self, value: float, thresholds: List[Tuple[float, float]]) -> float:
        """연속형 변수의 점수 계산"""
        for i in range(len(thresholds) - 1):
            if thresholds[i][0] <= value < thresholds[i + 1][0]:
                # 선형 보간
                x1, y1 = thresholds[i]
                x2, y2 = thresholds[i + 1]
                return y1 + (y2 - y1) * (value - x1) / (x2 - x1)
        
        # 범위를 벗어난 경우
        if value < thresholds[0][0]:
            return thresholds[0][1]
        return thresholds[-1][1]

    def get_risk_level(self, score: float) -> str:
        """위험도 점수를 기반으로 위험 수준 반환"""
        if score >= 80:
            return "심각"
        elif score >= 60:
            return "높음"
        elif score >= 40:
            return "보통"
        elif score >= 20:
            return "낮음"
        else:
            return "매우 낮음"

    def get_risk_factors_description(self, risk_factors: Dict) -> List[str]:
        """위험 요소들의 설명 반환"""
        descriptions = []
        
        if risk_factors.get('wind_speed', 0) >= 15:
            descriptions.append("강한 바람")
        if risk_factors.get('humidity', 100) <= 30:
            descriptions.append("건조한 날씨")
        if risk_factors.get('fuel_type', 1) >= 4:
            descriptions.append("높은 연료량")
        if risk_factors.get('slope', 0) >= 30:
            descriptions.append("가파른 경사")
        if risk_factors.get('damage_class', 1) >= 4:
            descriptions.append("높은 피해 등급")
            
        return descriptions

def main():
    print("화재 대응 자원 배치 최적화 시스템")
    
    # 데이터 로드 및 전처리
    print("데이터를 로드하고 전처리하는 중...")
    features_processed, target_processed = load_and_preprocess_data()
        
    if features_processed is None or target_processed is None:
        print("데이터 로드 중 오류가 발생했습니다.")
        return

    # 실제 데이터 기반 시나리오 생성
    scenarios = generate_scenarios_from_data(features_processed, target_processed)
    
    # 자원 할당기 초기화
    allocator = ResourceAllocator()
    
    # 각 시나리오별 최적화 수행
    all_results = []
    total_cost = 0
    successful_scenarios = 0
    
    print("\n시나리오별 최적화 수행 중...")
    for scenario in scenarios:
        results, cost = allocator.optimize_single_scenario(scenario)
        if results:  # 최적화가 성공한 경우에만 결과 추가
            all_results.extend(results)
            total_cost += cost * scenario.probability
            successful_scenarios += 1
            print(f"시나리오 {scenario.id} 최적화 완료")

    # 결과 출력
    print("\n최적화 결과")
    print(f"총 비용: {total_cost:,.0f}")
    print(f"배치된 자원 수: {len(all_results)}")
    print(f"성공한 시나리오 수: {successful_scenarios}")


