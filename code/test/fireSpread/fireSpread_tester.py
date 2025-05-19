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
from fireSpread import FireSpreadSimulator

"""
지도의 화재 지점을 누르면 화재 확산 시뮬레이션이 재생되게 구성하면 좋을 것 같음
"""

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
            # 클러스터 통계를 기반으로 수요 계산
            base_demand = max(1, int(self.cluster_stats['required_resources'] / num_sites))
            demand = max(1, min(5, base_demand + random.randint(-1, 1)))  # 1-5명
            
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
        
        self.firefighter_capabilities = { #소방관 투입 비용을 고려해야할까? 어차피 소방차를 운용하기 위해선 소방공무원이 필수적이라 비용 중복 여지가 있음.
            'FF1': {'max_deployments': 3, 'cost': 2000},
            'FF2': {'max_deployments': 3, 'cost': 2000},
            'FF3': {'max_deployments': 3, 'cost': 2000},
            'FF4': {'max_deployments': 3, 'cost': 2000},
            'FF5': {'max_deployments': 3, 'cost': 2000},
            'FF6': {'max_deployments': 3, 'cost': 2000}
        }

    def optimize_single_scenario(self, scenario: FireScenario) -> Tuple[List[Dict], float]:
        """단일 시나리오에 대한 최적화 수행"""
        model = pulp.LpProblem(f"Fire_Resource_Allocation_Scenario_{scenario.id}", pulp.LpMinimize)
        
        # 결정변수 정의 (정수형으로 변경)
        x = pulp.LpVariable.dicts("truck_assign", 
                                [(i,n) for i in self.truck_types for n in scenario.sites.keys()], 
                                cat='Integer', lowBound=0, upBound=5)  # 최대 2대까지 배치 가능
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
        # 소방차 최대 배치 수

        for i in self.truck_types:
            model += pulp.lpSum(x[(i,n)] for n in scenario.sites.keys()) <= 2  # 최대 2대까지 배치 가능
            # 지점별 배치 제한 (1개 지점만 선택)
            model += pulp.lpSum(x[(i,n)] for n in scenario.sites.keys()) <= 1

            # x와 z 변수 연결
            for n in scenario.sites.keys():
                model += x[(i,n)] <= 2 * z[(i,n)]  # z=0 → x=0, z=1 → x≤2
                model += x[(i,n)] >= z[(i,n)]      # z=1 → x≥1 (최소 1대 배치)
        
        # 대원별 최대 출동 횟수
        for j in self.firefighter_types:
            model += pulp.lpSum(y[(j,n)] for n in scenario.sites.keys()) <= \
                    self.firefighter_capabilities[j]['max_deployments']

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
    print(f"배치 수: {len(all_results)}")
    print(f"성공한 시나리오 수: {successful_scenarios}/{len(scenarios)}")
    
    # 시나리오별 배치 현황 출력 및 지도 시각화
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r['scenario'] == scenario.id]
        if scenario_results:  # 결과가 있는 시나리오만 출력
            print(f"\n시나리오 {scenario.id} 배치 현황:")
            print(f"위험 요소: 풍속={scenario.cluster_stats['wind_speed']:.1f}m/s, "
                  f"경사도={scenario.cluster_stats['slope']:.1f}도, "
                  f"습도={scenario.cluster_stats['humidity']:.1f}%, "
                  f"피해 등급={scenario.cluster_stats['damage_class']}")
            print(f"기준 소방서 위치: {scenario.base_station['latitude']:.4f}, {scenario.base_station['longitude']:.4f}")
            
def simulate_and_optimize():
    features_processed, target_processed = load_and_preprocess_data()
    scenarios = generate_scenarios_from_data(features_processed, target_processed)
    allocator = ResourceAllocator()

    for scenario in scenarios:
        # 시나리오별 환경 변수 추출
        wind_speed = scenario.cluster_stats['wind_speed']
        humidity = scenario.cluster_stats['humidity']
        slope = scenario.cluster_stats['slope']
        fuel_moisture = 1 - humidity / 100  # 습도 → 연료수분 변환
        ignition_points = []
        lat = 35.18035823746264  # 진주소방서 위도
        lon = 128.11851962302458  # 진주소방서 경도
        for site in scenario.sites.values():
            # 위도, 경도를 FireSpreadSimulator의 격자 좌표로 변환
            x = int(((site['latitude'] - lat) / 0.01) * 50)  # 격자 크기에 맞게 조정
            y = int(((site['longitude'] - lon) / 0.01) * 50)
            # 격자 범위 내로 좌표 제한
            x = max(0, min(99, x))
            y = max(0, min(99, y))
            ignition_points.append((x, y))

        sim = FireSpreadSimulator(
            grid_size=100,
            resolution=30,
            burn_time=3,
            wind_speed=wind_speed,
            wind_direction=(0,1),  # 풍향은 필요시 scenario에서 추출
            fuel_moisture=fuel_moisture,
            slope=slope,
            ignition_points=ignition_points
        )
        fire_history = sim.run(steps=12)
        burned_area = sim.get_burned_area()
        fire_perimeter = sim.get_fire_perimeter()
        # 결과를 scenario.cluster_stats 등에 저장하여 최적화에 활용
       # scenario.cluster_stats['predicted_burned_area'] = burned_area
       # scenario.cluster_stats['predicted_perimeter'] = fire_perimeter

        # 자원 최적화
       # results, cost = allocator.optimize_single_scenario(scenario)
        # 결과 활용 및 시각화 등  
        # 시각화
        for t, grid in enumerate(fire_history):
            sim.visualize(grid, t)

if __name__ == "__main__":
    #main()
    simulate_and_optimize()
