import pandas as pd
from sklearn.impute import SimpleImputer
import pulp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import random
from typing import Dict, List, Tuple
from sklearn.impute import KNNImputer
import math
# from code.Map.Map import WildfireMap # 필요한 경우 주석 해제
import joblib
import numpy as np

# (이전 코드의 KOREAN_GBRT_MODEL_PATH, US_LINEAR_MODEL_PATH 등은 그대로 유지)
KOREAN_GBRT_MODEL_PATH = 'code/test/LinearProgramming/model/korean_gbrt_demand_model.joblib'
US_LINEAR_MODEL_PATH = 'code/test/LinearProgramming/model/us_acres_personnel_linear_model.joblib'

# --- K-means 클러스터링 함수 (새로 추가 또는 이전 버전 사용) ---
def create_fire_scenarios_by_clustering(df, n_scenarios=5, random_state=42):
    from sklearn.cluster import KMeans
    clustering_features = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC']
    existing_clustering_features = [col for col in clustering_features if col in df.columns]
    if len(existing_clustering_features) < 2:
        print("오류: 클러스터링에 사용할 수 있는 유효한 특성이 부족합니다.")
        return None
    X = df[existing_clustering_features].copy()
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters_actual = min(n_scenarios, len(df))
    if n_clusters_actual == 0:
        print("오류: 클러스터링할 데이터가 없습니다.")
        return None
    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=random_state, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

# --- load_and_preprocess_data 함수 (이전 요청의 수정된 버전 사용) ---
def load_and_preprocess_data():
    # ... (이전 요청에서 제공된, df_processed 하나만 반환하는 버전의 상세한 전처리 로직) ...
    # (이 함수는 이제 단일 DataFrame 'df_processed'를 반환해야 함)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

    possible_paths = [
        os.path.join(project_root, 'datasets', 'WSQ000301.csv'),
        os.path.join(project_root, 'code', 'test', 'LinearProgramming', 'datasets', 'WSQ000301.csv'),
        os.path.join(current_dir, '..', '..', 'datasets', 'WSQ000301.csv'),
    ]

    korea_data_file_path_found = None
    for path_option in possible_paths:
        if os.path.exists(path_option):
            korea_data_file_path_found = path_option
            print(f"파일 찾음: {os.path.abspath(korea_data_file_path_found)}")
            break
    
    if korea_data_file_path_found is None:
        print(f"오류: 시나리오 생성용 한국 데이터 파일을 다음 경로들에서 찾을 수 없습니다: {possible_paths}")
        return None # 단일 값 반환으로 변경

    try:
        df_kr_raw = pd.read_csv(korea_data_file_path_found, encoding='UTF-8')
        print(f"시나리오 생성용 한국 데이터 로드 성공: {df_kr_raw.shape}")
    except Exception as e:
        print(f"시나리오 생성용 한국 데이터 로드 중 오류 발생: {e}")
        return None # 단일 값 반환으로 변경

    df_processed = df_kr_raw.copy()

    target_cols = ['POTFR_RSRC_INPT_QNTT', 'FRFR_DMG_AREA']
    for col in target_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).apply(lambda x: max(0,x) if pd.notnull(x) else 0)
        else: df_processed[col] = 0

    numeric_cols_for_stats = ['WDSP', 'HMDT', 'TPRT', 'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC']
    for col in numeric_cols_for_stats:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # 결측치는 이후 처리
        else: df_processed[col] = np.nan

    def map_dnst_cd(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).upper();
        if val_str == 'A': return 5.0;
        if val_str == 'B': return 3.0;
        if val_str == 'C': return 1.0
        try: return float(val_str)
        except ValueError: return np.nan
    
    if 'DNST_CD' in df_processed.columns:
        df_processed['slope_mapped'] = df_processed['DNST_CD'].apply(map_dnst_cd)
    else: df_processed['slope_mapped'] = np.nan

    categorical_cols_map_gbrt = { # GBRT 모델이 사용하는 최종 컬럼명으로 매핑
        'FRTP_CD': 'fuel_type_code', 'DMCLS_CD': 'damage_class_code',
        'FRFR_OCCRR_CAUS_NM': 'cause_code', 'STORUNST_CD': 'terrain_code',
        'AGCLS_CD': 'age_class_code', 'CMPSI_FG': 'compass_direction'
    }
    for original_col, new_col_name in categorical_cols_map_gbrt.items():
        if original_col in df_processed.columns:
            df_processed[new_col_name] = df_processed[original_col].astype(str).fillna('Unknown')
        else: df_processed[new_col_name] = 'Unknown'
    
    # GIS 임시 특성 (나중에 실제 데이터로 대체)
    df_processed['gis_actual_slope'] = np.random.uniform(0, 45, size=len(df_processed))
    df_processed['gis_fuel_category_detailed'] = np.random.choice(['침엽수밀집', '활엽수밀집', '혼효림', '초지'], size=len(df_processed))
    df_processed['gis_dist_to_fire_station'] = np.random.uniform(100, 15000, size=len(df_processed))

    print(f"기본 전처리 후 한국 데이터 shape: {df_processed.shape}")
    return df_processed # 전처리된 단일 DataFrame 반환

# --- generate_scenarios_from_data 함수 정의 (클래스 외부의 독립 함수) ---
# (이전 요청에서 제공된, 한국 데이터 클러스터링 기반 시나리오 생성 로직으로 수정)
def generate_scenarios_from_data(df_processed_for_scenario_creation, n_scenarios=5): # 파라미터 이름 변경 및 n_scenarios 추가
    if df_processed_for_scenario_creation is None or df_processed_for_scenario_creation.empty:
        print("오류: 시나리오 생성을 위한 데이터가 비어 있습니다.")
        return []

    print("한국 산불 데이터 기반 K-means 클러스터링 시작...")
    # create_fire_scenarios_by_clustering 함수는 df_processed_for_scenario_creation을 직접 사용
    clusters = create_fire_scenarios_by_clustering(df_processed_for_scenario_creation, n_scenarios=n_scenarios)
    if clusters is None:
        print("오류: K-means 클러스터링에 실패했습니다.")
        return []
    
    df_with_cluster = df_processed_for_scenario_creation.copy()
    df_with_cluster['cluster_id_for_scenario'] = clusters # 클러스터 ID 컬럼명 변경

    actual_n_clusters = len(np.unique(clusters)) # 실제 생성된 클러스터 수
    cluster_stats_list = []

    for cluster_idx in range(actual_n_clusters):
        cluster_data = df_with_cluster[df_with_cluster['cluster_id_for_scenario'] == cluster_idx]
        if cluster_data.empty:
            print(f"경고: 클러스터 {cluster_idx}에는 데이터가 없습니다.")
            continue
            
        stats = {}
        # GBRT 모델 입력에 필요한 모든 특성의 클러스터 통계 계산
        # (이 특성명들은 FireScenario._generate_sites_from_stats에서 사용하는 키와 일치해야 함)
        gbrt_input_cols = [
            'FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 'slope_mapped',
            'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
            'fuel_type_code', 'damage_class_code', 'cause_code',
            'terrain_code', 'age_class_code', 'compass_direction',
            'gis_actual_slope', 'gis_fuel_category_detailed', 'gis_dist_to_fire_station'
        ]
        for col in gbrt_input_cols:
            if col in cluster_data.columns:
                if pd.api.types.is_numeric_dtype(cluster_data[col]):
                    stats[col] = float(cluster_data[col].mean())
                else: # 범주형
                    stats[col] = cluster_data[col].mode()[0] if not cluster_data[col].mode().empty else 'Unknown'
            else: # 해당 컬럼이 원본 데이터에 없었을 경우 (load_and_preprocess_data에서 Unknown으로 채워짐)
                stats[col] = 'Unknown' if col in ['fuel_type_code', 'damage_class_code', 'cause_code', 'terrain_code', 'age_class_code', 'compass_direction', 'gis_fuel_category_detailed'] else 0.0


        stats['required_resources'] = float(cluster_data['POTFR_RSRC_INPT_QNTT'].mean()) # 시나리오 대표 수요 (클러스터 평균)
        # 'damage_area'는 GBRT 입력용 'FRFR_DMG_AREA'와 동일하게 사용
        stats['damage_area'] = stats.get('FRFR_DMG_AREA', 0.0) 
        stats['probability'] = len(cluster_data) / len(df_processed_for_scenario_creation)
        
        cluster_stats_list.append(stats)
    
    if not cluster_stats_list:
        print("경고: 생성된 클러스터 통계가 없습니다.")
        return []
        
    scenarios = []
    for i, stats_dict in enumerate(cluster_stats_list):
        scenario = FireScenario(
            scenario_id=i,
            probability=stats_dict.get('probability', 0),
            cluster_stats=stats_dict 
        )
        scenarios.append(scenario)
    print(f"{len(scenarios)}개의 시나리오 생성 완료.")
    return scenarios


class FireScenario:
    def __init__(self, scenario_id: int, probability: float, cluster_stats: Dict):
        self.id = scenario_id
        self.probability = probability
        self.cluster_stats = cluster_stats
        self.base_station = {'name': '기준 소방서', 'latitude': 35.18035823746264, 'longitude': 128.11851962302458}
        try:
            self.korean_gbrt_model_pipeline = joblib.load(KOREAN_GBRT_MODEL_PATH)
        except Exception: self.korean_gbrt_model_pipeline = None
        try:
            self.us_linear_model = joblib.load(US_LINEAR_MODEL_PATH)
        except Exception: self.us_linear_model = None
        self.sites = self._generate_sites_from_stats()

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad; dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)); distance = R * c
        return distance

    def _generate_sites_from_stats(self) -> Dict[str, Dict]:
        sites = {}
        
        # --- num_sites 결정 로직: 소규모 화재는 대부분 단일 지점으로 가정 ---
        # 시나리오의 대표 피해 면적을 기준으로 num_sites 결정
        scenario_avg_total_damage_ha = self.cluster_stats.get('damage_area', 1.0)
        
        if scenario_avg_total_damage_ha <= 5.0: # 예: 5헥타르 이하의 소규모 화재는 단일 지점으로 간주
            num_sites = 1
        elif scenario_avg_total_damage_ha <= 20.0: # 5~20헥타르는 2개 지점까지 고려
            num_sites = random.randint(1, 2)
        else: # 20헥타르 초과는 2~3개 지점 (여기서는 최대 2개로 제한하여 테스트)
            num_sites = random.randint(1, 2) # 또는 3으로 하되, 각 site 수요 조절 필요


        for i in range(num_sites):
            site_id = f'site{i+1}'
            site_features_dict = {} 

            # 각 site의 FRFR_DMG_AREA 할당
            if num_sites == 1:
                site_damage_area_ha = max(0.1, scenario_avg_total_damage_ha) # 시나리오 전체 면적 사용, 최소 0.1ha
            else:
                # 여러 지점일 경우, 면적을 나누되, 각 지점이 너무 작아지지 않도록 조정
                site_damage_area_ha = max(0.1, scenario_avg_total_damage_ha / num_sites + random.uniform(-0.05, 0.05) * (scenario_avg_total_damage_ha / num_sites) ) # 약간의 변동성 추가
            
            site_features_dict['FRFR_DMG_AREA'] = site_damage_area_ha
            
            # GBRT 모델 입력 특성 구성 (이전과 동일한 로직 사용)
            all_gbrt_keys_needed = [
                'FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 'slope_mapped',
                'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                'fuel_type_code', 'damage_class_code', 'cause_code',
                'terrain_code', 'age_class_code', 'compass_direction',
                'gis_actual_slope', 'gis_fuel_category_detailed', 'gis_dist_to_fire_station',
                'US_based_predicted_personnel'
            ]
            for key in all_gbrt_keys_needed:
                if key == 'FRFR_DMG_AREA': continue
                if key == 'US_based_predicted_personnel': continue
                default_value = 0.0 if key in ['WDSP','HMDT','TPRT','slope_mapped','FRSTTN_DSTNC','PTMNT_DSTNC','NNFRS_DSTNC','gis_actual_slope','gis_dist_to_fire_station'] else 'Unknown'
                site_features_dict[key] = self.cluster_stats.get(key, default_value)
            
            if self.us_linear_model and 'FRFR_DMG_AREA' in site_features_dict:
                hectare_to_acre = 2.47105
                current_site_dmg_area = site_features_dict.get('FRFR_DMG_AREA', 0.0)
                acres_kr_site_individual = current_site_dmg_area * hectare_to_acre
                log_acres_kr_site_individual = np.log1p(max(0, acres_kr_site_individual))
                X_for_us_site_individual = pd.DataFrame({'log_ACRES': [log_acres_kr_site_individual]})
                log_pred_us_site_individual = self.us_linear_model.predict(X_for_us_site_individual)
                pred_us_site_individual = np.expm1(log_pred_us_site_individual)
                site_features_dict['US_based_predicted_personnel'] = max(0, int(round(pred_us_site_individual[0])))
            else:
                site_features_dict['US_based_predicted_personnel'] = 0

            site_features_df = pd.DataFrame([site_features_dict])
            
            predicted_demand_gbrt_raw = 5 
            predicted_demand_gbrt_final = 5
            if self.korean_gbrt_model_pipeline:
                try:
                    log_predicted_potfr = self.korean_gbrt_model_pipeline.predict(site_features_df)
                    predicted_potfr = np.expm1(log_predicted_potfr)[0]
                    predicted_demand_gbrt_raw = max(1, int(round(predicted_potfr)))
                    
                    # --- 각 사이트별 수요 상한선 적용 (배경 설명 참고) ---
                    # 1헥타르 미만 소규모: 5~20명 (초기 대응팀 규모)
                    # 1~5헥타르: 15~40명
                    # 5헥타르 초과: GBRT 예측값을 사용하되, 최대 50~60명 정도로 제한
                    current_site_area_for_demand_cap = site_features_dict.get('FRFR_DMG_AREA', 1.0)
                    if current_site_area_for_demand_cap <= 1.0:
                        MAX_DEMAND_THIS_SITE = random.randint(10, 25) # 초기 대응팀 + 약간의 지원
                    elif current_site_area_for_demand_cap <= 5.0:
                        MAX_DEMAND_THIS_SITE = random.randint(20, 40)
                    else: # 5헥타르 초과
                        MAX_DEMAND_THIS_SITE = random.randint(30, 50) # 더 큰 화재는 더 많은 인원, 하지만 LP 고려
                    
                    predicted_demand_gbrt_final = min(predicted_demand_gbrt_raw, MAX_DEMAND_THIS_SITE)
                    predicted_demand_gbrt_final = max(5, predicted_demand_gbrt_final) # 최소 5명 보장
                    # ----------------------------------------------------
                    
                    print(f"Scenario {self.id}, Site {site_id}: 면적={site_damage_area_ha:.2f}ha, "
                          f"GBRT예측(Raw)={predicted_demand_gbrt_raw}명, 최종수요(capped)={predicted_demand_gbrt_final}명")
                except Exception as e:
                    print(f"Scenario {self.id}, Site {site_id} GBRT 수요 예측 중 오류: {e}")
                    predicted_demand_gbrt_final = 5 
            
            # ... (이하 site_lat, site_lon, site_distances, sites[site_id] 할당 로직은 이전과 동일) ...
            site_lat = random.uniform(35.10468233527785, 35.28450887192325)
            site_lon = random.uniform(128.01212832039607, 128.18678592428446)
            base_distance_to_site = self._calculate_distance(
                self.base_station['latitude'], self.base_station['longitude'], site_lat, site_lon
            )
            site_distances = {}
            defined_truck_types = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
            for truck_type in defined_truck_types:
                distance_variation_site = base_distance_to_site * random.uniform(-0.1, 0.1)
                site_distances[truck_type] = max(2, min(30, base_distance_to_site + distance_variation_site))

            sites[site_id] = {
                'demand': predicted_demand_gbrt_final,
                'distance': site_distances,
                'risk_factors': { 
                    'wind_speed': site_features_dict.get('WDSP', 1.0),
                    'fuel_type': str(site_features_dict.get('fuel_type_code', 'Unknown')),
                    'slope': site_features_dict.get('slope_mapped', 10.0),
                    'humidity': site_features_dict.get('HMDT', 50.0),
                    'damage_class': str(site_features_dict.get('damage_class_code', 'Unknown'))
                },
                'latitude': site_lat, 'longitude': site_lon,
                'predicted_damage_area_ha': site_features_dict.get('FRFR_DMG_AREA', 0.0)
            }
        return sites

# --- ResourceAllocator 클래스 (이전과 동일, LP 모델 제약 완화 버전 사용) ---
class ResourceAllocator:
    def __init__(self):
        self.truck_types = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
        self.firefighter_types = ['FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6']
        self.truck_capabilities = {
            'FT1': {'capacity': 2800, 'personnel': 4, 'fuel_efficiency': 4, 'speed': 36.8, 'cost': 300000},
            'FT2': {'capacity': 3000, 'personnel': 4, 'fuel_efficiency': 5, 'speed': 36.8, 'cost': 300000},
            'FT3': {'capacity': 2900, 'personnel': 4, 'fuel_efficiency': 6, 'speed': 40, 'cost': 300000},
            'FT4': {'capacity': 2800, 'personnel': 3, 'fuel_efficiency': 5, 'speed': 36, 'cost': 300000},
            'FT5': {'capacity': 2700, 'personnel': 2, 'fuel_efficiency': 4, 'speed': 32, 'cost': 300000},
            'FT6': {'capacity': 3000, 'personnel': 2, 'fuel_efficiency': 4, 'speed': 42.3, 'cost': 300000}
        }
        # 총 가용 소방관 수
        self.firefighter_capabilities = {
            'FF1': {'max_deployments': 10, 'cost': 2000}, 'FF2': {'max_deployments': 10, 'cost': 2000},
            'FF3': {'max_deployments': 10, 'cost': 2000}, 'FF4': {'max_deployments': 10, 'cost': 2000},
            'FF5': {'max_deployments': 10, 'cost': 2000}, 'FF6': {'max_deployments': 10, 'cost': 2000}
        }
        self.MAX_TRUCKS_PER_TYPE_TOTAL = 11 # 각 트럭 타입별 전체 보유 대수
        self.MAX_FF_PER_TYPE_TOTAL = 335     # 각 소방관 타입별 전체 보유 인원

        self.truck_deployments = {truck_type: 0 for truck_type in self.truck_types} # 시나리오 간 누적 X
        self.firefighter_deployments = {ff_type: 0 for ff_type in self.firefighter_types} # 시나리오 간 누적 X

    def optimize_single_scenario(self, scenario: FireScenario) -> Tuple[List[Dict], float]:
        # 시나리오 시작 시 현재 가용 자원 계산 (전체 보유량 기준)
        available_trucks_for_scenario = {
            truck_type: self.MAX_TRUCKS_PER_TYPE_TOTAL for truck_type in self.truck_types
        }
        available_ff_for_scenario = {
            ff_type: self.MAX_FF_PER_TYPE_TOTAL for ff_type in self.firefighter_types
        }
        
        # 디버깅용 로그 추가
        total_scenario_demand = sum(site_info['demand'] for site_info in scenario.sites.values())
        max_personnel_from_trucks = sum(self.truck_capabilities[tt]['personnel'] * available_trucks_for_scenario[tt] for tt in self.truck_types)
        max_personnel_from_ff = sum(available_ff_for_scenario[ff] for ff in self.firefighter_types)
        print(f"시나리오 {scenario.id}: 총 수요={total_scenario_demand}")


        model = pulp.LpProblem(f"Fire_Resource_Allocation_Scenario_{scenario.id}", pulp.LpMinimize)
        
        # 변수 정의: x는 각 트럭타입(i)을 각 지점(n)에 몇 대 할당할지 (0, 1, 또는 2대)
        # z는 각 트럭타입(i)이 각 지점(n)에 "배치되었는지 여부"(0 또는 1)
        x = pulp.LpVariable.dicts("truck_assign", [(i,n) for i in self.truck_types for n in scenario.sites.keys()], cat='Integer', lowBound=0, upBound=self.MAX_TRUCKS_PER_TYPE_TOTAL) # 최대 보유 대수만큼
        y = pulp.LpVariable.dicts("ff_assign", [(j,n) for j in self.firefighter_types for n in scenario.sites.keys()], cat='Integer', lowBound=0, upBound=self.MAX_FF_PER_TYPE_TOTAL) # 최대 보유 인원만큼
        z = pulp.LpVariable.dicts("truck_at_location", [(i,n) for i in self.truck_types for n in scenario.sites.keys()], cat='Binary') # 트럭이 해당 위치에 갔는지 여부

        # 목적 함수 (비용 최소화)
        model += pulp.lpSum(
            self.truck_capabilities[i]['cost'] * x[(i,n)] +
            (scenario.sites[n]['distance'][i] / self.truck_capabilities[i]['fuel_efficiency']) * 1000 * x[(i,n)] # 연료비 가정
            for i in self.truck_types for n in scenario.sites.keys()
        ) # 소방관 비용은 트럭 비용에 포함되거나, 별도로 추가 (여기서는 제외)

        # 제약 조건
        # 1. 각 트럭 타입별 총 배치 대수 제한
        for i in self.truck_types:
            model += pulp.lpSum(x[(i,n)] for n in scenario.sites.keys()) <= available_trucks_for_scenario[i]

        # 2. 각 소방관 타입별 총 배치 인원 제한
        for j in self.firefighter_types:
            model += pulp.lpSum(y[(j,n)] for n in scenario.sites.keys()) <= available_ff_for_scenario[j]

        # 3. 각 지점별 수요 충족 (트럭 탑승 인원 + 추가 소방관 >= 수요)
        for n in scenario.sites.keys():
            model += pulp.lpSum(self.truck_capabilities[i]['personnel'] * x[(i,n)] for i in self.truck_types) + \
                     pulp.lpSum(y[(j,n)] for j in self.firefighter_types) >= scenario.sites[n]['demand']

        # 4. 한 지점에 배치되는 총 추가 소방관 수는 해당 지점에 배치된 트럭의 총 탑승 가능 인원 수를 초과할 수 없음
        for n in scenario.sites.keys():
            model += pulp.lpSum(y[(j,n)] for j in self.firefighter_types) <= \
                     pulp.lpSum(self.truck_capabilities[i]['personnel'] * x[(i,n)] for i in self.truck_types)
        
        # 5. 트럭이 특정 위치에 가기로 결정되면(z=1) 최소 1대 이상, 최대 MAX_TRUCKS_PER_TYPE_SCENARIO 대까지 (또는 해당 타입의 총 보유 대수)
        #    트럭이 가지 않으면(z=0) 0대.
        #    그리고 각 트럭 "타입"은 전체 시나리오에서 최대 "한 곳"의 지점에만 대표로 갈 수 있다는 제약은 제거 (이전 버전에서 문제 유발 가능성)
        for i in self.truck_types:
            for n in scenario.sites.keys():
                model += x[(i,n)] <= self.MAX_TRUCKS_PER_TYPE_TOTAL * z[(i,n)] # 해당 타입의 최대 가용대수 * z
                model += x[(i,n)] >= 0 # 이미 lowBound로 처리됨 (z[(i,n)]과의 직접적인 하한 연결은 복잡도 증가)

        # 6. 이동 시간 제약 (예: 2시간 이내 도착)
        MAX_TRAVEL_TIME = 3.0 # 이동시간 제약 완화 (3시간)
        for i in self.truck_types:
            for n in scenario.sites.keys():
                if self.truck_capabilities[i]['speed'] > 0 : # 속도가 0보다 클 때만
                    travel_time = scenario.sites[n]['distance'][i] / self.truck_capabilities[i]['speed']
                    model += travel_time * z[(i,n)] <= MAX_TRAVEL_TIME # 트럭이 그 지점에 "가기로 결정했을 때만" 이동시간 제약 적용
                else: # 속도가 0인 트럭은 이동 불가하므로 해당 지점에 배치 불가 (이런 경우는 없어야 함)
                    model += z[(i,n)] == 0


        try:
            # solver = pulp.PULP_CBC_CMD(msg=False) # 로그 출력 안함
            # model.solve(solver)
            model.solve() # 기본 솔버 사용
        except pulp.PulpSolverError:
            print(f"시나리오 {scenario.id}: LP 솔버를 찾을 수 없거나 오류 발생. pulp.pulpTestAll()로 확인하세요.")
            return [], float('inf')

        results = []
        if model.status == pulp.LpStatusOptimal:
            current_scenario_truck_deployments = {truck_type: 0 for truck_type in self.truck_types}
            current_scenario_ff_deployments = {ff_type: 0 for ff_type in self.firefighter_types}

            for i_truck in self.truck_types:
                for n_site in scenario.sites.keys():
                    if pulp.value(x[(i_truck,n_site)]) > 0.5:
                        num_trucks_assigned = int(round(pulp.value(x[(i_truck,n_site)]))) # 반올림하여 정수화
                        current_scenario_truck_deployments[i_truck] += num_trucks_assigned
                        results.append({
                            'scenario': scenario.id, 'type': i_truck, 'location': n_site,
                            'resource_type': 'truck', 'quantity': num_trucks_assigned,
                            'distance': scenario.sites[n_site]['distance'][i_truck],
                            'latitude': scenario.sites[n_site]['latitude'], 'longitude': scenario.sites[n_site]['longitude']
                        })
            for j_ff in self.firefighter_types:
                for n_site in scenario.sites.keys():
                    if pulp.value(y[(j_ff,n_site)]) > 0.5:
                        num_ff_assigned = int(round(pulp.value(y[(j_ff,n_site)])))
                        current_scenario_ff_deployments[j_ff] += num_ff_assigned
                        results.append({
                            'scenario': scenario.id, 'type': j_ff, 'location': n_site,
                            'resource_type': 'firefighter', 'quantity': num_ff_assigned,
                            'distance': 0, 
                            'latitude': scenario.sites[n_site]['latitude'], 'longitude': scenario.sites[n_site]['longitude']
                        })
            
            # !!! 중요: self.truck_deployments 와 self.firefighter_deployments를 업데이트하지 않음 !!!
            # 각 시나리오는 독립적으로 최적화되며, 이전 시나리오의 자원 사용이 다음 시나리오에 영향을 주지 않음.
            # 만약 시나리오 간 자원 제약이 누적되어야 한다면, 이 부분의 로직 수정 필요.
            # (현재는 각 시나리오마다 초기 가용 자원 사용)

            return results, pulp.value(model.objective)
        
        elif model.status == pulp.LpStatusInfeasible:
            print(f"시나리오 {scenario.id}: 문제가 실행 불가능합니다 (Infeasible). 제약 조건을 확인하세요.")
        elif model.status == pulp.LpStatusUnbounded:
            print(f"시나리오 {scenario.id}: 문제가 무한한 해를 가집니다 (Unbounded). 목적 함수나 제약 조건에 오류가 있을 수 있습니다.")
        else:
            print(f"시나리오 {scenario.id}: 최적해를 찾지 못했습니다. 상태: {pulp.LpStatus[model.status]}")
            
        return [], float('inf')


# RiskCalculator 클래스 정의 (이전과 동일)
class RiskCalculator:
    # ... (이전 요청의 RiskCalculator 클래스 내용 그대로 사용) ...
    def __init__(self):
        self.risk_factors = {
            'wind_speed': {'weight': 0.3, 'thresholds': [(0,0.2),(5,0.4),(10,0.6),(15,0.8),(20,1.0)]},
            'humidity': {'weight': 0.2, 'thresholds': [(80,0.2),(60,0.4),(40,0.6),(20,0.8),(0,1.0)]},
            'fuel_type': {'weight': 0.2, 'values': { '1.0':0.2,'2.0':0.4,'3.0':0.6,'4.0':0.8,'5.0':1.0, 'Unknown':0.5, '0.0':0.1, 'A':0.9, 'B':0.7, 'C':0.3 }},
            'slope': {'weight': 0.15, 'thresholds': [(0,0.2),(10,0.4),(20,0.6),(30,0.8),(45,1.0)]},
            'damage_class': {'weight': 0.15, 'values': { '1.0':0.2,'2.0':0.4,'3.0':0.6,'4.0':0.8,'5.0':1.0, 'Unknown':0.5, '0.0':0.1 }}
        }
    def calculate_risk_score(self, risk_factors: Dict) -> float:
        total_score = 0; total_weight = 0
        for factor, value in risk_factors.items():
            if factor in self.risk_factors:
                factor_info = self.risk_factors[factor]; weight = factor_info['weight']
                if 'thresholds' in factor_info: 
                    try: val_float = float(value)
                    except: val_float = 0 
                    score = self._calculate_continuous_score(val_float, factor_info['thresholds'])
                else: 
                    score = factor_info['values'].get(str(value).split('.')[0] if isinstance(value, float) and '.' in str(value) else str(value), 0.5) # 소수점 앞자리만 사용하거나 문자열 그대로
                total_score += score * weight; total_weight += weight
        return round((total_score / total_weight) * 100 if total_weight > 0 else 0, 1)

    def _calculate_continuous_score(self, value: float, thresholds: List[Tuple[float, float]]) -> float:
        for i in range(len(thresholds) - 1):
            if thresholds[i][0] <= value < thresholds[i+1][0]:
                x1,y1=thresholds[i]; x2,y2=thresholds[i+1]
                return y1+(y2-y1)*(value-x1)/(x2-x1)
        if value < thresholds[0][0]: return thresholds[0][1]
        return thresholds[-1][1]

    def get_risk_level(self, score: float) -> str:
        if score >= 80: return "심각"
        elif score >= 60: return "높음"
        elif score >= 40: return "보통"
        elif score >= 20: return "낮음"
        else: return "매우 낮음"

    def get_risk_factors_description(self, risk_factors: Dict) -> List[str]:
        descriptions = []
        try:
            if float(risk_factors.get('wind_speed', 0)) >= 15: descriptions.append("강한 바람")
            if float(risk_factors.get('humidity', 100)) <= 30: descriptions.append("건조한 날씨")
            if str(risk_factors.get('fuel_type', 'Unknown')) in ['4.0', '5.0', 'A']: descriptions.append("높은 연료량") 
            if float(risk_factors.get('slope', 0)) >= 30: descriptions.append("가파른 경사")
            if str(risk_factors.get('damage_class', 'Unknown')) in ['4.0', '5.0']: descriptions.append("높은 피해 등급") 
        except Exception: pass
        return descriptions


# main 함수 (이전과 동일)
def main():
    print("화재 대응 자원 배치 최적화 시스템 (GBRT 수요 예측 적용)")
    
    print("데이터를 로드하고 시나리오 생성을 위한 전처리 중...")
    df_processed_for_scenario_creation = load_and_preprocess_data() # 단일 DataFrame 반환

    if df_processed_for_scenario_creation is None:
        print("시나리오 생성을 위한 데이터 로드 중 오류가 발생했습니다.")
        return

    print("실제 데이터 기반 시나리오 생성 중...")
    scenarios = generate_scenarios_from_data(df_processed_for_scenario_creation, n_scenarios=5)
    
    if not scenarios:
        print("생성된 시나리오가 없습니다.")
        return

    allocator = ResourceAllocator() 
    risk_calc = RiskCalculator() 

    all_results = []
    total_cost = 0
    
    print("\n시나리오별 최적화 수행 중...")
    for scenario in scenarios: 
        if scenario.korean_gbrt_model_pipeline is None: 
            print(f"시나리오 {scenario.id}: GBRT 모델이 로드되지 않아 최적화를 건너뜁니다.")
            continue
        
        if scenario.sites: 
            first_site_key = list(scenario.sites.keys())[0]
            current_risk_factors_for_calc = scenario.sites[first_site_key]['risk_factors']
            
            scenario_risk_score = risk_calc.calculate_risk_score(current_risk_factors_for_calc)
            scenario_risk_level = risk_calc.get_risk_level(scenario_risk_score)
            print(f"\n시나리오 {scenario.id} (확률: {scenario.probability:.2f}):")
            print(f"  예상 위험도 점수: {scenario_risk_score}, 수준: {scenario_risk_level}")
            print(f"  주요 위험 요소: {risk_calc.get_risk_factors_description(current_risk_factors_for_calc)}")
            for site_id, site_info in scenario.sites.items():
                 print(f"  Site {site_id}: 예측 수요={site_info['demand']}명, 예상 피해면적={site_info.get('predicted_damage_area_ha', 'N/A'):.1f}ha")
        
        results, cost = allocator.optimize_single_scenario(scenario) 
        if results: # 최적해를 찾았을 경우에만
            all_results.extend(results)
            total_cost += cost * scenario.probability 
            print(f"시나리오 {scenario.id} 최적화 완료. 비용: {cost:.0f}")
        # else: # 최적해 못찾았을 때 (infeasible 등) 메시지는 optimize_single_scenario 내부에서 출력
            # print(f"시나리오 {scenario.id} 최적화 실패 또는 배치할 자원 없음.")


    print("\n--- 최종 최적화 결과 (모든 시나리오 가중 평균 비용) ---")
    print(f"총 가중 평균 비용: {total_cost:,.0f}")
    print(f"총 배치 건수: {len(all_results)}")


if __name__ == "__main__":
    # --- 중요: 실행 전 학습된 모델 파일 준비 ---
    # (이전과 동일한 더미 모델 생성 코드 또는 실제 모델 파일 준비)
    # 예시: 더미 모델 파일 생성 (실제로는 학습된 모델 사용)
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.preprocessing import StandardScaler, OneHotEncoder
    # from sklearn.compose import ColumnTransformer
    # from sklearn.pipeline import Pipeline
    # from sklearn.impute import SimpleImputer
    # from sklearn.linear_model import LinearRegression
    # import numpy as np
    # import pandas as pd
    # import joblib

    # # 더미 GBRT 모델 파이프라인 생성 (실제 학습 시와 유사한 구조)
    # numeric_features_dummy = ['FRFR_DMG_AREA','WDSP','HMDT','TPRT','slope_mapped', 'US_based_predicted_personnel',
    #                           'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
    #                           'gis_actual_slope', 'gis_dist_to_fire_station']
    # categorical_features_dummy = ['fuel_type_code', 'damage_class_code', 'cause_code',
    #                               'terrain_code', 'age_class_code', 'compass_direction',
    #                               'gis_fuel_category_detailed']
    # dummy_X_gbrt_cols = numeric_features_dummy + categorical_features_dummy
    # dummy_X_gbrt_data = {}
    # for col in numeric_features_dummy:
    #     dummy_X_gbrt_data[col] = np.random.rand(20)
    # for col in categorical_features_dummy:
    #     dummy_X_gbrt_data[col] = np.random.choice(['TypeA', 'TypeB', 'Unknown'], size=20)
    
    # dummy_X_gbrt = pd.DataFrame(dummy_X_gbrt_data)
    # dummy_y_gbrt = pd.Series(np.random.rand(20) * 100)

    # numeric_transformer_dummy = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    # categorical_transformer_dummy = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    # preprocessor_dummy = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer_dummy, numeric_features_dummy),
    #         ('cat', categorical_transformer_dummy, categorical_features_dummy)
    #     ], remainder='passthrough') # passthrough로 변경하여 GBRT에 모든 특성 전달되도록
    # dummy_gbrt_full_pipeline = Pipeline(steps=[('preprocessor', preprocessor_dummy), ('regressor', GradientBoostingRegressor(random_state=42))])
    
    # try:
    #     dummy_gbrt_full_pipeline.fit(dummy_X_gbrt, np.log1p(dummy_y_gbrt)) # 실제 모델과 동일하게 로그 변환된 y로 학습
    #     joblib.dump(dummy_gbrt_full_pipeline, KOREAN_GBRT_MODEL_PATH)
    #     print(f"더미 GBRT 모델 '{KOREAN_GBRT_MODEL_PATH}' 생성 및 저장 완료.")
    # except Exception as e:
    #     print(f"더미 GBRT 모델 생성 중 오류: {e}")
    #     import traceback
    #     traceback.print_exc()


    # dummy_linear = LinearRegression().fit(np.random.rand(10,1).reshape(-1,1), np.random.rand(10))
    # joblib.dump(dummy_linear, US_LINEAR_MODEL_PATH)
    # print(f"더미 Linear 모델 '{US_LINEAR_MODEL_PATH}' 생성 완료.")

    main()
