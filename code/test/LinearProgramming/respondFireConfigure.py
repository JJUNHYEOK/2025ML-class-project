import pandas as pd
import numpy as np
import joblib
import random
import math
from typing import Dict, List, Tuple
import pulp

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# from ml import add_us_based_personnel_prediction # ml.py의 함수를 직접 임포트하려면 경로 설정 또는 패키지화 필요

# 모델 및 특성명 파일 경로 (ml.py에서 저장한 경로와 일치)
KOREAN_GBRT_MODEL_PATH = 'code/test/LinearProgramming/model/korean_gbrt_demand_model.joblib'
US_LINEAR_MODEL_PATH = 'code/test/LinearProgramming/model/us_acres_personnel_linear_model.joblib'
GBRT_FEATURE_NAMES_PATH = 'code/test/LinearProgramming/model/gbrt_trained_feature_names.txt' # 학습된 특성명 리스트 파일 code\test\LinearProgramming\model\gbrt_trained_feature_names.txt

# --- 0. (ml.py에서 복사 또는 임포트) add_us_based_personnel_prediction 함수 ---
# respondFireConfigure.py가 ml.py와 다른 환경에서 실행될 경우,
# 이 함수를 여기에 직접 정의하거나, ml.py를 import 할 수 있도록 경로 설정 필요.
# 여기서는 간단히 ml.py의 함수와 동일한 로직을 복사한다고 가정.
def add_us_based_personnel_prediction_local(df_kr_processed, us_model, acres_col_kr='FRFR_DMG_AREA'):
    if us_model is None or df_kr_processed is None:
        if df_kr_processed is not None: df_kr_processed['US_based_predicted_personnel'] = 0
        return df_kr_processed
    if acres_col_kr not in df_kr_processed.columns:
        df_kr_processed['US_based_predicted_personnel'] = 0
        return df_kr_processed
    df_with_prediction = df_kr_processed.copy()
    hectare_to_acre = 2.47105
    acres_values_kr_ha = pd.to_numeric(df_with_prediction[acres_col_kr], errors='coerce').fillna(0)
    acres_values_kr_ha = np.maximum(0, acres_values_kr_ha)
    acres_kr = acres_values_kr_ha * hectare_to_acre
    log_acres_kr = np.log1p(acres_kr)
    X_for_us_model_kr = pd.DataFrame({'log_ACRES': log_acres_kr})
    try:
        log_predicted_personnel_us_based = us_model.predict(X_for_us_model_kr)
        predicted_personnel_us_based = np.expm1(log_predicted_personnel_us_based)
        df_with_prediction['US_based_predicted_personnel'] = np.maximum(0, predicted_personnel_us_based).round().astype(int)
    except Exception: df_with_prediction['US_based_predicted_personnel'] = 0
    return df_with_prediction

# --- 1. 데이터 로드 및 시나리오 생성을 위한 기본 전처리 ---
def load_and_preprocess_data_for_scenario(korea_data_file_path):
    try:
        df_kr_raw = pd.read_csv(korea_data_file_path, encoding='UTF-8')
        print(f"시나리오 생성용 한국 데이터 로드 성공: {df_kr_raw.shape}")
    except FileNotFoundError:
        print(f"오류: 시나리오 생성용 한국 데이터 파일을 찾을 수 없습니다 - {korea_data_file_path}")
        return None
    except Exception as e:
        print(f"시나리오 생성용 한국 데이터 로드 중 오류 발생: {e}")
        return None

    df_processed = df_kr_raw.copy()

    # 목표 변수 및 주요 수치형 특성 기본 처리 (ml.py와 유사하게)
    cols_to_process_numeric_scenario = ['POTFR_RSRC_INPT_QNTT', 'FRFR_DMG_AREA', 'WDSP', 'HMDT', 
                               'TPRT', 'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                               'FRTP_TRE_HGHT', 'HASLV', 'PRCPT_QNTT'] # FRFR_POTFR_TM은 아래에서 minutes로
    for col in cols_to_process_numeric_scenario:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # 결측치는 나중에 imputer로
            if col != 'TPRT': # 기온은 음수 가능
                 df_processed[col] = df_processed[col].apply(lambda x: max(x, 0) if pd.notnull(x) else np.nan)
        else:
            df_processed[col] = np.nan # 없는 컬럼은 NaN으로 (클러스터링/평균 계산 시 제외)

    # FRFR_POTFR_TM (산불진화시간) -> minutes (ml.py와 동일 로직)
    if 'FRFR_POTFR_TM' in df_processed.columns:
        def time_to_minutes_scenario(time_str):
            if pd.isna(time_str) or not isinstance(time_str, str) or ':' not in time_str: return np.nan
            try:
                parts = time_str.split(':'); h, m = map(int, parts[:2]); return h * 60 + m
            except ValueError: return np.nan
        df_processed['FRFR_POTFR_TM_minutes'] = df_processed['FRFR_POTFR_TM'].apply(time_to_minutes_scenario)
    else:
        df_processed['FRFR_POTFR_TM_minutes'] = np.nan

    # 코드값 매핑 (ml.py와 동일한 로직)
    def map_dnst_cd_scenario(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).strip().upper();
        if val_str == 'A': return 1.0;
        if val_str == 'B': return 2.0;
        if val_str == 'C': return 3.0;
        try: return float(val_str)
        except ValueError: return np.nan
    if 'DNST_CD' in df_processed.columns: df_processed['density_mapped'] = df_processed['DNST_CD'].apply(map_dnst_cd_scenario)
    else: df_processed['density_mapped'] = np.nan
        
    def map_frtp_cd_scenario(val):
        if pd.isna(val) or str(val).strip() == '': return 'Unknown'
        val_str = str(val).split('.')[0];
        if val_str == '1': return '침엽수림';
        if val_str == '2': return '활엽수림';
        if val_str == '3': return '혼효림';
        if val_str == '4': return '죽림';
        return 'Unknown'
    if 'FRTP_CD' in df_processed.columns: df_processed['fuel_type'] = df_processed['FRTP_CD'].apply(map_frtp_cd_scenario)
    else: df_processed['fuel_type'] = 'Unknown'

    def map_dmcls_cd_scenario(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).split('.')[0];
        if val_str == '0': return 0.0;
        if val_str == '1': return 1.0;
        if val_str == '2': return 2.0;
        if val_str == '3': return 3.0;
        return np.nan
    if 'DMCLS_CD' in df_processed.columns: df_processed['tree_diameter_class'] = df_processed['DMCLS_CD'].apply(map_dmcls_cd_scenario)
    else: df_processed['tree_diameter_class'] = np.nan

    if 'AGCLS_CD' in df_processed.columns: df_processed['age_class'] = pd.to_numeric(df_processed['AGCLS_CD'], errors='coerce')
    else: df_processed['age_class'] = np.nan

    if 'STORUNST_CD' in df_processed.columns: df_processed['is_forest'] = pd.to_numeric(df_processed['STORUNST_CD'], errors='coerce').fillna(0).astype(int)
    else: df_processed['is_forest'] = 0

    if 'FRFR_OCCRR_CAUS_NM' in df_processed.columns: df_processed['cause'] = df_processed['FRFR_OCCRR_CAUS_NM'].astype(str).fillna('Unknown')
    else: df_processed['cause'] = 'Unknown'

    for col_bool_scenario in ['CMPSI_FG', 'PTMNT_FG', 'CMTRY_FG']:
        if col_bool_scenario in df_processed.columns: df_processed[col_bool_scenario] = pd.to_numeric(df_processed[col_bool_scenario], errors='coerce').fillna(0).astype(int)
        else: df_processed[col_bool_scenario] = 0

    if 'OCCRR_MNT' in df_processed.columns: df_processed['month'] = pd.to_numeric(df_processed['OCCRR_MNT'], errors='coerce')
    else: df_processed['month'] = np.nan
    if 'OCCRR_DYWK_NM' in df_processed.columns: df_processed['day_of_week'] = df_processed['OCCRR_DYWK_NM'].astype(str).fillna('Unknown')
    else: df_processed['day_of_week'] = 'Unknown'
    
    # 임시 GIS 특성 (ml.py와 동일하게)
    df_processed['gis_actual_slope'] = np.random.uniform(0, 45, size=len(df_processed))
    df_processed['gis_fuel_category_detailed'] = np.random.choice(['침엽수밀집', '활엽수밀집', '혼효림', '초지'], size=len(df_processed))
    df_processed['gis_dist_to_fire_station'] = np.random.uniform(100, 15000, size=len(df_processed))
    
    # US_based_predicted_personnel 특성 추가
    try:
        us_model_loaded_scenario = joblib.load(US_LINEAR_MODEL_PATH)
        df_processed = add_us_based_personnel_prediction_local(df_processed, us_model_loaded_scenario)
    except Exception as e_us_load_scenario:
        print(f"시나리오 생성용 데이터 전처리 중 US 모델 로드/적용 오류: {e_us_load_scenario}")
        df_processed['US_based_predicted_personnel'] = 0

    print(f"시나리오 생성용 데이터 최종 전처리 후 shape: {df_processed.shape}")
    return df_processed


# --- 2. K-means 클러스터링 및 시나리오 생성 함수 ---
def create_fire_scenarios_by_clustering(df_for_clustering, n_scenarios=5, random_state=42):
    clustering_features_scenario = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 
                           'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                           'HASLV', 'density_mapped', 'FRFR_POTFR_TM_minutes'] # 예시
    
    existing_clustering_features_scenario = [col for col in clustering_features_scenario if col in df_for_clustering.columns and df_for_clustering[col].notna().sum() > 1]
    
    if len(existing_clustering_features_scenario) < 2:
        print(f"오류: 클러스터링에 사용할 유효한 특성이 부족합니다 (현재: {existing_clustering_features_scenario}).")
        return None

    X_cluster_scenario = df_for_clustering[existing_clustering_features_scenario].copy()
    
    imputer_cluster_scenario = SimpleImputer(strategy='median')
    X_cluster_imputed_scenario = imputer_cluster_scenario.fit_transform(X_cluster_scenario)
    
    scaler_cluster_scenario = StandardScaler()
    X_scaled_scenario = scaler_cluster_scenario.fit_transform(X_cluster_imputed_scenario)
    
    n_clusters_actual_scenario = min(n_scenarios, len(X_scaled_scenario))
    if n_clusters_actual_scenario <= 1 : # 클러스터 수가 1 이하면 의미 없음
        print(f"경고: 클러스터링을 위한 유효 데이터 부족 또는 n_scenarios가 너무 작아 클러스터링을 수행하지 않습니다 (n_clusters_actual={n_clusters_actual_scenario}). 단일 시나리오로 처리합니다.")
        if len(df_for_clustering) > 0:
            return np.zeros(len(df_for_clustering), dtype=int) # 모든 데이터를 0번 클러스터로
        else:
            return None

    kmeans_scenario = KMeans(n_clusters=n_clusters_actual_scenario, random_state=random_state, n_init='auto')
    try:
        clusters_result = kmeans_scenario.fit_predict(X_scaled_scenario)
    except Exception as e_kmeans:
        print(f"K-means 클러스터링 중 오류 발생: {e_kmeans}")
        return None
        
    return clusters_result

def generate_scenarios_from_data(df_processed_scenario_input, n_scenarios=5):
    if df_processed_scenario_input is None or df_processed_scenario_input.empty:
        return []

    clusters = create_fire_scenarios_by_clustering(df_processed_scenario_input, n_scenarios=n_scenarios)
    
    if clusters is None:
        return []
    
    df_with_cluster_scenario = df_processed_scenario_input.copy()
    df_with_cluster_scenario['cluster_id_for_scenario'] = clusters

    actual_n_clusters_gen = len(np.unique(clusters))
    print(f"실제 생성된 클러스터 수 (generate_scenarios): {actual_n_clusters_gen}")
    cluster_stats_list_gen = []

    # GBRT 모델 학습 시 사용된 "원본 형태의" 특성명 리스트 (ml.py에서 정의된 것과 일치해야 함)
    # 이 리스트는 ml.py의 existing_numeric_features + existing_categorical_features 와 동일해야 함.
    # 여기서는 예시로 ml.py에 정의된 리스트와 유사하게 작성.
    # 실제로는 ml.py에서 이 리스트를 저장하고 여기서 로드하는 것이 가장 좋음.
    gbrt_original_numeric_features = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT',
                                    'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                                    'FRTP_TRE_HGHT', 'HASLV', 'PRCPT_QNTT', 'FRFR_POTFR_TM_minutes',
                                    'density_mapped', 'tree_diameter_class', 'age_class',
                                    'is_forest', 'CMPSI_FG', 'PTMNT_FG', 'CMTRY_FG', 'month',
                                    'US_based_predicted_personnel',
                                    'gis_actual_slope', 'gis_dist_to_fire_station']
    gbrt_original_categorical_features = ['fuel_type', 'cause', 'day_of_week',
                                        'gis_fuel_category_detailed']
    all_gbrt_original_features = gbrt_original_numeric_features + gbrt_original_categorical_features


    for cluster_idx_gen in range(actual_n_clusters_gen):
        current_cluster_data = df_with_cluster_scenario[df_with_cluster_scenario['cluster_id_for_scenario'] == cluster_idx_gen]
        if current_cluster_data.empty:
            continue
            
        stats_for_scenario = {}
        for col_key_stat in all_gbrt_original_features:
            if col_key_stat in current_cluster_data.columns:
                if pd.api.types.is_numeric_dtype(current_cluster_data[col_key_stat]):
                    stats_for_scenario[col_key_stat] = float(current_cluster_data[col_key_stat].mean(skipna=True))
                else:
                    mode_val_stat = current_cluster_data[col_key_stat].mode()
                    stats_for_scenario[col_key_stat] = mode_val_stat[0] if not mode_val_stat.empty else 'Unknown'
            else: # 원본 특성이 없으면 기본값
                stats_for_scenario[col_key_stat] = 'Unknown' if col_key_stat in gbrt_original_categorical_features else 0.0
        
        stats_for_scenario['required_resources_cluster_avg'] = float(current_cluster_data['POTFR_RSRC_INPT_QNTT'].mean(skipna=True))
        stats_for_scenario['damage_area'] = float(current_cluster_data['FRFR_DMG_AREA'].mean(skipna=True)) # FireScenario의 damage_area
        stats_for_scenario['probability'] = len(current_cluster_data) / len(df_processed_scenario_input)
        
        cluster_stats_list_gen.append(stats_for_scenario)
    
    if not cluster_stats_list_gen: return []
        
    scenarios_output = []
    for i_scen, stats_dict_scen in enumerate(cluster_stats_list_gen):
        scenario_obj = FireScenario(
            scenario_id=i_scen,
            probability=stats_dict_scen.get('probability', 0),
            cluster_stats=stats_dict_scen
        )
        scenarios_output.append(scenario_obj)
    print(f"{len(scenarios_output)}개의 시나리오 생성 완료.")
    return scenarios_output

# --- 3. FireScenario 클래스 개선 ---
class FireScenario:
    # GBRT 모델 학습 시 사용된 원본 특성명 리스트 (ml.py 실행 후 생성된 gbrt_trained_feature_names.txt 에서 로드)
    _gbrt_input_feature_names_ordered = None # 클래스 변수로 선언

    @classmethod
    def load_gbrt_feature_names(cls, path=GBRT_FEATURE_NAMES_PATH):
        if cls._gbrt_input_feature_names_ordered is None: # 한 번만 로드
            try:
                with open(path, 'r', encoding='utf-8') as f: # 인코딩 추가
                    # 이 파일은 preprocessor.get_feature_names_out()의 결과 (원핫인코딩된 이름 포함)
                    # 하지만 우리가 GBRT 파이프라인에 넣을 때는 원본 컬럼명으로 넣어야 함.
                    # 따라서, 이 파일 대신 ml.py의 existing_numeric_features + existing_categorical_features 리스트를 사용해야 함.
                    # 여기서는 임시로 ml.py 에서 사용된 원본 특성명 리스트를 하드코딩하거나, 별도 파일로 관리
                    # (가장 좋은 것은 ml.py 실행 시 이 "원본" 특성명 리스트를 저장하는 것)
                    cls._gbrt_input_feature_names_original_order = [ # ml.py의 existing_numeric_features + existing_categorical_features 순서
                        'FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT',
                        'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                        'FRTP_TRE_HGHT', 'HASLV', 'PRCPT_QNTT', 'FRFR_POTFR_TM_minutes',
                        'density_mapped', 'tree_diameter_class', 'age_class',
                        'is_forest', 'CMPSI_FG', 'PTMNT_FG', 'CMTRY_FG', 'month',
                        'US_based_predicted_personnel',
                        'gis_actual_slope', 'gis_dist_to_fire_station', # Numeric 끝
                        'fuel_type', 'cause', 'day_of_week', # Categorical 시작
                        'gis_fuel_category_detailed'
                    ]
                    print(f"GBRT 예측용 원본 특성명 로드 완료 (개수: {len(cls._gbrt_input_feature_names_original_order)})")

            except FileNotFoundError:
                print(f"경고: GBRT 특성명 파일({path})을 찾을 수 없습니다. 예측 시 컬럼 순서 문제가 발생할 수 있습니다.")
                # 이 경우, 위에서 하드코딩한 리스트가 사용됨.
            except Exception as e_feat_load:
                print(f"GBRT 특성명 파일 로드 중 오류: {e_feat_load}")


    def __init__(self, scenario_id: int, probability: float, cluster_stats: Dict):
        self.id = scenario_id
        self.probability = probability
        self.cluster_stats = cluster_stats
        self.base_station = {'name': '기준 소방서', 'latitude': 35.18035823746264, 'longitude': 128.11851962302458}
        
        # 모델 및 특성명 로드는 한 번만 (클래스 레벨에서)
        FireScenario.load_gbrt_feature_names() 

        try:
            self.korean_gbrt_model_pipeline = joblib.load(KOREAN_GBRT_MODEL_PATH)
        except Exception as e_gbrt_load:
            print(f"한국 GBRT 모델 로드 실패 ({KOREAN_GBRT_MODEL_PATH}): {e_gbrt_load}")
            self.korean_gbrt_model_pipeline = None
        try:
            self.us_linear_model = joblib.load(US_LINEAR_MODEL_PATH)
        except Exception as e_us_load:
            print(f"미국 Linear 모델 로드 실패 ({US_LINEAR_MODEL_PATH}): {e_us_load}")
            self.us_linear_model = None
            
        self.sites = self._generate_sites_from_stats()

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371; lat1_r,lon1_r,lat2_r,lon2_r = map(math.radians,[lat1,lon1,lat2,lon2])
        dlat=lat2_r-lat1_r; dlon=lon2_r-lon1_r
        a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _generate_sites_from_stats(self) -> Dict[str, Dict]:
        sites = {}
        scenario_avg_total_damage_ha = self.cluster_stats.get('damage_area', 1.0)
        
        if scenario_avg_total_damage_ha <= 1.0: num_sites_val = 1
        elif scenario_avg_total_damage_ha <= 10.0: num_sites_val = random.randint(1, 2)
        else: num_sites_val = random.randint(2, 3)
            
        for i_site in range(num_sites_val):
            site_id_val = f'site{i_site+1}'
            # --- site_features_dict 선언 및 초기화 ---
            site_features_dict_for_gbrt = {} 

            if num_sites_val == 1: site_damage_area_ha_val = max(0.01, scenario_avg_total_damage_ha)
            else:
                base_area = scenario_avg_total_damage_ha / num_sites_val
                site_damage_area_ha_val = max(0.01, base_area + random.uniform(-0.1, 0.1) * base_area)
            
            site_features_dict_for_gbrt['FRFR_DMG_AREA'] = site_damage_area_ha_val
            
            # GBRT 입력 특성 구성 (self.cluster_stats 사용)
            # FireScenario._gbrt_input_feature_names_original_order 에 있는 모든 특성에 대해 값을 채움
            if FireScenario._gbrt_input_feature_names_original_order:
                for key_gbrt in FireScenario._gbrt_input_feature_names_original_order:
                    if key_gbrt == 'FRFR_DMG_AREA': continue # 이미 위에서 설정
                    if key_gbrt == 'US_based_predicted_personnel': continue # 아래에서 재계산
                    
                    # 기본값 설정 (ml.py의 existing_numeric_features/categorical_features 참고)
                    is_categorical_key_site = key_gbrt in ['fuel_type', 'cause', 'day_of_week', 'gis_fuel_category_detailed']
                    default_val_site = 'Unknown' if is_categorical_key_site else 0.0
                    site_features_dict_for_gbrt[key_gbrt] = self.cluster_stats.get(key_gbrt, default_val_site)
            else: # 특성명 리스트 로드 실패 시, cluster_stats의 모든 키를 사용 (위험)
                print("경고: GBRT 특성명 리스트가 없어 cluster_stats의 모든 키를 사용합니다. 순서 문제 발생 가능.")
                for key_cs, val_cs in self.cluster_stats.items():
                    if key_cs not in ['FRFR_DMG_AREA', 'US_based_predicted_personnel', 
                                      'required_resources_cluster_avg', 'damage_area', 'probability']: # 시스템용 키 제외
                        site_features_dict_for_gbrt[key_cs] = val_cs


            if self.us_linear_model:
                current_site_dmg_area_us = site_features_dict_for_gbrt.get('FRFR_DMG_AREA', 0.0)
                acres_kr_site_us = current_site_dmg_area_us * 2.47105
                log_acres_kr_site_us = np.log1p(max(0, acres_kr_site_us))
                X_us_site = pd.DataFrame({'log_ACRES': [log_acres_kr_site_us]})
                try:
                    log_pred_us = self.us_linear_model.predict(X_us_site)
                    pred_us = np.expm1(log_pred_us[0])
                    site_features_dict_for_gbrt['US_based_predicted_personnel'] = max(0, int(round(pred_us)))
                except Exception: site_features_dict_for_gbrt['US_based_predicted_personnel'] = 0
            else: site_features_dict_for_gbrt['US_based_predicted_personnel'] = 0

            # GBRT 예측을 위한 DataFrame 생성 (ml.py 학습 시 컬럼 순서와 일치하도록)
            # site_features_dict_for_gbrt의 키들이 FireScenario._gbrt_input_feature_names_original_order의 모든 요소를 포함하고 있어야 함.
            # 누락된 키는 기본값(0 또는 'Unknown')으로 채워야 함.
            final_features_for_df = {}
            if FireScenario._gbrt_input_feature_names_original_order:
                for feature_name_model in FireScenario._gbrt_input_feature_names_original_order:
                    is_categorical_key_final = feature_name_model in ['fuel_type', 'cause', 'day_of_week', 'gis_fuel_category_detailed']
                    default_val_final = 'Unknown' if is_categorical_key_final else 0.0
                    final_features_for_df[feature_name_model] = site_features_dict_for_gbrt.get(feature_name_model, default_val_final)
                site_features_df_for_pred = pd.DataFrame([final_features_for_df])[FireScenario._gbrt_input_feature_names_original_order] # 순서 보장
            else: # 특성명 리스트 로드 실패 시, dict 그대로 사용 (위험)
                site_features_df_for_pred = pd.DataFrame([site_features_dict_for_gbrt])


            predicted_demand_gbrt_raw_val = 5 
            predicted_demand_gbrt_final_val = 5
            if self.korean_gbrt_model_pipeline:
                try:
                    log_pred_potfr = self.korean_gbrt_model_pipeline.predict(site_features_df_for_pred)
                    pred_potfr = np.expm1(log_pred_potfr[0])
                    predicted_demand_gbrt_raw_val = max(1, int(round(pred_potfr)))
                    
                    current_site_area_cap = site_features_dict_for_gbrt.get('FRFR_DMG_AREA', 1.0)
                    if current_site_area_cap <= 1.0: MAX_DEMAND_SITE = random.randint(10, 20)
                    elif current_site_area_cap <= 5.0: MAX_DEMAND_SITE = random.randint(15, 30)
                    else: MAX_DEMAND_SITE = random.randint(25, 40)
                    
                    predicted_demand_gbrt_final_val = min(predicted_demand_gbrt_raw_val, MAX_DEMAND_SITE)
                    predicted_demand_gbrt_final_val = max(5, predicted_demand_gbrt_final_val)
                except Exception as e_gbrt_site_pred:
                    print(f"Scenario {self.id}, Site {site_id_val} GBRT 수요 예측 중 오류: {e_gbrt_site_pred}")
                    predicted_demand_gbrt_final_val = max(5, int(self.cluster_stats.get('required_resources_cluster_avg', 5)))
                    predicted_demand_gbrt_final_val = min(predicted_demand_gbrt_final_val, 40)
            else:
                predicted_demand_gbrt_final_val = max(5, int(self.cluster_stats.get('required_resources_cluster_avg', 5)))
                predicted_demand_gbrt_final_val = min(predicted_demand_gbrt_final_val, 40)

            print(f"Scenario {self.id}, Site {site_id_val}: 면적={site_damage_area_ha_val:.2f}ha, "
                  f"GBRT예측(Raw)={predicted_demand_gbrt_raw_val}, 최종수요(capped)={predicted_demand_gbrt_final_val}")
            
            site_lat_val = random.uniform(35.10, 35.28)
            site_lon_val = random.uniform(128.01, 128.18)
            base_dist = self._calculate_distance(self.base_station['latitude'], self.base_station['longitude'], site_lat_val, site_lon_val)
            site_dists = {}
            truck_types_site = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
            for truck_type_s in truck_types_site:
                dist_var_factor = random.uniform(0.95, 1.15) 
                actual_dist_s = base_dist * dist_var_factor
                site_dists[truck_type_s] = max(1, min(50, actual_dist_s))

            sites[site_id_val] = {
                'demand': predicted_demand_gbrt_final_val, 'distance': site_dists,
                'risk_factors': { 
                    'wind_speed': site_features_dict_for_gbrt.get('WDSP', 0.0),
                    'fuel_type': str(site_features_dict_for_gbrt.get('fuel_type', 'Unknown')),
                    'slope': site_features_dict_for_gbrt.get('density_mapped', 0.0), # 또는 gis_actual_slope
                    'humidity': site_features_dict_for_gbrt.get('HMDT', 100.0),
                    'damage_class': str(site_features_dict_for_gbrt.get('tree_diameter_class', 'Unknown'))
                },
                'latitude': site_lat_val, 'longitude': site_lon_val,
                'predicted_damage_area_ha': site_features_dict_for_gbrt.get('FRFR_DMG_AREA', 0.0)
            }
        return sites

# --- 4. ResourceAllocator 클래스 개선 (이전과 유사, 상수값 현실화) ---
class ResourceAllocator:
    def __init__(self):
        self.truck_types = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
        self.firefighter_types = ['FF_TypeA', 'FF_TypeB'] # 예시: 소방관 타입 단순화

        self.truck_capabilities = {
            'FT1': {'capacity': 2800, 'personnel': 3, 'fuel_efficiency': 4, 'speed': 40, 'cost': 300}, # 인원/비용 등 현실화
            'FT2': {'capacity': 3000, 'personnel': 3, 'fuel_efficiency': 5, 'speed': 40, 'cost': 320},
            'FT3': {'capacity': 1500, 'personnel': 2, 'fuel_efficiency': 6, 'speed': 50, 'cost': 250}, # 소형펌프차
            'FT4': {'capacity': 4500, 'personnel': 4, 'fuel_efficiency': 3, 'speed': 35, 'cost': 400}, # 대형물탱크
            'FT5': {'capacity': 0, 'personnel': 5, 'fuel_efficiency': 7, 'speed': 60, 'cost': 150},    # 지휘차/인력수송
            'FT6': {'capacity': 800, 'personnel': 2, 'fuel_efficiency': 5, 'speed': 45, 'cost': 200}     # 산악용소형
        }
        self.firefighter_capabilities = { # 타입별로 다른 능력치나 비용 설정 가능
            'FF_TypeA': {'max_deployments': 20, 'cost': 10}, # 예: 정규 소방관
            'FF_TypeB': {'max_deployments': 30, 'cost': 5}   # 예: 지원 인력 (의용소방대 등)
        }
        self.MAX_TRUCKS_PER_TYPE_TOTAL = 2 # 각 트럭 타입별 실제 보유 대수
        self.MAX_FF_TOTAL_DEPLOYABLE = 40 # 전체 시나리오에서 동원 가능한 추가 소방관 총원 한계
        
        # 자원 배치 현황을 저장할 딕셔너리 추가
        self.deployed_resources = {
            'truck': {truck_type: 0 for truck_type in self.truck_types},
            'firefighter': {ff_type: 0 for ff_type in self.firefighter_types}
        }

    def set_resource_deployment(self, resource_type: str, resource_id: str, quantity: int):
        """자원 배치 수량을 설정하는 메서드"""
        if resource_type not in ['truck', 'firefighter']:
            raise ValueError(f"잘못된 자원 타입: {resource_type}")
        
        if resource_type == 'truck' and resource_id not in self.truck_types:
            raise ValueError(f"잘못된 트럭 타입: {resource_id}")
        elif resource_type == 'firefighter' and resource_id not in self.firefighter_types:
            raise ValueError(f"잘못된 소방관 타입: {resource_id}")
        
        if quantity < 0:
            raise ValueError(f"배치 수량은 0 이상이어야 합니다: {quantity}")
        
        if resource_type == 'truck' and quantity > self.MAX_TRUCKS_PER_TYPE_TOTAL:
            quantity = self.MAX_TRUCKS_PER_TYPE_TOTAL
        elif resource_type == 'firefighter' and quantity > self.firefighter_capabilities[resource_id]['max_deployments']:
            quantity = self.firefighter_capabilities[resource_id]['max_deployments']
        
        self.deployed_resources[resource_type][resource_id] = quantity

    def get_available_resources(self, resource_type: str, resource_id: str) -> int:
        """현재 가용 가능한 자원 수량을 반환하는 메서드"""
        if resource_type not in ['truck', 'firefighter']:
            raise ValueError(f"잘못된 자원 타입: {resource_type}")
        
        if resource_type == 'truck':
            return self.MAX_TRUCKS_PER_TYPE_TOTAL - self.deployed_resources[resource_type][resource_id]
        else:
            return self.firefighter_capabilities[resource_id]['max_deployments'] - self.deployed_resources[resource_type][resource_id]

    def optimize_single_scenario(self, scenario: FireScenario) -> Tuple[List[Dict], float]:
        available_trucks_scen = {
            truck_type: self.MAX_TRUCKS_PER_TYPE_TOTAL for truck_type in self.truck_types
        }
        available_ff_type_scen = {
            ff_type: self.firefighter_capabilities[ff_type]['max_deployments'] for ff_type in self.firefighter_types
        }
        
        total_demand_scen = sum(site_info['demand'] for site_info in scenario.sites.values())
        
        theoretic_max_pers_trucks = sum(self.truck_capabilities[tt]['personnel'] * available_trucks_scen[tt] for tt in self.truck_types)
        theoretic_max_pers_ff = sum(available_ff_type_scen.values())
        
        print(f"시나리오 {scenario.id}: 총 수요={total_demand_scen}, "
              f"이론상 최대 트럭인원={theoretic_max_pers_trucks}, "
              f"이론상 최대 추가대원={theoretic_max_pers_ff} (타입별 합, 전체 한도: {self.MAX_FF_TOTAL_DEPLOYABLE})")

        model = pulp.LpProblem(f"Fire_Allocation_Scen_{scenario.id}", pulp.LpMinimize)
        
        x_truck = pulp.LpVariable.dicts("truck", [(i,n) for i in self.truck_types for n in scenario.sites.keys()], cat='Integer', lowBound=0)
        y_ff = pulp.LpVariable.dicts("ff", [(j,n) for j in self.firefighter_types for n in scenario.sites.keys()], cat='Integer', lowBound=0)
        z_truck_site = pulp.LpVariable.dicts("z_truck_site", [(i,n) for i in self.truck_types for n in scenario.sites.keys()], cat='Binary')

        model += pulp.lpSum([self.truck_capabilities[i]['cost'] * x_truck[(i,n)] for i in self.truck_types for n in scenario.sites.keys()] +
                           [self.firefighter_capabilities[j]['cost'] * y_ff[(j,n)] for j in self.firefighter_types for n in scenario.sites.keys()]), "Total_Cost"

        for i in self.truck_types: # 1. 트럭 타입별 총 배치 대수 <= 가용 대수
            model += pulp.lpSum(x_truck[(i,n)] for n in scenario.sites.keys()) <= available_trucks_scen[i], f"MaxTrucks_{i}"
        
        # for i in self.truck_types: # 2. (선택적) 트럭 타입 i는 최대 한 곳의 지점만
        #     model += pulp.lpSum(z_truck_site[(i,n)] for n in scenario.sites.keys()) <= 1, f"TruckType_{i}_OneSite"

        for i in self.truck_types: # 3. x와 z 연결
            for n in scenario.sites.keys():
                model += x_truck[(i,n)] <= available_trucks_scen[i] * z_truck_site[(i,n)], f"XZ_Link_{i}_{n}"
                # model += x_truck[(i,n)] >= z_truck_site[(i,n)] # 최소 1대 강제 시

        for j in self.firefighter_types: # 4. 소방관 타입별 총 배치 인원 <= 가용 인원
            model += pulp.lpSum(y_ff[(j,n)] for n in scenario.sites.keys()) <= available_ff_type_scen[j], f"MaxFF_{j}"
        
        # 4b. 전체 추가 소방관 총원 제한
        model += pulp.lpSum(y_ff[(j,n)] for j in self.firefighter_types for n in scenario.sites.keys()) <= self.MAX_FF_TOTAL_DEPLOYABLE, "TotalMaxFF"

        for n in scenario.sites.keys(): # 5. 각 지점 수요 충족
            model += pulp.lpSum(self.truck_capabilities[i]['personnel'] * x_truck[(i,n)] for i in self.truck_types) + \
                     pulp.lpSum(y_ff[(j,n)] for j in self.firefighter_types) >= scenario.sites[n]['demand'], f"DemandMet_{n}"

        for n in scenario.sites.keys(): # 6. 추가 소방관 <= 트럭 탑승인원
            model += pulp.lpSum(y_ff[(j,n)] for j in self.firefighter_types) <= \
                     pulp.lpSum(self.truck_capabilities[i]['personnel'] * x_truck[(i,n)] for i in self.truck_types), f"FF_TruckRatio_{n}"
        
        MAX_TRAVEL_TIME_LP = 3.0 
        for i in self.truck_types: # 7. 이동 시간 제약
            for n in scenario.sites.keys():
                if self.truck_capabilities[i]['speed'] > 0:
                    travel_t = scenario.sites[n]['distance'][i] / self.truck_capabilities[i]['speed']
                    model += travel_t * z_truck_site[(i,n)] <= MAX_TRAVEL_TIME_LP, f"TravelTime_{i}_{n}"
                else: model += z_truck_site[(i,n)] == 0, f"NoDeployZeroSpeed_{i}_{n}"
        
        try:
            solver = pulp.PULP_CBC_CMD(msg=False)
            status = model.solve(solver)
            print(f"시나리오 {scenario.id} LP Solve Status: {pulp.LpStatus[status]}")
        except Exception as e_lp_solve:
            print(f"시나리오 {scenario.id}: LP 해결 중 예외: {e_lp_solve}")
            return [], float('inf')

        results_lp = []
        if model.status == pulp.LpStatusOptimal:
            # ... (결과 수집 로직, 이전과 유사하게 pulp.value(x_truck[...]) 사용) ...
            for i_res in self.truck_types:
                for n_res in scenario.sites.keys():
                    if pulp.value(x_truck[(i_res,n_res)]) > 0.1:
                        qty = int(round(pulp.value(x_truck[(i_res,n_res)])))
                        if qty > 0: results_lp.append({'scenario': scenario.id, 'type': i_res, 'location': n_res, 'resource_type': 'truck', 'quantity': qty, 'distance': scenario.sites[n_res]['distance'][i_res], 'latitude': scenario.sites[n_res]['latitude'], 'longitude': scenario.sites[n_res]['longitude']})
            for j_res in self.firefighter_types:
                for n_res in scenario.sites.keys():
                    if pulp.value(y_ff[(j_res,n_res)]) > 0.1:
                        qty = int(round(pulp.value(y_ff[(j_res,n_res)])))
                        if qty > 0: results_lp.append({'scenario': scenario.id, 'type': j_res, 'location': n_res, 'resource_type': 'firefighter', 'quantity': qty, 'distance': 0, 'latitude': scenario.sites[n_res]['latitude'], 'longitude': scenario.sites[n_res]['longitude']})
            return results_lp, pulp.value(model.objective)
        
        elif model.status == pulp.LpStatusInfeasible: print(f"시나리오 {scenario.id}: Infeasible.")
        else: print(f"시나리오 {scenario.id}: 최적해 못찾음. 상태: {pulp.LpStatus[model.status]}")
        return [], float('inf')

# --- 5. RiskCalculator 클래스 (이전과 유사하게, fuel_type, damage_class 키 문자열로) ---
class RiskCalculator:
    def __init__(self):
        self.risk_factors = {
            'wind_speed': {'weight': 0.3, 'thresholds': [(0,0.2),(5,0.4),(10,0.6),(15,0.8),(20,1.0)]},
            'humidity': {'weight': 0.2, 'thresholds': [(80,0.2),(60,0.4),(40,0.6),(20,0.8),(0,1.0)]},
            'fuel_type': {'weight': 0.2, 'values': { '침엽수림':0.8, '활엽수림':0.4, '혼효림':0.6, '죽림':0.5, 'Unknown':0.3}},
            'slope': {'weight': 0.15, 'thresholds': [(0,0.2),(10,0.4),(20,0.6),(30,0.8),(45,1.0)]},
            'damage_class': {'weight': 0.15, 'values': { '0.0':0.2,'1.0':0.4,'2.0':0.6,'3.0':0.8, 'Unknown':0.3 }} # tree_diameter_class 값 기준
        }
    # ... (calculate_risk_score, _calculate_continuous_score, get_risk_level, get_risk_factors_description 메소드는 이전과 거의 동일, float 변환 및 .get() 주의) ...
    def calculate_risk_score(self, risk_factors_input: Dict) -> float:
        total_score, total_weight = 0.0, 0.0
        for factor, value in risk_factors_input.items():
            if factor in self.risk_factors:
                info, weight = self.risk_factors[factor], self.risk_factors[factor]['weight']
                score = 0.0
                if 'thresholds' in info:
                    try: val_f = float(value)
                    except: val_f = 0.0
                    score = self._calculate_continuous_score(val_f, info['thresholds'])
                elif 'values' in info:
                    score = info['values'].get(str(value), 0.3)
                total_score += score * weight; total_weight += weight
        return round((total_score / total_weight) * 100 if total_weight > 0 else 0, 1)

    def _calculate_continuous_score(self, value: float, thresholds: List[Tuple[float, float]]) -> float:
        if not thresholds: return 0.0
        if value <= thresholds[0][0]: return thresholds[0][1]
        if value >= thresholds[-1][0]: return thresholds[-1][1]
        for i in range(len(thresholds) - 1):
            x1,y1=thresholds[i]; x2,y2=thresholds[i+1]
            if x1 <= value < x2: return y1 + (y2-y1)*(value-x1)/(x2-x1) if (x2-x1)!=0 else y1
        return thresholds[-1][1]

    def get_risk_level(self, score: float) -> str:
        if score >= 80: return "심각"
        elif score >=60: return "높음"
        elif score >=40: return "보통"
        elif score >= 20: return "낮음"
        else: return "매우 낮음"

    def get_risk_factors_description(self, risk_factors_input: Dict) -> List[str]:
        desc = []
        try:
            if float(risk_factors_input.get('wind_speed',0)) >=15: desc.append("강한 바람")
            if float(risk_factors_input.get('humidity',100)) <=30: desc.append("건조")
            if str(risk_factors_input.get('fuel_type','Unknown')) == '침엽수림': desc.append("연료(침엽수)")
            if float(risk_factors_input.get('slope',0)) >=30: desc.append("급경사")
            if str(risk_factors_input.get('damage_class','Unknown')) in ['2.0','3.0']: desc.append("큰나무")
        except: pass
        return desc
    
# --- 6. 메인 실행 함수 ---
def main():
    print("화재 대응 자원 배치 최적화 시스템 시작")
    korea_data_file_main = './datasets/WSQ000301.csv'
    
    df_processed_main = load_and_preprocess_data_for_scenario(korea_data_file_main)
    if df_processed_main is None: return

    num_scenarios_main = 3
    scenarios_main_list = generate_scenarios_from_data(df_processed_main, n_scenarios=num_scenarios_main)
    if not scenarios_main_list: return

    allocator_main = ResourceAllocator() 
    risk_calc_main = RiskCalculator() 
    all_results_main, total_weighted_cost_main = [], 0.0
    
    print("\n--- 시나리오별 최적화 수행 ---")
    for scen_obj in scenarios_main_list:
        if scen_obj.korean_gbrt_model_pipeline is None: 
            print(f"시나리오 {scen_obj.id}: GBRT 모델 로드 실패. 건너뜁니다.")
            continue
        
        if scen_obj.sites:
            first_site_key_main = list(scen_obj.sites.keys())[0]
            risk_factors_main = scen_obj.sites[first_site_key_main]['risk_factors']
            risk_score_main = risk_calc_main.calculate_risk_score(risk_factors_main)
            print(f"\n시나리오 {scen_obj.id} (P={scen_obj.probability:.3f}): 위험도={risk_score_main} ({risk_calc_main.get_risk_level(risk_score_main)}), "
                  f"위험요소: {risk_calc_main.get_risk_factors_description(risk_factors_main)}")
            for site_id_m, site_info_m in scen_obj.sites.items():
                 print(f"  Site {site_id_m}: 최종수요={site_info_m['demand']}, 면적={site_info_m.get('predicted_damage_area_ha','N/A'):.2f}ha")
        else:
            print(f"\n시나리오 {scen_obj.id} (P={scen_obj.probability:.3f}): 사이트 없음.")
            continue

        opt_results, scen_cost = allocator_main.optimize_single_scenario(scen_obj) 
        if opt_results:
            all_results_main.extend(opt_results)
            total_weighted_cost_main += scen_cost * scen_obj.probability 
            print(f"시나리오 {scen_obj.id} 최적화 완료. 비용: {scen_cost:.0f}")
        else:
            print(f"시나리오 {scen_obj.id} 최적화 실패/자원 없음.")

    print("\n\n--- 최종 결과 요약 ---")
    print(f"총 가중 평균 비용: {total_weighted_cost_main:,.0f}")
    print(f"총 배치 건수: {len(all_results_main)}")
    if all_results_main:
        print("\n상세 배치 내역 (상위 5건):"); print(pd.DataFrame(all_results_main).head())

if __name__ == "__main__":
    main()
