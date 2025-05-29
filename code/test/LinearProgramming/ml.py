import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# import shap # 필요시 주석 해제

# Matplotlib 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"한글 폰트 설정 중 오류 (무시하고 진행): {e}")
    pass

# --- 1. 미국 ICS-209-PLUS 데이터 로드 및 "규모-자원" 선형 회귀 모델 학습 ---
def train_us_acres_personnel_model(us_data_file_path, model_save_path='us_acres_personnel_linear_model.joblib'):
    try:
        df_us = pd.read_csv(us_data_file_path, low_memory=False)
        print(f"미국 데이터 로드 성공: {df_us.shape}")
    except FileNotFoundError:
        print(f"오류: 미국 데이터 파일을 찾을 수 없습니다 - {us_data_file_path}")
        return None
    except Exception as e:
        print(f"미국 데이터 로드 중 오류 발생: {e}")
        return None

    cols_us = ['ACRES', 'TOTAL_PERSONNEL']
    df_us_processed = df_us[cols_us].copy()
    df_us_processed['ACRES'] = pd.to_numeric(df_us_processed['ACRES'], errors='coerce').fillna(0)
    df_us_processed['TOTAL_PERSONNEL'] = pd.to_numeric(df_us_processed['TOTAL_PERSONNEL'], errors='coerce').fillna(0)
    
    df_us_processed = df_us_processed[(df_us_processed['ACRES'] > 0) & (df_us_processed['TOTAL_PERSONNEL'] > 0)]
    if df_us_processed.empty:
        print("오류: 미국 데이터 필터링 후 (ACRES > 0, TOTAL_PERSONNEL > 0) 데이터가 없습니다.")
        return None

    acres_upper_cap = df_us_processed['ACRES'].quantile(0.99)
    personnel_upper_cap = df_us_processed['TOTAL_PERSONNEL'].quantile(0.99)
    df_us_processed['ACRES_capped'] = np.clip(df_us_processed['ACRES'], 1, acres_upper_cap)
    df_us_processed['TOTAL_PERSONNEL_capped'] = np.clip(df_us_processed['TOTAL_PERSONNEL'], 1, personnel_upper_cap)
    
    print(f"미국 데이터 Capping 및 필터링 후 shape: {df_us_processed.shape}")

    df_us_processed.loc[:, 'log_ACRES'] = np.log1p(df_us_processed['ACRES_capped'])
    df_us_processed.loc[:, 'log_TOTAL_PERSONNEL'] = np.log1p(df_us_processed['TOTAL_PERSONNEL_capped'])

    X_us = df_us_processed[['log_ACRES']]
    y_us = df_us_processed['log_TOTAL_PERSONNEL']

    model = LinearRegression()
    model.fit(X_us, y_us)

    print("\n--- 미국 데이터 기반 '규모-자원' 선형 회귀 모델 학습 결과 (이상치 Capping 적용) ---")
    print(f"회귀 계수 (log_ACRES에 대한): {model.coef_[0]:.4f}")
    print(f"절편: {model.intercept_:.4f}")
    pred_us_train = model.predict(X_us)
    r2_us_train = r2_score(y_us, pred_us_train)
    print(f"학습 데이터 R-squared: {r2_us_train:.4f}")

    joblib.dump(model, model_save_path)
    print(f"학습된 미국 '규모-자원' 모델이 '{model_save_path}'에 저장되었습니다.")
    return model

# --- 2. 학습된 미국 모델을 사용하여 한국 데이터에 "미국 기준 예상 인력" 특성 추가 ---
def add_us_based_personnel_prediction(df_kr_processed, us_model, acres_col_kr='FRFR_DMG_AREA'):
    if us_model is None or df_kr_processed is None:
        print("오류: 미국 모델 또는 한국 데이터가 준비되지 않았습니다.")
        if df_kr_processed is not None:
            df_kr_processed['US_based_predicted_personnel'] = 0
        return df_kr_processed
    if acres_col_kr not in df_kr_processed.columns:
        print(f"오류: 한국 데이터에 면적 컬럼 '{acres_col_kr}'이 없습니다.")
        df_kr_processed['US_based_predicted_personnel'] = 0
        return df_kr_processed

    df_with_prediction = df_kr_processed.copy()
    hectare_to_acre = 2.47105
    
    acres_values_kr_ha = pd.to_numeric(df_with_prediction[acres_col_kr], errors='coerce').fillna(0)
    acres_values_kr_ha = np.maximum(0, acres_values_kr_ha) # 음수 0으로
    
    acres_kr = acres_values_kr_ha * hectare_to_acre
    log_acres_kr = np.log1p(acres_kr)

    X_for_us_model_kr = pd.DataFrame({'log_ACRES': log_acres_kr})
    
    try:
        log_predicted_personnel_us_based = us_model.predict(X_for_us_model_kr)
        predicted_personnel_us_based = np.expm1(log_predicted_personnel_us_based)
        df_with_prediction['US_based_predicted_personnel'] = np.maximum(0, predicted_personnel_us_based).round().astype(int)
        print("한국 데이터에 'US_based_predicted_personnel' 특성 추가 완료.")
    except Exception as e:
        print(f"'US_based_predicted_personnel' 예측 중 오류: {e}")
        df_with_prediction['US_based_predicted_personnel'] = 0

    return df_with_prediction

# --- 3. 한국 데이터 로드 및 최종 전처리 (컬럼 정의서 기반 개선) ---
def load_and_preprocess_korean_data_final(korea_data_file_path, target_col='POTFR_RSRC_INPT_QNTT', target_capping_quantile=0.995):
    try:
        df_kr_raw = pd.read_csv(korea_data_file_path, encoding='UTF-8')
        print(f"한국 원본 데이터 로드 성공: {df_kr_raw.shape}")
    except FileNotFoundError:
        print(f"오류: 한국 데이터 파일을 찾을 수 없습니다 - {korea_data_file_path}")
        return None
    except Exception as e:
        print(f"한국 데이터 로드 중 오류 발생: {e}")
        return None

    df_kr_processed = df_kr_raw.copy()

    df_kr_processed[target_col] = pd.to_numeric(df_kr_processed[target_col], errors='coerce').fillna(0)
    df_kr_processed[target_col] = df_kr_processed[target_col].apply(lambda x: max(x, 0))

    if target_capping_quantile is not None and 0 < target_capping_quantile < 1:
        upper_bound_target = df_kr_processed[target_col].quantile(target_capping_quantile)
        df_kr_processed[target_col] = np.clip(df_kr_processed[target_col], 0, upper_bound_target)
        print(f"목표 변수 '{target_col}'에 상위 {100*(1-target_capping_quantile):.1f}% Capping 적용 (상한값: {upper_bound_target:.2f})")

    numeric_cols_from_spec = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 
                              'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC', 
                              'FRTP_TRE_HGHT', 'HASLV', 'PRCPT_QNTT', 'FRFR_POTFR_TM']
    
    for col in numeric_cols_from_spec:
        if col in df_kr_processed.columns:
            df_kr_processed[col] = pd.to_numeric(df_kr_processed[col], errors='coerce')
            if col != 'TPRT':
                 df_kr_processed[col] = df_kr_processed[col].apply(lambda x: max(x, 0) if pd.notnull(x) else np.nan)
        else:
            print(f"경고: 한국 데이터에 수치형 특성 '{col}'이 없습니다. 해당 컬럼은 무시됩니다.")
            if col == 'FRFR_POTFR_TM' and col not in df_kr_processed.columns : # FRFR_POTFR_TM이 아예 없으면 생성
                df_kr_processed[col] = np.nan
    
    # FRFR_POTFR_TM 특별 처리
    if 'FRFR_POTFR_TM' in df_kr_processed.columns:
        # 시간 문자열(예: "1:50")을 분으로 변환하는 함수
        def time_to_minutes(time_str):
            if pd.isna(time_str) or not isinstance(time_str, str) or ':' not in time_str:
                return np.nan # 또는 0
            try:
                parts = time_str.split(':')
                if len(parts) == 2: # HH:MM
                    h, m = map(int, parts)
                    return h * 60 + m
                elif len(parts) == 3: # HH:MM:SS
                    h, m, s = map(int, parts)
                    return h * 60 + m + s / 60
                return np.nan # 또는 0
            except ValueError:
                return np.nan # 또는 0
        
        df_kr_processed['FRFR_POTFR_TM_minutes'] = df_kr_processed['FRFR_POTFR_TM'].apply(time_to_minutes)
        # 기존 FRFR_POTFR_TM 컬럼은 삭제하거나, minutes 컬럼으로 대체하여 사용
        # df_kr_processed.drop('FRFR_POTFR_TM', axis=1, inplace=True)
        # 여기서는 일단 minutes 컬럼을 사용하고, 원래 컬럼은 그대로 두지만 학습에는 사용 안 함
        # 결측치는 나중에 Imputer로
    else:
        df_kr_processed['FRFR_POTFR_TM_minutes'] = np.nan # 컬럼이 없으면 NaN으로 생성


    def map_dnst_cd(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).strip().upper()
        if val_str == 'A': return 1.0
        if val_str == 'B': return 2.0
        if val_str == 'C': return 3.0
        try: return float(val_str)
        except ValueError: return np.nan
    if 'DNST_CD' in df_kr_processed.columns:
        df_kr_processed['density_mapped'] = df_kr_processed['DNST_CD'].apply(map_dnst_cd)
    else:
        df_kr_processed['density_mapped'] = np.nan
        
    def map_frtp_cd(val):
        if pd.isna(val) or str(val).strip() == '': return 'Unknown'
        val_str = str(val).split('.')[0]
        if val_str == '1': return '침엽수림'
        if val_str == '2': return '활엽수림'
        if val_str == '3': return '혼효림'
        if val_str == '4': return '죽림'
        return 'Unknown'
    if 'FRTP_CD' in df_kr_processed.columns:
        df_kr_processed['fuel_type'] = df_kr_processed['FRTP_CD'].apply(map_frtp_cd)
    else:
        df_kr_processed['fuel_type'] = 'Unknown'

    def map_dmcls_cd(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).split('.')[0]
        if val_str == '0': return 0.0
        if val_str == '1': return 1.0
        if val_str == '2': return 2.0
        if val_str == '3': return 3.0
        return np.nan
    if 'DMCLS_CD' in df_kr_processed.columns:
        df_kr_processed['tree_diameter_class'] = df_kr_processed['DMCLS_CD'].apply(map_dmcls_cd)
    else:
        df_kr_processed['tree_diameter_class'] = np.nan

    if 'AGCLS_CD' in df_kr_processed.columns:
        df_kr_processed['age_class'] = pd.to_numeric(df_kr_processed['AGCLS_CD'], errors='coerce')
    else:
        df_kr_processed['age_class'] = np.nan

    if 'STORUNST_CD' in df_kr_processed.columns:
        df_kr_processed['is_forest'] = pd.to_numeric(df_kr_processed['STORUNST_CD'], errors='coerce').fillna(0).astype(int)
    else:
        df_kr_processed['is_forest'] = 0

    if 'FRFR_OCCRR_CAUS_NM' in df_kr_processed.columns:
        df_kr_processed['cause'] = df_kr_processed['FRFR_OCCRR_CAUS_NM'].astype(str).fillna('Unknown')
    else:
        df_kr_processed['cause'] = 'Unknown'

    for col_bool in ['CMPSI_FG', 'PTMNT_FG', 'CMTRY_FG']:
        if col_bool in df_kr_processed.columns:
            df_kr_processed[col_bool] = pd.to_numeric(df_kr_processed[col_bool], errors='coerce').fillna(0).astype(int)
        else:
            df_kr_processed[col_bool] = 0

    if 'OCCRR_MNT' in df_kr_processed.columns:
        df_kr_processed['month'] = pd.to_numeric(df_kr_processed['OCCRR_MNT'], errors='coerce')
    else:
        df_kr_processed['month'] = np.nan

    if 'OCCRR_DYWK_NM' in df_kr_processed.columns:
        df_kr_processed['day_of_week'] = df_kr_processed['OCCRR_DYWK_NM'].astype(str).fillna('Unknown')
    else:
        df_kr_processed['day_of_week'] = 'Unknown'

    df_kr_processed['gis_actual_slope'] = np.random.uniform(0, 45, size=len(df_kr_processed))
    df_kr_processed['gis_fuel_category_detailed'] = np.random.choice(['침엽수밀집', '활엽수밀집', '혼효림', '초지'], size=len(df_kr_processed))
    df_kr_processed['gis_dist_to_fire_station'] = np.random.uniform(100, 15000, size=len(df_kr_processed))

    print(f"한국 데이터 최종 전처리 후 shape: {df_kr_processed.shape}")
    return df_kr_processed

# --- 4. 한국 특화 모델 학습 및 평가 (GBRT 사용, 하이퍼파라미터 튜닝 추가) ---
def train_and_evaluate_korean_gbrt_model(df_korea_final_input,
                                         target_col='POTFR_RSRC_INPT_QNTT',
                                         korean_model_save_path='korean_gbrt_demand_model.joblib',
                                         feature_names_save_path='gbrt_trained_feature_names.txt', # 특성명 저장 경로
                                         perform_grid_search=False):

    if df_korea_final_input is None or df_korea_final_input.empty:
        print("오류: 모델 학습을 위한 한국 데이터가 없습니다.")
        return None, None # 모델과 특성명 리스트 모두 None 반환
    
    df_model_input = df_korea_final_input.copy()

    if target_col not in df_model_input.columns:
        print(f"오류: 목표 변수 '{target_col}'가 한국 데이터에 없습니다.")
        return None, None

    df_model_input['log_target'] = np.log1p(df_model_input[target_col])

    numeric_features = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT',
                        'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                        'FRTP_TRE_HGHT', 'HASLV', 'PRCPT_QNTT', 'FRFR_POTFR_TM_minutes', # 수정된 진화시간 사용
                        'density_mapped', 'tree_diameter_class', 'age_class',
                        'is_forest', 'CMPSI_FG', 'PTMNT_FG', 'CMTRY_FG', 'month',
                        'US_based_predicted_personnel',
                        'gis_actual_slope', 'gis_dist_to_fire_station']

    categorical_features = ['fuel_type', 'cause', 'day_of_week',
                            'gis_fuel_category_detailed']

    existing_numeric_features = [f for f in numeric_features if f in df_model_input.columns and df_model_input[f].nunique() > 1] # 유효한 값이 있는 특성만
    existing_categorical_features = [f for f in categorical_features if f in df_model_input.columns and df_model_input[f].nunique() > 1]
    
    # FRFR_POTFR_TM_minutes 컬럼이 모두 NaN이거나 단일 값일 경우 경고 및 제외 처리 확인
    if 'FRFR_POTFR_TM_minutes' in existing_numeric_features and df_model_input['FRFR_POTFR_TM_minutes'].notna().sum() < 2:
        print("경고: 'FRFR_POTFR_TM_minutes' 컬럼은 유효한 데이터가 부족하여 학습 특성에서 제외합니다.")
        existing_numeric_features.remove('FRFR_POTFR_TM_minutes')


    if not existing_numeric_features and not existing_categorical_features:
        print("오류: 한국 모델 학습에 사용할 수치형 또는 범주형 특성이 하나도 없습니다.")
        return None, None

    X_final_kr = df_model_input[existing_numeric_features + existing_categorical_features]
    y_final_kr_log = df_model_input['log_target']
    y_final_kr_original = df_model_input[target_col]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, existing_numeric_features),
            ('cat', categorical_transformer, existing_categorical_features)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train_log, y_test_log, _, y_test_original = train_test_split(
        X_final_kr, y_final_kr_log, y_final_kr_original, test_size=0.2, random_state=42
    ) # y_test_original도 함께 분할

    print("\n--- 최종 한국 특화 Gradient Boosting Regressor 모델 학습 ---")
    
    gbrt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', GradientBoostingRegressor(random_state=42))])

    if perform_grid_search:
        print("GridSearchCV를 사용한 하이퍼파라미터 튜닝 시작...")
        param_grid = {
            'regressor__n_estimators': [100, 200], # 옵션 줄여서 시간 단축
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [3, 5],
            # ... (다른 파라미터는 필요시 추가)
        }
        grid_search = GridSearchCV(gbrt_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train_log)
        print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
        final_model_pipeline = grid_search.best_estimator_
    else:
        default_gbrt_params = {
            'regressor__n_estimators': 200, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 4,
            'regressor__subsample': 0.7, 'regressor__loss': 'huber', 'regressor__min_samples_split': 10,
            'regressor__min_samples_leaf': 5, 'regressor__random_state': 42
        }
        gbrt_pipeline.set_params(**default_gbrt_params)
        final_model_pipeline = gbrt_pipeline
        final_model_pipeline.fit(X_train, y_train_log)

    pred_log_final_gbrt_kr = final_model_pipeline.predict(X_test)
    pred_original_final_gbrt_kr = np.expm1(pred_log_final_gbrt_kr)
    pred_original_final_gbrt_kr = np.maximum(0, pred_original_final_gbrt_kr) # 음수 예측 방지

    mse_final_gbrt_kr = mean_squared_error(y_test_original, pred_original_final_gbrt_kr)
    r2_final_gbrt_kr = r2_score(y_test_original, pred_original_final_gbrt_kr)
    mae_final_gbrt_kr = mean_absolute_error(y_test_original, pred_original_final_gbrt_kr)
    
    epsilon = 1e-8 
    y_test_original_safe = np.where(y_test_original == 0, epsilon, y_test_original)
    zero_actual_count = np.sum(y_test_original == 0)
    if zero_actual_count > 0:
        print(f"경고: MAPE 계산 시 실제 값이 0인 샘플 {zero_actual_count}개가 발견되었습니다. 계산 시 분모에 epsilon({epsilon}) 적용.")
    mape_final_gbrt_kr = np.mean(np.abs((y_test_original - pred_original_final_gbrt_kr) / y_test_original_safe)) * 100

    print(f"최종 GBRT (한국 특화) - MSE: {mse_final_gbrt_kr:.2f}, RMSE: {np.sqrt(mse_final_gbrt_kr):.2f}, "
          f"R2: {r2_final_gbrt_kr:.4f}, MAE: {mae_final_gbrt_kr:.2f}, MAPE (주의 필요): {mape_final_gbrt_kr:.2f}%")
    print(f"주요 성능 지표: R2 Score = {r2_final_gbrt_kr:.4f}, MAE = {mae_final_gbrt_kr:.2f}, RMSE = {np.sqrt(mse_final_gbrt_kr):.2f}")

    joblib.dump(final_model_pipeline, korean_model_save_path)
    print(f"학습된 최종 한국 GBRT 모델 파이프라인이 '{korean_model_save_path}'에 저장되었습니다.")

    trained_feature_names = None
    try:
        trained_feature_names = final_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        with open(feature_names_save_path, 'w', encoding='utf-8') as f: # 인코딩 추가
            for feature_name in trained_feature_names:
                f.write(f"{feature_name}\n")
        print(f"학습된 GBRT 모델의 최종 입력 특성명 리스트가 '{feature_names_save_path}'에 저장되었습니다.")
        
        # 특성 중요도 시각화
        importances = final_model_pipeline.named_steps['regressor'].feature_importances_
        if len(trained_feature_names) == len(importances):
            forest_importances = pd.Series(importances, index=trained_feature_names)
            forest_importances = forest_importances.sort_values(ascending=False)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=forest_importances.head(20).values, y=forest_importances.head(20).index)
            plt.title("GBRT 모델 특성 중요도 (상위 20개)")
            plt.tight_layout()
            plt.show()
        else:
             print(f"경고: 특성 이름 개수({len(trained_feature_names)})와 중요도 개수({len(importances)}) 불일치로 시각화 생략.")
    except Exception as e:
        print(f"최종 입력 특성명 저장 또는 특성 중요도 시각화 중 오류: {e}")
        
    return final_model_pipeline, trained_feature_names # 특성명 리스트도 반환

# --- 메인 실행 부분 ---
if __name__ == '__main__':
    us_data_file = './datasets/ics209plus-wildfire/ics209plus-wildfire/ics209-plus-wf_sitreps_1999to2020.csv'
    us_model_save_name = 'us_acres_personnel_linear_model.joblib'
    korea_data_file = './datasets/WSQ000301.csv'
    korean_model_save_name = 'korean_gbrt_demand_model.joblib'
    gbrt_feature_names_file = 'gbrt_trained_feature_names.txt'

    model_us_acres_personnel = train_us_acres_personnel_model(us_data_file, model_save_path=us_model_save_name)

    if model_us_acres_personnel:
        df_korea_final_processed = load_and_preprocess_korean_data_final(korea_data_file, target_capping_quantile=0.995)
        if df_korea_final_processed is not None:
            df_korea_with_us_pred = add_us_based_personnel_prediction(df_korea_final_processed, model_us_acres_personnel)
            
            print("\n--- 한국 데이터에 미국 기준 예상 인력 추가 결과 (상위 5개) ---")
            cols_to_show_kr_main = ['FRFR_DMG_AREA', 'POTFR_RSRC_INPT_QNTT', 'US_based_predicted_personnel']
            existing_cols_to_show_kr_main = [c for c in cols_to_show_kr_main if c in df_korea_with_us_pred.columns]
            if existing_cols_to_show_kr_main and not df_korea_with_us_pred.empty :
                print(df_korea_with_us_pred[existing_cols_to_show_kr_main].head())

            trained_korean_model, _ = train_and_evaluate_korean_gbrt_model( # _ 로 특성명 받음
                df_korea_with_us_pred, 
                korean_model_save_path=korean_model_save_name,
                feature_names_save_path=gbrt_feature_names_file,
                perform_grid_search=False
            )
            if trained_korean_model:
                print(f"\n--- 한국 특화 모델 학습 및 저장 완료: {korean_model_save_name} ---")
            else:
                print("한국 특화 모델 학습에 실패했습니다.")
        else:
            print("한국 데이터 최종 전처리 실패.")
    else:
        print("미국 모델 학습 또는 로드 실패.")

