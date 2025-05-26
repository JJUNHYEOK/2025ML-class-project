import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # 미국 모델용
from sklearn.ensemble import GradientBoostingRegressor # 한국 모델용
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # 모델 저장/로드용

# Matplotlib 한글 폰트 설정 (Windows 예시)
try:
    from matplotlib import font_manager, rc
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
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

    cols = ['ACRES', 'TOTAL_PERSONNEL']
    df_us_processed = df_us[cols].copy()
    df_us_processed['ACRES'] = pd.to_numeric(df_us_processed['ACRES'], errors='coerce').fillna(0)
    df_us_processed['TOTAL_PERSONNEL'] = pd.to_numeric(df_us_processed['TOTAL_PERSONNEL'], errors='coerce').fillna(0)
    df_us_processed = df_us_processed[(df_us_processed['ACRES'] >= 0) & (df_us_processed['TOTAL_PERSONNEL'] >= 0)]

    acres_upper_cap = df_us_processed['ACRES'].quantile(0.99)
    personnel_upper_cap = df_us_processed['TOTAL_PERSONNEL'].quantile(0.99)
    
    df_us_processed['ACRES_capped'] = np.clip(df_us_processed['ACRES'], 0, acres_upper_cap)
    df_us_processed['TOTAL_PERSONNEL_capped'] = np.clip(df_us_processed['TOTAL_PERSONNEL'], 0, personnel_upper_cap)
    
    df_filtered = df_us_processed[
        (df_us_processed['ACRES_capped'] > 0) & 
        (df_us_processed['TOTAL_PERSONNEL_capped'] > 0) 
    ].copy()

    if df_filtered.empty:
        print("오류: 미국 데이터 필터링 후 데이터가 없습니다.")
        return None
    
    print(f"미국 데이터 Capping 및 필터링 후 shape: {df_filtered.shape}")

    df_filtered.loc[:, 'log_ACRES'] = np.log1p(df_filtered['ACRES_capped'])
    df_filtered.loc[:, 'log_TOTAL_PERSONNEL'] = np.log1p(df_filtered['TOTAL_PERSONNEL_capped'])

    X_us = df_filtered[['log_ACRES']]
    y_us = df_filtered['log_TOTAL_PERSONNEL']

    model = LinearRegression()
    model.fit(X_us, y_us)

    print("\n--- 미국 데이터 기반 '규모-자원' 선형 회귀 모델 학습 결과 (이상치 Capping 적용) ---")
    print(f"회귀 계수 (log_ACRES에 대한): {model.coef_[0]:.4f}")
    print(f"절편: {model.intercept_:.4f}")

    pred_us_train = model.predict(X_us)
    r2_us_train = r2_score(y_us, pred_us_train)
    print(f"학습 데이터 R-squared: {r2_us_train:.4f}")

    joblib.dump(model, model_save_path) # 미국 모델 저장
    print(f"학습된 미국 '규모-자원' 모델이 '{model_save_path}'에 저장되었습니다.")
    return model

# --- 2. 학습된 미국 모델을 사용하여 한국 데이터에 "미국 기준 예상 인력" 특성 추가 ---
def add_us_based_personnel_prediction(df_kr_processed, us_model, acres_col_kr='FRFR_DMG_AREA'):
    if us_model is None or df_kr_processed is None or acres_col_kr not in df_kr_processed.columns:
        print("오류: 미국 모델 또는 한국 데이터, 또는 한국 데이터의 면적 컬럼이 준비되지 않았습니다.")
        if df_kr_processed is not None:
            df_kr_processed['US_based_predicted_personnel'] = 0
        return df_kr_processed

    df_with_prediction = df_kr_processed.copy()
    hectare_to_acre = 2.47105
    acres_values = pd.to_numeric(df_with_prediction[acres_col_kr], errors='coerce').fillna(0)
    acres_kr = acres_values * hectare_to_acre
    
    log_acres_kr = np.log1p(acres_kr.clip(lower=0))
    X_for_us_model_kr = pd.DataFrame({'log_ACRES': log_acres_kr})
    log_predicted_personnel_us_based = us_model.predict(X_for_us_model_kr)
    predicted_personnel_us_based = np.expm1(log_predicted_personnel_us_based)
    df_with_prediction['US_based_predicted_personnel'] = np.maximum(0, predicted_personnel_us_based).round().astype(int)

    print("한국 데이터에 'US_based_predicted_personnel' 특성 추가 완료.")
    return df_with_prediction

# --- 3. 한국 데이터 로드 및 최종 전처리 (상세 구현 필요!) ---
def load_and_preprocess_korean_data_final(korea_data_file_path, target_capping_quantile=0.995):
    try:
        df_kr_raw = pd.read_csv(korea_data_file_path)
        print(f"한국 원본 데이터 로드 성공: {df_kr_raw.shape}")
    except FileNotFoundError:
        print(f"오류: 한국 데이터 파일을 찾을 수 없습니다 - {korea_data_file_path}")
        return None
    except Exception as e:
        print(f"한국 데이터 로드 중 오류 발생: {e}")
        return None

    df_kr_processed = df_kr_raw.copy()

    target_col = 'POTFR_RSRC_INPT_QNTT'
    df_kr_processed[target_col] = pd.to_numeric(df_kr_processed[target_col], errors='coerce').fillna(0)
    df_kr_processed[target_col] = df_kr_processed[target_col].apply(lambda x: max(x,0) if pd.notnull(x) else 0)
    
    if target_capping_quantile is not None and 0 < target_capping_quantile < 1:
        upper_bound_target = df_kr_processed[target_col].quantile(target_capping_quantile)
        df_kr_processed[target_col] = np.clip(df_kr_processed[target_col], 0, upper_bound_target)
        print(f"목표 변수 '{target_col}'에 상위 {100*(1-target_capping_quantile):.1f}% Capping 적용 (상한값: {upper_bound_target:.2f})")

    numeric_cols = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT', 'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC']
    for col in numeric_cols:
        if col in df_kr_processed.columns:
            df_kr_processed[col] = pd.to_numeric(df_kr_processed[col], errors='coerce')
            df_kr_processed[col] = df_kr_processed[col].apply(lambda x: max(x,0) if pd.notnull(x) and col != 'TPRT' else (x if pd.notnull(x) else np.nan) )
        else:
            print(f"경고: 한국 데이터에 수치형 특성 '{col}'이 없습니다.")

    def map_dnst_cd(val):
        if pd.isna(val) or str(val).strip() == '': return np.nan
        val_str = str(val).upper();
        if val_str == 'A': return 5.0
        if val_str == 'B': return 3.0
        if val_str == 'C': return 1.0
        try: return float(val_str)
        except ValueError: return np.nan
    if 'DNST_CD' in df_kr_processed.columns:
        df_kr_processed['slope_mapped'] = df_kr_processed['DNST_CD'].apply(map_dnst_cd)
    else:
        df_kr_processed['slope_mapped'] = np.nan

    categorical_cols_map = {
        'FRTP_CD': 'fuel_type_code', 'DMCLS_CD': 'damage_class_code',
        'FRFR_OCCRR_CAUS_NM': 'cause_code', 'STORUNST_CD': 'terrain_code',
        'AGCLS_CD': 'age_class_code', 'CMPSI_FG': 'compass_direction'
    }
    for original_col, new_col_name in categorical_cols_map.items():
        if original_col in df_kr_processed.columns:
            df_kr_processed[new_col_name] = df_kr_processed[original_col].astype(str).fillna('Unknown')
        else:
            df_kr_processed[new_col_name] = 'Unknown'
            print(f"경고: 한국 데이터에 범주형 특성 '{original_col}'이 없어 '{new_col_name}' 특성을 'Unknown'으로 생성합니다.")

    if 'FRFR_OCCRR_LCTN_XCRD' in df_kr_processed.columns: # GIS 특성 생성 (임시)
        df_kr_processed['gis_actual_slope'] = np.random.uniform(0, 45, size=len(df_kr_processed))
        df_kr_processed['gis_fuel_category_detailed'] = np.random.choice(['침엽수밀집', '활엽수밀집', '혼효림', '초지'], size=len(df_kr_processed))
        df_kr_processed['gis_dist_to_fire_station'] = np.random.uniform(100, 15000, size=len(df_kr_processed))
    else:
        print("경고: 좌표 정보가 없어 임의의 GIS 특성을 생성할 수 없습니다. (실제 GIS 특성 추가 필요)")

    print(f"한국 데이터 최종 전처리 후 shape (상세): {df_kr_processed.shape}")
    return df_kr_processed

# --- 4. 한국 특화 모델 학습 및 평가 (Gradient Boosting Regressor 사용) ---
def train_and_evaluate_korean_gbrt_model(df_korea_final_input, korean_model_save_path='korean_gbrt_demand_model.joblib'):
    df_model_input = df_korea_final_input.copy()

    if 'POTFR_RSRC_INPT_QNTT' not in df_model_input.columns:
        print("오류: 목표 변수 'POTFR_RSRC_INPT_QNTT'가 한국 데이터에 없습니다.")
        return None # 모델 반환 대신 None 반환

    df_model_input['log_POTFR_RSRC_INPT_QNTT'] = np.log1p(df_model_input['POTFR_RSRC_INPT_QNTT'])

    numeric_features_final_kr = ['FRFR_DMG_AREA', 'WDSP', 'HMDT', 'TPRT',
                                 'slope_mapped', 'US_based_predicted_personnel',
                                 'FRSTTN_DSTNC', 'PTMNT_DSTNC', 'NNFRS_DSTNC',
                                 'gis_actual_slope', 'gis_dist_to_fire_station']
    
    categorical_features_final_kr = ['fuel_type_code', 'damage_class_code', 'cause_code',
                                     'terrain_code', 'age_class_code', 'compass_direction',
                                     'gis_fuel_category_detailed']

    existing_numeric_features = [f for f in numeric_features_final_kr if f in df_model_input.columns]
    existing_categorical_features = [f for f in categorical_features_final_kr if f in df_model_input.columns]

    if not existing_numeric_features and not existing_categorical_features:
        print("오류: 한국 모델 학습에 사용할 수치형 또는 범주형 특성이 하나도 없습니다.")
        return None

    X_final_kr = df_model_input[existing_numeric_features + existing_categorical_features]
    y_final_kr_log = df_model_input['log_POTFR_RSRC_INPT_QNTT']
    y_final_kr_original = df_model_input['POTFR_RSRC_INPT_QNTT']

    numeric_transformer_final = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer_final = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers_list = []
    if existing_numeric_features:
        transformers_list.append(('num', numeric_transformer_final, existing_numeric_features))
    if existing_categorical_features:
        transformers_list.append(('cat', categorical_transformer_final, existing_categorical_features))
    
    if not transformers_list:
        print("오류: ColumnTransformer에 적용할 특성이 없습니다.")
        return None

    preprocessor_final_kr = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'
    )

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X_final_kr, y_final_kr_log, test_size=0.2, random_state=42
    )
    _, _, _, y_test_original = train_test_split(
        X_final_kr, y_final_kr_original, test_size=0.2, random_state=42
    )

    print("\n--- 최종 한국 특화 Gradient Boosting Regressor 모델 학습 ---")
    gbrt_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4,
        'subsample': 0.7, 'loss': 'huber', 'min_samples_split': 10,
        'min_samples_leaf': 5, 'random_state': 42
    }
    gbrt_model = GradientBoostingRegressor(**gbrt_params)
    
    final_gbrt_pipeline_kr = Pipeline(steps=[('preprocessor', preprocessor_final_kr),
                                           ('regressor', gbrt_model)])
    
    final_gbrt_pipeline_kr.fit(X_train, y_train_log)

    pred_log_final_gbrt_kr = final_gbrt_pipeline_kr.predict(X_test)
    pred_original_final_gbrt_kr = np.expm1(pred_log_final_gbrt_kr)
    pred_original_final_gbrt_kr[pred_original_final_gbrt_kr < 0] = 0

    mse_final_gbrt_kr = mean_squared_error(y_test_original, pred_original_final_gbrt_kr)
    r2_final_gbrt_kr = r2_score(y_test_original, pred_original_final_gbrt_kr)
    mae_final_gbrt_kr = mean_absolute_error(y_test_original, pred_original_final_gbrt_kr)

    print(f"최종 GBRT (한국 특화) - MSE: {mse_final_gbrt_kr:.4f}, R2: {r2_final_gbrt_kr:.4f}, MAE: {mae_final_gbrt_kr:.4f}")
    
    # --- 학습된 한국 모델 저장 ---
    joblib.dump(final_gbrt_pipeline_kr, korean_model_save_path)
    print(f"학습된 최종 한국 GBRT 모델 파이프라인이 '{korean_model_save_path}'에 저장되었습니다.")
    # --------------------------

    # 특성 중요도 확인
    try:
        regressor_step_gbrt = final_gbrt_pipeline_kr.named_steps['regressor']
        preprocessor_step_gbrt = final_gbrt_pipeline_kr.named_steps['preprocessor']
        processed_feature_names_gbrt = []
        if existing_numeric_features:
            processed_feature_names_gbrt.extend(existing_numeric_features)
        if existing_categorical_features and 'cat' in preprocessor_step_gbrt.named_transformers_:
            cat_transformer_gbrt = preprocessor_step_gbrt.named_transformers_['cat']
            if hasattr(cat_transformer_gbrt.named_steps['onehot'], 'get_feature_names_out'):
                 onehot_cols_final_gbrt = cat_transformer_gbrt.named_steps['onehot'].get_feature_names_out(existing_categorical_features)
                 processed_feature_names_gbrt.extend(list(onehot_cols_final_gbrt))

        importances_final_gbrt = regressor_step_gbrt.feature_importances_

        if len(processed_feature_names_gbrt) > 0 and len(importances_final_gbrt) == len(processed_feature_names_gbrt):
            final_gbrt_importances = pd.Series(importances_final_gbrt, index=processed_feature_names_gbrt)
            final_gbrt_importances = final_gbrt_importances.sort_values(ascending=False)
            print("\n최종 GBRT 특성 중요도 (한국 특화):")
            print(final_gbrt_importances.head(20))
        elif len(importances_final_gbrt) != len(processed_feature_names_gbrt) :
             print(f"경고: GBRT 최종 모델 특성 이름 개수({len(processed_feature_names_gbrt)})와 중요도 개수({len(importances_final_gbrt)}) 불일치.")
             print("추출된 특성 이름 후보:", processed_feature_names_gbrt)
             print("모델에 실제 적용된 특성 수:", regressor_step_gbrt.n_features_in_)
    except Exception as e:
        print(f"최종 GBRT 모델 특성 중요도 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return final_gbrt_pipeline_kr # 학습된 한국 모델 파이프라인 반환


# --- 메인 실행 부분 ---
if __name__ == '__main__':
    # 1. 미국 '규모-자원' 모델 학습 및 저장
    # !!! 사용자 실제 미국 데이터 파일 경로로 변경 필요 !!!
    us_data_file = './datasets/ics209plus-wildfire/ics209plus-wildfire/ics209-plus-wf_sitreps_1999to2020.csv'
    us_model_save_name = 'us_acres_personnel_linear_model.joblib'
    model_us_acres_personnel = train_us_acres_personnel_model(us_data_file, model_save_path=us_model_save_name)

    if model_us_acres_personnel:
        # 2. 한국 데이터 로드 및 상세 전처리 (GIS 특성 생성 포함)
        # !!! 사용자 실제 한국 데이터 파일 경로로 변경 필요 !!!
        korea_data_file = './datasets/WSQ000301.csv'
        df_korea_final_processed = load_and_preprocess_korean_data_final(korea_data_file, target_capping_quantile=0.995) 

        if df_korea_final_processed is not None:
            # 3. 한국 데이터에 "미국 기준 예상 인력" 특성 추가
            df_korea_with_us_pred = add_us_based_personnel_prediction(df_korea_final_processed, model_us_acres_personnel)

            print("\n--- 한국 데이터에 미국 기준 예상 인력 추가 결과 (상위 5개) ---")
            cols_to_show = ['FRFR_DMG_AREA', 'POTFR_RSRC_INPT_QNTT', 'US_based_predicted_personnel']
            existing_cols_to_show = [c for c in cols_to_show if c in df_korea_with_us_pred.columns]
            if existing_cols_to_show and not df_korea_with_us_pred.empty :
                 print(df_korea_with_us_pred[existing_cols_to_show].head())

            # 4. 최종 한국 특화 GBRT 모델 학습, 평가 및 저장
            korean_model_save_name = 'korean_gbrt_demand_model.joblib'
            trained_korean_model = train_and_evaluate_korean_gbrt_model(df_korea_with_us_pred, korean_model_save_path=korean_model_save_name)

            if trained_korean_model:
                print(f"\n--- 한국 특화 모델 학습 및 저장 완료: {korean_model_save_name} ---")
                # 이제 respondFireConfigure.py 에서는 KOREAN_GBRT_MODEL_PATH = 'korean_gbrt_demand_model.joblib' 로 설정하여 사용 가능
            else:
                print("한국 특화 모델 학습에 실패했습니다.")
        else:
            print("한국 데이터 최종 전처리 실패.")
    else:
        print("미국 모델 학습 실패.")

