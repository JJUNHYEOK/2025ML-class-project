import pandas as pd
import os
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

def map_slope_fuel_codes(val):
    """ 경사도 및 연료 유형 문자 코드를 수치로 매핑 (예시) """
    if pd.isna(val) or str(val).strip() == '':
        return np.nan # 명시적으로 NaN으로 처리
    val_str = str(val).upper()
    if val_str == 'A': return 5.0 # 예시: A는 높은 값
    if val_str == 'B': return 3.0
    if val_str == 'C': return 1.0 # 예시: C는 낮은 값
    try:
        return float(val_str) # 이미 수치인 경우 float으로
    except ValueError:
        return np.nan # 그 외 변환 불가 시 NaN

def load_and_preprocess_data():
    # ... (파일 로드 부분은 이전과 동일하게 유지) ...
    current_dir = os.getcwd()
    # 제공된 CSV 파일 경로로 직접 지정
    csv_file_path = 'datasets/WSQ000301.csv' # 스크립트와 동일한 위치에 파일이 있다고 가정

    loaded_df = None
    if os.path.exists(csv_file_path):
        try:
            loaded_df = pd.read_csv(csv_file_path, encoding='UTF-8')
            print(f"파일 로드 성공: {os.path.abspath(csv_file_path)}")
        except Exception as e:
            print(f"파일 로드 중 오류 ({csv_file_path}): {e}")
            return None, None
    else:
        print(f"지정된 경로에 파일 없음: {csv_file_path}")
        # Fallback 로직 (필요시 추가)
        return None, None
        
    try:
        df = loaded_df.copy()
        
        feature_columns_map = {
            'WDSP': 'wind_speed',
            'FRTP_CD': 'fuel_type', # 매핑 함수 적용 예정
            'DNST_CD': 'slope',     # 매핑 함수 적용 예정
            'HMDT': 'humidity',
            'DMCLS_CD': 'damage_class' # 범주형으로 처리 예정
        }
        
        original_feature_cols_ordered = ['WDSP', 'FRTP_CD', 'DNST_CD', 'HMDT', 'DMCLS_CD']
        existing_original_cols = [col for col in original_feature_cols_ordered if col in df.columns]
        
        features = df[existing_original_cols].copy()
        new_feature_names_ordered = [feature_columns_map[col] for col in existing_original_cols]
        features.columns = new_feature_names_ordered
        print(f"선택된 특성 (새 이름): {features.columns.tolist()}")

        target = df[['POTFR_RSRC_INPT_QNTT', 'FRFR_DMG_AREA']].copy()
        target.columns = ['required_resources', 'damage_area']

        # --- 특성별 맞춤 전처리 ---
        # 경사도 (slope) 와 연료유형 (fuel_type) 처리: 코드 매핑 및 수치 변환
        if 'slope' in features.columns:
            print(f"원본 slope (DNST_CD) 고유값: {features['slope'].unique()[:20]}") # 샘플 확인
            features['slope'] = features['slope'].apply(map_slope_fuel_codes)
            print(f"매핑 후 slope 고유값 (NaN 제외): {features['slope'].dropna().unique()[:20]}")
        
        if 'fuel_type' in features.columns:
            print(f"원본 fuel_type (FRTP_CD) 고유값: {features['fuel_type'].unique()[:20]}") # 샘플 확인
            features['fuel_type'] = features['fuel_type'].apply(map_slope_fuel_codes)
            # 연료 유형은 범주형으로 다루기 위해 다시 문자열로 변환 후 SimpleImputer(most_frequent) 사용
            # 또는 매핑된 수치를 그대로 사용하고 KNNImputer로 처리할 수도 있음 (여기서는 범주형으로 유지)
            # fuel_type은 이후 범주형 처리에서 SimpleImputer(most_frequent) 후 get_dummies 적용
            # 따라서 여기서는 NaN 처리만 하고, 이후 pd.to_numeric은 하지 않음
            print(f"매핑 후 fuel_type 고유값 (NaN 제외): {features['fuel_type'].dropna().unique()[:20]}")


        # 수치형 특성 이름 정의 (매핑 후 slope, fuel_type도 수치형이 될 수 있음)
        numeric_feature_names = ['wind_speed', 'humidity']
        if 'slope' in features.columns and features['slope'].dtype != 'object': # map_slope_fuel_codes가 수치를 반환했다면
            numeric_feature_names.append('slope')
        # 만약 fuel_type도 수치형으로 다루고 싶다면 여기에 추가
        # else: # fuel_type을 범주형으로 다룰 경우
        #    pass

        categorical_feature_names = ['damage_class'] # DMCLS_CD
        if 'fuel_type' in features.columns: # fuel_type을 범주형으로 다룬다면
             categorical_feature_names.append('fuel_type')


        existing_numeric_final_names = [name for name in numeric_feature_names if name in features.columns]
        existing_categorical_final_names = [name for name in categorical_feature_names if name in features.columns]
        
        print(f"최종 처리할 수치형 특성: {existing_numeric_final_names}")
        print(f"최종 처리할 범주형 특성: {existing_categorical_final_names}")

        numeric_features_df = features[existing_numeric_final_names].copy()
        categorical_features_df = features[existing_categorical_final_names].copy()
        
        # --- 수치형 데이터 처리 ---
        if not numeric_features_df.empty:
            for col in numeric_features_df.columns:
                # 이미 map_slope_fuel_codes에서 float으로 변환 시도했으므로, 여기서는 errors='coerce' 불필요할 수 있음
                # 하지만 안전을 위해 pd.to_numeric 유지
                numeric_features_df[col] = pd.to_numeric(numeric_features_df[col], errors='coerce')
            
            print(f"수치형 특성 Imputation 전 NaN 수:\n{numeric_features_df.isnull().sum()}")
            
            if numeric_features_df.isnull().all().all():
                print("경고: 수치형 특성의 모든 값이 NaN입니다. 0으로 채웁니다.")
                numeric_imputed_data = np.zeros(numeric_features_df.shape)
                numeric_imputed = pd.DataFrame(numeric_imputed_data, columns=numeric_features_df.columns)
            elif numeric_features_df.dropna(how='all').empty: # 모든 행이 다 NaN인 경우 제외하고, 컬럼별로 NaN이 아닌 값이 하나도 없는 경우
                print("경고: 일부 수치형 특성에서 NaN을 제외하면 데이터가 없습니다. SimpleImputer(mean)를 사용합니다.")
                # 컬럼별로 처리
                imputed_cols = []
                for col_name in numeric_features_df.columns:
                    col_data = numeric_features_df[[col_name]]
                    if col_data.dropna().empty:
                        print(f"경고: 컬럼 '{col_name}'은 모든 값이 NaN이거나 유효한 값이 없어 평균 imputation이 불가능합니다. 0으로 채웁니다.")
                        imputed_cols.append(pd.DataFrame(np.zeros(len(col_data)), columns=[col_name]))
                    else:
                        simple_num_imputer = SimpleImputer(strategy='mean')
                        imputed_cols.append(pd.DataFrame(simple_num_imputer.fit_transform(col_data), columns=[col_name]))
                numeric_imputed = pd.concat(imputed_cols, axis=1)
            else:
                # KNNImputer는 NaN이 아닌 값이 2개 이상 있어야 n_neighbors > 1 가능
                n_neighbors = min(5, len(numeric_features_df.dropna(how='all'))) # 모든 값이 NaN인 행 제외하고 계산
                if n_neighbors < 1: n_neighbors = 1 # 최소 1

                numeric_imputer = KNNImputer(n_neighbors=n_neighbors)
                numeric_imputed_data = numeric_imputer.fit_transform(numeric_features_df)
                numeric_imputed = pd.DataFrame(numeric_imputed_data, columns=numeric_features_df.columns)
            print(f"수치형 특성 Imputation 후 NaN 수:\n{numeric_imputed.isnull().sum()}")
        else:
            numeric_imputed = pd.DataFrame()


        # --- 범주형 데이터 처리 ---
        if not categorical_features_df.empty:
            for col in categorical_features_df.columns:
                categorical_features_df[col] = categorical_features_df[col].astype(str) # NaN도 문자열 'nan'으로
            
            categorical_imputer = SimpleImputer(strategy='most_frequent') # 최빈값으로 NaN 채우기
            categorical_imputed_data = categorical_imputer.fit_transform(categorical_features_df)
            categorical_imputed = pd.DataFrame(categorical_imputed_data, columns=categorical_features_df.columns)
            
            # prefix 생성 시 컬럼명에 기반하여 생성
            # 예: 'fuel_type' -> 'fuel', 'damage_class' -> 'damage'
            custom_prefixes = {col: col.split('_')[0] if '_' in col else col for col in categorical_imputed.columns}
            categorical_encoded = pd.get_dummies(categorical_imputed, prefix=custom_prefixes, dummy_na=False)
        else:
            categorical_encoded = pd.DataFrame()

        # ... (타겟 데이터 처리 및 최종 병합, 결측치 제거는 이전과 유사하게 진행) ...
        # 타겟 데이터 처리
        target['required_resources'] = pd.to_numeric(target['required_resources'], errors='coerce')
        if not target[['required_resources']].dropna().empty:
            target_imputer_n_neighbors = min(5, len(target[['required_resources']].dropna()))
            if target_imputer_n_neighbors == 0 : target_imputer_n_neighbors = 1 # 데이터가 없을경우
            target_imputer = KNNImputer(n_neighbors=target_imputer_n_neighbors)
            target_imputed_data = target_imputer.fit_transform(target[['required_resources']])
            target_imputed_required_resources = pd.DataFrame(target_imputed_data, columns=['required_resources'])
        else:
            print("경고: 'required_resources' 타겟 변수의 모든 값이 NaN이거나 데이터가 없습니다. 0으로 채웁니다.")
            target_imputed_required_resources = pd.DataFrame({'required_resources': np.zeros(len(target))})

        target['damage_area'] = pd.to_numeric(target['damage_area'], errors='coerce').fillna(0)

        if not numeric_imputed.empty: numeric_imputed.reset_index(drop=True, inplace=True)
        if not categorical_encoded.empty: categorical_encoded.reset_index(drop=True, inplace=True)
        if not target_imputed_required_resources.empty: target_imputed_required_resources.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)

        features_processed = pd.concat([numeric_imputed, categorical_encoded], axis=1)
        target_processed = pd.concat([target_imputed_required_resources, target[['damage_area']]], axis=1)

        features_processed.reset_index(drop=True, inplace=True)
        target_processed.reset_index(drop=True, inplace=True)
        
        combined_df = pd.concat([features_processed, target_processed], axis=1)
        print(f"병합 후 combined_df shape: {combined_df.shape}")
        
        # 주요 특성 및 목표변수에 NaN이 있는 행 제거
        essential_cols_for_dropna = [col for col in ['wind_speed', 'humidity', 'required_resources'] if col in combined_df.columns]
        # slope가 존재하고, 숫자형이며, numeric_imputed에 포함되었다면 추가
        if 'slope' in numeric_imputed.columns:
             essential_cols_for_dropna.append('slope')

        combined_df_cleaned = combined_df.dropna(subset=essential_cols_for_dropna)
        print(f"결측치 제거 후 cleaned_df shape: {combined_df_cleaned.shape}")

        if combined_df_cleaned.empty:
            print("오류: 결측치 제거 후 데이터가 남아있지 않습니다.")
            return None, None

        final_features_cols = [col for col in features_processed.columns if col in combined_df_cleaned.columns]
        final_target_cols = [col for col in target_processed.columns if col in combined_df_cleaned.columns]

        final_features = combined_df_cleaned[final_features_cols].copy() # SettingWithCopyWarning 방지
        final_target = combined_df_cleaned[final_target_cols].copy()

        # 최종적으로 slope가 포함되었는지 확인하고, 없다면 0으로 채우는 옵션
        if 'slope' not in final_features.columns and 'slope' in new_feature_names_ordered : # 원래 slope를 쓰려고 했다면
            print("최종 특성 데이터에 경사도(slope) 컬럼이 누락되어 0으로 채웁니다.")
            final_features.loc[:, 'slope'] = 0 # .loc 사용
        elif 'slope' in final_features.columns:
            print(f"최종 특성 데이터의 경사도(slope) 기술 통계:\n{final_features['slope'].describe()}")
        
        print(f"최종 특성 데이터 shape: {final_features.shape}")
        print(f"최종 타겟 데이터 shape: {final_target.shape}")
        
        return final_features, final_target
        
    except Exception as e:
        print(f"데이터 전처리 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# # 테스트
if __name__ == '__main__':
    features, target = load_and_preprocess_data()
    if features is not None and target is not None:
        print("\n최종 로드된 특성 데이터 샘플:")
        print(features.head())
        print(f"특성 컬럼: {features.columns}")
        print("\n최종 로드된 타겟 데이터 샘플:")
        print(target.head())
        if 'slope' in features.columns:
            print(f"\nSlope 데이터 포함 여부: True, dtype: {features['slope'].dtype}")
            print(features['slope'].describe())
        else:
            print("\nSlope 데이터 포함 여부: False")
    else:
        print("데이터 로드 및 전처리 최종 실패")
