import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # 로그 변환 및 기타 수치 연산을 위해 추가

# Matplotlib 한글 폰트 설정 (Windows 예시, 다른 OS는 폰트 경로 및 이름 수정 필요)
# Colab 등에서는 다른 방법으로 한글 폰트 설정 필요
try:
    from matplotlib import font_manager, rc
    font_path = "c:/Windows/Fonts/malgun.ttf"  # 사용자의 시스템에 설치된 한글 폰트 경로
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
except Exception as e:
    print(f"한글 폰트 설정 중 오류: {e}. 그래프의 한글이 깨질 수 있습니다.")
    pass

# 1. 데이터 로드 함수
def load_ics209plus_data(file_path, low_memory=True):
    """
    ICS-209-PLUS CSV 파일을 로드합니다.
    low_memory=False는 다양한 데이터 타입을 더 잘 추론하지만 메모리를 더 사용할 수 있습니다.
    파일 크기가 매우 크다면 low_memory=True를 고려하거나, 필요한 컬럼만 선택하여 로드합니다.
    """
    try:
        df = pd.read_csv(file_path, low_memory=low_memory)
        print(f"데이터 로드 성공: {file_path}")
        print(f"원본 데이터 shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return None
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None

# 2. 기본 데이터 전처리 및 주요 컬럼 선택 함수
def preprocess_incident_data(df_raw):
    """
    원본 DataFrame에서 주요 관심 컬럼을 선택하고 기본적인 전처리를 수행합니다.
    """
    if df_raw is None:
        return None

    # 이전 대화에서 언급된 컬럼 목록을 기반으로 관심 컬럼 확장
    # 실제 데이터셋의 컬럼명을 정확히 확인하고 필요한 컬럼을 선택해야 합니다.
    # 예시로 몇 가지 주요 컬럼을 선택합니다.
    # 'TOTAL_PERSONNEL'이 없다면, 다른 인력 관련 컬럼이나 계산 방법 모색 필요
    cols_of_interest = [
        'INCIDENT_ID', 'INCIDENT_NAME', 'REPORT_FROM_DATE', 'REPORT_TO_DATE', # sitreps 데이터용
        'DISCOVERY_DATE', 'START_YEAR', # incidents 데이터용
        'ACRES', 'EVENT_FINAL_ACRES', # 피해 면적 (sitreps는 ACRES, incidents는 EVENT_FINAL_ACRES 가능성)
        'TOTAL_PERSONNEL', # 총 투입 인력 (가장 중요한 목표 변수 후보)
        # 추가적인 자원 변수 (실제 컬럼명 확인 필요)
        # 'CREWS_CNT', 'ENGINES_CNT', 'HELICOPTERS_CNT', 'DOZERS_CNT', 'WATER_TENDERS_CNT',
        'STR_DESTROYED', 'STR_THREATENED', # 구조물 피해/위협
        'CAUSE', 'FUEL_MODEL', 'TERRAIN', # 화재 환경 특성
        'POO_LATITUDE', 'POO_LONGITUDE', # 발화점 위치
        'PCT_CONTAINED_COMPLETED' # 진압율
    ]

    # DataFrame에 실제 존재하는 컬럼만 선택
    existing_cols = [col for col in cols_of_interest if col in df_raw.columns]
    print(f"선택된 (존재하는) 관심 컬럼: {existing_cols}")
    df = df_raw[existing_cols].copy()

    # 날짜 컬럼 변환 (오류 발생 시 NaT로 처리)
    date_cols = ['REPORT_FROM_DATE', 'REPORT_TO_DATE', 'DISCOVERY_DATE']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 수치형으로 변환해야 할 자원 및 피해 관련 컬럼
    # 실제 컬럼명을 확인하고, TOTAL_PERSONNEL이 없다면 관련 인력 컬럼들을 합산하는 로직 필요
    numeric_cols_to_fill_zero = [
        'TOTAL_PERSONNEL', 'ACRES', 'EVENT_FINAL_ACRES',
        'STR_DESTROYED', 'STR_THREATENED', 'PCT_CONTAINED_COMPLETED'
        # 'CREWS_CNT', 'ENGINES_CNT', 'HELICOPTERS_CNT' 등도 있다면 추가
    ]
    for col in numeric_cols_to_fill_zero:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # 극단적인 값 처리 (예: 음수 방지)
            if df[col].min() < 0 and col != 'EVENT_FINAL_ACRES': # ACRES, TOTAL_PERSONNEL 등은 음수가 될 수 없음
                 print(f"경고: 컬럼 '{col}'에 음수 값이 있어 0으로 처리합니다.")
                 df[col] = df[col].apply(lambda x: max(x, 0))


    # 사용할 최종 피해 면적 컬럼 결정 (EVENT_FINAL_ACRES가 있다면 우선 사용)
    if 'EVENT_FINAL_ACRES' in df.columns and 'ACRES' in df.columns:
        df['final_acres_used'] = df.apply(lambda row: row['EVENT_FINAL_ACRES'] if row['EVENT_FINAL_ACRES'] > 0 else row['ACRES'], axis=1)
    elif 'EVENT_FINAL_ACRES' in df.columns:
        df['final_acres_used'] = df['EVENT_FINAL_ACRES']
    elif 'ACRES' in df.columns:
        df['final_acres_used'] = df['ACRES']
    else:
        print("오류: 피해 면적 관련 컬럼(ACRES, EVENT_FINAL_ACRES)을 찾을 수 없습니다.")
        return None

    # 분석에 필요한 주요 컬럼만 남기기 (예시)
    # 실제 분석 목적에 따라 컬럼을 더 추가하거나 제외할 수 있음
    final_selected_cols = ['INCIDENT_ID', 'INCIDENT_NAME', 'final_acres_used', 'TOTAL_PERSONNEL',
                           'CAUSE', 'FUEL_MODEL', 'TERRAIN', 'STR_THREATENED', 'PCT_CONTAINED_COMPLETED']
    
    existing_final_cols = [col for col in final_selected_cols if col in df.columns]
    if 'TOTAL_PERSONNEL' not in existing_final_cols or 'final_acres_used' not in existing_final_cols:
        print("오류: 분석에 필요한 핵심 컬럼(TOTAL_PERSONNEL 또는 final_acres_used)이 누락되었습니다.")
        return None

    df_processed = df[existing_final_cols].copy()
    print(f"전처리 후 데이터 shape: {df_processed.shape}")
    return df_processed

# 3. 화재 규모(피해 면적)와 투입 인력 간 관계 시각화 함수
def plot_acres_vs_personnel(df_processed, acres_col='final_acres_used', personnel_col='TOTAL_PERSONNEL'):
    """
    피해 면적과 투입 인력 간의 산점도를 그립니다.
    데이터 포인트가 너무 많으면 샘플링하거나 alpha 값을 조절합니다.
    """
    if df_processed is None or acres_col not in df_processed.columns or personnel_col not in df_processed.columns:
        print("오류: 시각화를 위한 데이터 또는 컬럼이 준비되지 않았습니다.")
        return

    plt.figure(figsize=(12, 7))
    # 데이터가 매우 클 경우, 일부만 샘플링하여 그릴 수 있음
    sample_df = df_processed.sample(n=min(50000, len(df_processed)), random_state=42) if len(df_processed) > 50000 else df_processed
    
    sns.scatterplot(data=sample_df, x=acres_col, y=personnel_col, alpha=0.3, s=15) # s는 점 크기
    plt.title('화재 규모(피해 면적)와 투입 인력 관계', fontsize=15)
    plt.xlabel('최종 피해 면적 (에이커)', fontsize=12)
    plt.ylabel('총 투입 인력 수', fontsize=12)
    plt.xscale('log') # 데이터 분포가 넓으므로 로그 스케일 사용
    plt.yscale('log')
    plt.grid(True, which="both", ls="-.", alpha=0.5)
    plt.show()

    # 상관관계 계산 (로그 변환 후)
    # 0이나 음수를 피하기 위해 작은 값(1)을 더한 후 로그 변환
    log_acres = np.log1p(df_processed[acres_col])
    log_personnel = np.log1p(df_processed[personnel_col])
    correlation = log_acres.corr(log_personnel)
    print(f"\n로그 변환된 피해 면적과 투입 인력 간의 피어슨 상관 계수: {correlation:.4f}")

# 4. 피해 면적 구간별 평균 투입 인력 계산 및 시각화 함수
def plot_avg_personnel_by_acres_bins(df_processed, acres_col='final_acres_used', personnel_col='TOTAL_PERSONNEL'):
    """
    피해 면적을 여러 구간으로 나누어 각 구간별 평균 투입 인력을 계산하고 막대그래프로 시각화합니다.
    """
    if df_processed is None or acres_col not in df_processed.columns or personnel_col not in df_processed.columns:
        print("오류: 구간별 분석을 위한 데이터 또는 컬럼이 준비되지 않았습니다.")
        return

    df_temp = df_processed.copy()
    # 매우 큰 ACRES 값을 가진 이상치로 인해 구간 설정이 어려울 수 있으므로, 상위 1% 제외 고려
    # acres_cap = df_temp[acres_col].quantile(0.99)
    # df_temp = df_temp[df_temp[acres_col] <= acres_cap]

    max_acres = df_temp[acres_col].max()
    if pd.isna(max_acres) or max_acres == 0 : # max_acres가 NaN이거나 0이면 구간 설정 불가
        print("경고: 피해 면적 데이터가 유효하지 않아 구간 분석을 생략합니다.")
        return

    bins = [0, 10, 100, 1000, 10000, 100000, max_acres + 1]
    labels = ['0-10', '11-100', '101-1k', '1k-10k', '10k-100k', '100k+']
    
    # max_acres에 따라 labels 마지막 수정
    if max_acres < 100000 :
        bins = [b for b in bins if b <= max_acres +1]
        if max_acres >=10000: labels = labels[:len(bins)-1]
        elif max_acres >=1000: labels = labels[:len(bins)-1]
        elif max_acres >=100: labels = labels[:len(bins)-1]
        elif max_acres >=10: labels = labels[:len(bins)-1]
        else: labels = labels[:len(bins)-1]


    df_temp['acres_bin'] = pd.cut(df_temp[acres_col], bins=bins, labels=labels, right=True, include_lowest=True)
    
    avg_personnel_by_bin = df_temp.groupby('acres_bin', observed=True)[personnel_col].mean().reset_index()
    # observed=True 추가 (Pandas 1.3.0 이상)

    print("\n피해 면적 구간별 평균 투입 인력:")
    print(avg_personnel_by_bin)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_personnel_by_bin, x='acres_bin', y=personnel_col, palette="viridis")
    plt.title('피해 면적 구간별 평균 투입 인력', fontsize=15)
    plt.xlabel('피해 면적 구간 (에이커)', fontsize=12)
    plt.ylabel('평균 투입 인력 수', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# 5. 연료 모델별 평균 투입 인력 분석 (예시)
def plot_avg_personnel_by_fuel_model(df_processed, personnel_col='TOTAL_PERSONNEL', fuel_col='FUEL_MODEL'):
    if df_processed is None or personnel_col not in df_processed.columns or fuel_col not in df_processed.columns:
        print("오류: 연료 모델별 분석을 위한 데이터 또는 컬럼이 준비되지 않았습니다.")
        return
    
    df_temp = df_processed.copy()
    # FUEL_MODEL 결측치 처리 (여기서는 'Unknown'으로 대체)
    df_temp[fuel_col] = df_temp[fuel_col].fillna('Unknown')
    # 너무 많은 고유값을 가진 경우, 상위 N개만 표시
    top_n_fuels = df_temp[fuel_col].value_counts().nlargest(15).index
    df_fuel_subset = df_temp[df_temp[fuel_col].isin(top_n_fuels)]

    avg_personnel_by_fuel = df_fuel_subset.groupby(fuel_col, observed=True)[personnel_col].mean().sort_values(ascending=False).reset_index()

    print(f"\n{fuel_col}별 평균 투입 인력 (상위 15개):")
    print(avg_personnel_by_fuel)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=avg_personnel_by_fuel, x=fuel_col, y=personnel_col, palette="crest")
    plt.title(f'{fuel_col}별 평균 투입 인력 (상위 15개)', fontsize=15)
    plt.xlabel(fuel_col, fontsize=12)
    plt.ylabel('평균 투입 인력 수', fontsize=12)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.show()


# --- 메인 실행 부분 ---
if __name__ == '__main__':
    # 사용자는 이 파일 경로를 실제 다운로드한 ICS-209-PLUS 데이터 파일 경로로 변경해야 합니다.
    # 예시: 'ics209plus_allWFincidents_1999to2014.csv' (사건 요약 테이블)
    # 또는 'ics209plus_allWFsitreps_1999to2020.csv' (일일 상황 보고서 테이블 - 더 클 수 있음)
    # 여기서는 incidents 테이블을 사용하는 것을 가정합니다. sitreps는 INCIDENT_ID별로 여러 행이 있을 수 있습니다.
    # Figshare 등에서 다운로드한 파일명을 확인하세요.
    # 예시 파일명: 'ics209-plus-wf_incidents_1999to2014.csv' (Result 1, 7의 Figshare 링크에서 얻을 수 있는 파일)
    file_path = 'datasets\ics209plus-wildfire\ics209plus-wildfire\ics209-plus-wf_sitreps_1999to2020.csv' # <--- 사용자가 실제 파일 경로로 변경!!!

    raw_df = load_ics209plus_data(file_path)

    if raw_df is not None:
        # 데이터가 매우 크므로, 초기 탐색 시에는 샘플링하여 사용하거나,
        # 전처리 단계에서 필요한 최소한의 컬럼만 선택하는 것이 좋습니다.
        # 예: raw_df_sample = raw_df.sample(n=100000, random_state=42) if len(raw_df) > 100000 else raw_df
        
        processed_df = preprocess_incident_data(raw_df)

        if processed_df is not None:
            print("\n전처리된 데이터 샘플 (상위 5개):")
            print(processed_df.head())

            print("\n전처리된 데이터 기술 통계:")
            # 수치형 컬럼에 대해서만 describe() 호출
            numeric_cols_for_describe = processed_df.select_dtypes(include=np.number).columns
            if not numeric_cols_for_describe.empty:
                print(processed_df[numeric_cols_for_describe].describe())
            else:
                print("기술 통계를 표시할 수치형 컬럼이 없습니다.")


            # 1. 화재 규모(피해 면적) vs. 투입 인력 관계 시각화
            plot_acres_vs_personnel(processed_df)

            # 2. 피해 면적 구간별 평균 투입 인력 시각화
            plot_avg_personnel_by_acres_bins(processed_df)
            
            # 3. 연료 모델별 평균 투입 인력 분석 (예시)
            # FUEL_MODEL 컬럼이 실제로 데이터에 어떤 값을 가지는지 확인 후 실행
            if 'FUEL_MODEL' in processed_df.columns:
                 plot_avg_personnel_by_fuel_model(processed_df)
            else:
                print("\nFUEL_MODEL 컬럼이 없어 관련 분석을 생략합니다.")

            # 추가 분석 아이디어:
            # - TERRAIN (지형) 별 평균 투입 인력
            # - CAUSE (원인) 별 평균 투입 인력
            # - STR_THREATENED (위협 구조물 수) 와 투입 인력 간 관계
            # - PCT_CONTAINED_COMPLETED (진압율) 변화에 따른 투입 인력 변화 (sitreps 데이터 활용 시)
            # - POO_STATE 별 화재 규모 및 투입 인력 비교

        else:
            print("데이터 전처리 중 오류가 발생했거나 필요한 컬럼이 없어 분석을 진행할 수 없습니다.")
    else:
        print("데이터 로드에 실패하여 분석을 진행할 수 없습니다.")

