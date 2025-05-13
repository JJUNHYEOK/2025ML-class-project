import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pulp
from sklearn.preprocessing import StandardScaler
import os

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
                
                # 실제 존재하는 컬럼 선택
                features = df[['WDSP', 'FRTP_CD', 'DNST_CD', 'HMDT', 'DMCLS_CD']]
                target = df[['POTFR_RSRC_INPT_QNTT', 'FRFR_DMG_AREA']]

                # 컬럼 이름 변경
                features.columns = ['wind_speed', 'fuel_type', 'slope', 'humidity', 'damage_class']
                target.columns = ['required_resources', 'damage_area']

                # 결측치 처리 개선
                numeric_imputer = SimpleImputer(strategy='mean')
                features_numeric = features.select_dtypes(include=['float64', 'int64'])
                features_numeric_imputed = pd.DataFrame(
                    numeric_imputer.fit_transform(features_numeric),
                    columns=features_numeric.columns
                )

                categorical_imputer = SimpleImputer(strategy='most_frequent')
                features_categorical = features.select_dtypes(include=['object'])
                features_categorical_imputed = pd.DataFrame(
                    categorical_imputer.fit_transform(features_categorical),
                    columns=features_categorical.columns
                )

                target_imputed = pd.DataFrame(
                    numeric_imputer.fit_transform(target),
                    columns=target.columns
                )

                features_processed = pd.concat([features_numeric_imputed, features_categorical_imputed], axis=1)
                features_processed = features_processed.dropna()
                target_processed = target_imputed.loc[features_processed.index]

                return features_processed, target_processed
                
            except Exception as e:
                print(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
                continue
    
    print("데이터 파일을 찾을 수 없습니다. 다음 경로들을 확인해주세요:")
    for path in possible_paths:
        print(f"- {path}")
    return None, None

def optimize_deployment(features_processed, target_processed):
    # 범주형 컬럼 인코딩
    le = LabelEncoder()
    for col in features_processed.select_dtypes(include=['object']).columns:
        features_processed[col] = le.fit_transform(features_processed[col].astype(str))

    # 데이터 정규화
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_processed)

    # 클러스터링 실행
    cluster_model = DBSCAN(eps=0.5, min_samples=5)
    scenarios = cluster_model.fit_predict(features_scaled)

    # 시나리오 정의
    unique_scenarios = np.unique(scenarios)
    scenario_prob = {s: np.mean(scenarios == s) for s in unique_scenarios}

    # 집합 정의
    I = ['FT1', 'FT2']  # 소방차 유형
    J = ['FF1', 'FF2', 'FF3']  # 소방 대원
    N = ['site1', 'site2', 'site3']  # 배치 위치
    S = unique_scenarios  # 시나리오

    # 파라미터 정의
    cap = {'FT1': 1000, 'FT2': 1200}  # 물탱크 용량 (l)
    tp = {'FT1': 5, 'FT2': 6}  # 수송 가능 인원 (명)
    fe = {'FT1': 8, 'FT2': 7}  # 연비 (km/l)
    vs = {'FT1': 60, 'FT2': 55}  # 평균속도 (km/h)
    fc = {'FT1': 1500, 'FT2': 1400}  # 연료비 (원/l)
    
    # 거리 및 수요 정의
    dist = {('FT1','site1'):10, ('FT1','site2'):20, ('FT1','site3'):15,
            ('FT2','site1'):12, ('FT2','site2'):18, ('FT2','site3'):14}
    d = {'site1': 4, 'site2': 1, 'site3': 2}  # 지점별 수요
    md = {'FF1': 2, 'FF2': 5, 'FF3': 1}  # 대원별 최대 출동 횟수
    T_max = 1.5  # 최대 허용 시간 (h)
    M = 1e5  # Big-M

    # 추가 비용 파라미터
    truck_cost = {'FT1': 5000, 'FT2': 6000}  # 소방차 운영 비용
    ff_cost = {'FF1': 2000, 'FF2': 2000, 'FF3': 2000}  # 대원 운영 비용
    risk_cost = {s: 10000 for s in S}  # 시나리오별 위험 비용

    # 최적화 모델 정의
    model = pulp.LpProblem("Fire_Resource_Allocation", pulp.LpMinimize)

    # 결정변수 정의
    x = pulp.LpVariable.dicts("truck_assign", 
                             [(i,n,s) for i in I for n in N for s in S], 
                             cat='Binary')
    y = pulp.LpVariable.dicts("ff_assign", 
                             [(j,n,s) for j in J for n in N for s in S], 
                             cat='Binary')

    # 목적함수: 총 비용 최소화 (운영비용 + 연료비 + 위험비용)
    model += pulp.lpSum(
        scenario_prob[s] * (
            truck_cost[i] * x[(i,n,s)] +  # 소방차 운영 비용
            ff_cost[j] * y[(j,n,s)] +     # 대원 운영 비용
            (dist[(i,n)]/fe[i]) * fc[i] * x[(i,n,s)] +  # 연료비
            risk_cost[s]                  # 위험 비용
        )
        for i in I for j in J for n in N for s in S
    )

    # 제약조건
    # 소방차 1대당 하나의 지점만 배치
    for i in I:
        for s in S:
            model += pulp.lpSum(x[(i,n,s)] for n in N) <= 1

    # 대원별 최대 출동 횟수
    for j in J:
        for s in S:
            model += pulp.lpSum(y[(j,n,s)] for n in N) <= md[j]

    # 지점별 인원 수요 충족 (소방차와 대원 모두 고려)
    for n in N:
        for s in S:
            model += pulp.lpSum(tp[i] * x[(i,n,s)] for i in I) + \
                    pulp.lpSum(y[(j,n,s)] for j in J) >= d[n]

    # 대원은 배치된 소방차가 있어야 이동 가능
    for j in J:
        for n in N:
            for s in S:
                model += pulp.lpSum(x[(i,n,s)] for i in I) >= y[(j,n,s)]

    # 소방차별 수송인원 상한
    for n in N:
        for s in S:
            model += pulp.lpSum(y[(j,n,s)] for j in J) <= \
                    pulp.lpSum(tp[i]*x[(i,n,s)] for i in I)

    # 시간 제약: 도착 시간 ≤ T_max
    for i in I:
        for n in N:
            for s in S:
                model += (dist[(i,n)]/vs[i]) <= T_max + M*(1 - x[(i,n,s)])

    # 최소 배치 제약 (각 시나리오별로 최소 1대의 소방차 배치)
    for s in S:
        model += pulp.lpSum(x[(i,n,s)] for i in I for n in N) >= 1

    # 최적화 문제 해결
    model.solve()

    # 결과 저장
    deployment_results = []
    for i in I:
        for n in N:
            for s in S:
                if pulp.value(x[(i,n,s)]) > 0.5:
                    deployment_results.append({
                        'scenario': s,
                        'type': i,
                        'location': n,
                        'resource_type': 'truck'
                    })
    
    for j in J:
        for n in N:
            for s in S:
                if pulp.value(y[(j,n,s)]) > 0.5:
                    deployment_results.append({
                        'scenario': s,
                        'type': j,
                        'location': n,
                        'resource_type': 'firefighter'
                    })

    return deployment_results, pulp.value(model.objective), scenarios

def main():
    print("화재 대응 자원 배치 최적화 시스템")
    
    # 데이터 로드 및 전처리
    print("데이터를 로드하고 전처리하는 중...")
    features_processed, target_processed = load_and_preprocess_data()
        
    if features_processed is None or target_processed is None:
        print("데이터 로드 중 오류가 발생했습니다.")
        return
    
    # 최적화 실행
    print("최적화 문제를 해결하는 중...")
    deployment_results, total_cost, scenarios = optimize_deployment(features_processed, target_processed)
    
    # 결과 표시
    print("최적화 결과")
    
    
    # 배치 상세 정보
    print("총 비용: ", total_cost)
    print("배치 수: ", len(deployment_results))
    print("시나리오 커버리지: ", 
          f"{len(set(d['scenario'] for d in deployment_results)) / len(np.unique(scenarios)):.2%}")
    print("배치 상세 정보: ", deployment_results)

if __name__ == "__main__":
    main()
