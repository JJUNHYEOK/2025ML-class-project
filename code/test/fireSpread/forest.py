import pandas as pd

def parse_gps_key(gps_key):
    """'35_18_38.5_127_51_10.4' 형식의 키를 위도, 경도로 변환"""
    parts = gps_key.split('_')
    lat = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
    lon = float(parts[3]) + float(parts[4])/60 + float(parts[5])/3600
    return lat, lon

# CSV 파일 로드 및 좌표 변환
df_forest = pd.read_csv('code/test/fireSpread/forest_last.csv')
df_forest[['latitude', 'longitude']] = df_forest['gps_key'].apply(
    lambda x: pd.Series(parse_gps_key(x))
)

# 수종별 연료 모델 맵핑 (예시)
# 실제 산림청 데이터를 기반으로 보정하면 정확도가 높아집니다.
fuel_model_map = {
    '소나무': 4,      # 대표적인 침엽수, 확산 위험 높음
    '신갈나무': 2,    # 대표적인 참나무류(활엽수), 확산 위험 보통
    '굴참나무': 2,
    '졸참나무': 2,
    '떡갈나무': 2,
    '상수리나무': 2,
    '층층나무': 1,    # 활엽수, 확산 위험 낮음
    '비목나무': 1,
    # 기타 수종 추가...
    '기타': 1         # 정보가 없는 경우 기본값
}

df_forest['fuel_type'] = df_forest['교목우점_species'].map(fuel_model_map).fillna(1)
