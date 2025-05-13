import pandas as pd
import pulp
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
from shapely.geometry import Polygon, Point
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, sqrt, atan2

# 데이터 로드
df = pd.read_csv('datasets/WSQ000301.csv')

# 진주시 데이터 필터링
df_jinju = df[
    (df['FRFR_OCCRR_CTPRV_NM'] == '경상남도') &
    (df['FRFR_OCCRR_LCTN_YCRD'].between(35.0, 35.3)) &
    (df['FRFR_OCCRR_LCTN_XCRD'].between(127.9, 128.3))
].copy()
print(f"진주시 산불 사건 수: {len(df_jinju)}")

# 시간 정보 파싱
# OCCRR_YR, OCCRR_MNT, OCCRR_DT, OCCRR_TM을 문자열로 변환
df_jinju['OCCRR_YR'] = df_jinju['OCCRR_YR'].astype(str)
df_jinju['OCCRR_MNT'] = df_jinju['OCCRR_MNT'].astype(str).str.zfill(2)
df_jinju['OCCRR_DT'] = df_jinju['OCCRR_DT'].astype(str).str.zfill(2)
df_jinju['OCCRR_TM'] = df_jinju['OCCRR_TM'].astype(str).str.zfill(2)

# 날짜와 시간 결합
df_jinju['OCCRR_DATE'] = pd.to_datetime(
    df_jinju['OCCRR_YR'] + '-' + df_jinju['OCCRR_MNT'] + '-' + df_jinju['OCCRR_DT'] + ' ' + df_jinju['OCCRR_TM'] + ':00',
    errors='coerce'
)

# 시간대 그룹화 (4시간 단위)
df_jinju['time_group'] = df_jinju['OCCRR_DATE'].dt.floor('4h')
time_groups = df_jinju.groupby('time_group')
print(f"시간대 수: {len(time_groups)}")

# 진압 시간 변환
def convert_time(tm):
    try:
        h, m = map(int, tm.split(':')[:2])
        return h * 60 + m
    except:
        return np.nan
df_jinju['FRFR_POTFR_TM_MIN'] = df_jinju['FRFR_POTFR_TM'].apply(convert_time)

# 심각도 계산
scaler = MinMaxScaler()
df_jinju['severity'] = scaler.fit_transform(
    df_jinju[['FRFR_DMG_AREA', 'FRFR_POTFR_TM_MIN', 'POTFR_RSRC_INPT_QNTT']].fillna(0)
).mean(axis=1)

# 응답 시간
df_jinju['response_time_min'] = df_jinju['FRSTTN_DSTNC'] / 60  # 평균 속도 60 km/h

# 시간대별 사이트 선택
time_group_sites = {}
for tg, group in time_groups:
    top_sites = group.nlargest(3, 'severity')[['FRFR_OCCRR_LCTN_YCRD', 'FRFR_OCCRR_LCTN_XCRD', 'severity', 'FRSTTN_DSTNC']]
    time_group_sites[tg] = {f'site{i+1}_{tg.strftime("%Y%m%d_%H%M")}': 
                            (row['FRFR_OCCRR_LCTN_YCRD'], row['FRFR_OCCRR_LCTN_XCRD'], row['severity'], row['FRSTTN_DSTNC'])
                            for i, row in top_sites.iterrows()}

# Haversine 거리 함수
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# PuLP 모델 파라미터
I = ['FT1', 'FT2']  # 소방차
J = ['FF1', 'FF2', 'FF3']  # 소방관
tp = {'FT1': 5, 'FT2': 6}  # 소방차별 인력
fe = {'FT1': 8, 'FT2': 7}  # 연비
vs = {'FT1': 60, 'FT2': 55}  # 속도 (km/h)
fc = {'FT1': 1500, 'FT2': 1400}  # 연료비
md = {'FF1': 2, 'FF2': 1, 'FF3': 1}  # 소방관 최대 배치
T_max = 3.0  # 최대 시간 3시간
M = 1e5
truck_coords = {'FT1': (35.15, 128.0), 'FT2': (35.2, 128.1)}  # 진주시 소방서

# 시간대별 PuLP 최적화
results = {}
for tg, sites in time_group_sites.items():
    if not sites:
        continue
    N = list(sites.keys())
    d = {n: min(sites[n][2] * 3, 3) for n in N}  # 수요 스케일링
    dist = {(i, n): sites[n][3] / 1000 if sites[n][3] > 0 else haversine(truck_coords[i][1], truck_coords[i][0], sites[n][1], sites[n][0])
            for i in I for n in N}

    prob = pulp.LpProblem(f"Fire_Transport_{tg}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("truck", [(i, n) for i in I for n in N], cat='Binary')
    y = pulp.LpVariable.dicts("ff", [(j, n) for j in J for n in N], cat='Binary')
    slack = pulp.LpVariable.dicts("slack", N, lowBound=0)

    # 목적 함수
    prob += pulp.lpSum((dist[(i, n)] / fe[i]) * fc[i] * x[(i, n)] for i in I for n in N) + \
            pulp.lpSum(slack[n] * 0.01 for n in N)

    # 제약 조건
    for i in I:
        prob += pulp.lpSum(x[(i, n)] for n in N) <= 1, f"Truck_{i}_limit"
    for j in J:
        prob += pulp.lpSum(y[(j, n)] for n in N) <= md[j], f"FF_{j}_limit"
    for n in N:
        prob += pulp.lpSum(tp[i] * x[(i, n)] for i in I) + pulp.lpSum(y[(j, n)] for j in J) + slack[n] >= d[n], f"Demand_{n}"
    for j in J:
        for n in N:
            prob += pulp.lpSum(x[(i, n)] for i in I) >= y[(j, n)], f"Truck_FF_{j}_{n}"
    for n in N:
        prob += pulp.lpSum(y[(j, n)] for j in J) <= pulp.lpSum(tp[i] * x[(i, n)] for i in I), f"FF_limit_{n}"
    for i in I:
        for n in N:
            prob += (dist[(i, n)] / vs[i] - M * x[(i, n)] <= T_max - M), f"Time_{i}_{n}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    assigned = [(i, n) for i in I for n in N if pulp.value(x[(i, n)]) > 0.5]
    results[tg] = {'status': status, 'assigned': assigned, 'sites': sites}
    print(f"시간대 {tg}: 상태={status}, 배치={assigned}")

# Folium 시각화
min_lat, max_lat = 35.0, 35.3
min_lon, max_lon = 127.9, 128.3
cell_size = 0.005
lons = np.arange(min_lon, max_lon, cell_size)
lats = np.arange(min_lat, max_lat, cell_size)
polygons = []
cell_ids = []
idx = 0
for lon in lons:
    for lat in lats:
        poly = Polygon([(lon, lat), (lon + cell_size, lat), (lon + cell_size, lat + cell_size), (lon, lat + cell_size)])
        polygons.append(poly)
        cell_ids.append(idx)
        idx += 1
grid = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': polygons}, crs='EPSG:4326')

map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
m = folium.Map(location=map_center, zoom_start=12)

# 히트맵
heat_data = [[row['FRFR_OCCRR_LCTN_YCRD'], row['FRFR_OCCRR_LCTN_XCRD'], row['severity']]
             for _, row in df_jinju.iterrows()]
HeatMap(heat_data).add_to(m)

# 시간대별 마커 및 그리드
features = []
for tg, res in results.items():
    if res['status'] != 'Optimal':
        continue
    sites = res['sites']
    assigned = set(n for _, n in res['assigned'])
    sites_gdf = gpd.GeoDataFrame(
        {'site': list(sites.keys())},
        geometry=[Point(sites[n][1], sites[n][0]) for n in sites],
        crs='EPSG:4326'
    )
    grid['assigned'] = 0
    joined = gpd.sjoin(grid, sites_gdf, how='left')
    for idx, row in joined.iterrows():
        if pd.notna(row['site']) and row['site'] in assigned:
            grid.at[idx, 'assigned'] = 1

    for n in sites:
        lat, lon, sev, _ = sites[n]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {
                'time': tg.strftime('%Y-%m-%dT%H:%M:%S'),
                'popup': f"사이트 {n}: 심각도={sev:.2f}",
                'style': {'fillColor': 'red' if n in assigned else 'blue'}
            }
        })

TimestampedGeoJson({'type': 'FeatureCollection', 'features': features}).add_to(m)
folium.GeoJson(
    grid.to_json(),
    style_function=lambda feat: {
        'fillColor': 'red' if feat['properties']['assigned'] == 1 else 'green',
        'color': 'gray',
        'weight': 0.5,
        'fillOpacity': 0.4
    }
).add_to(m)

m.save('jinju_fire_map.html')
print("지도가 jinju_fire_time_grouped_map.html로 저장되었습니다")