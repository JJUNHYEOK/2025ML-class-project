import pulp
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import folium

# --------------------------------------------------
# 1. 선형 계획 모형 정의 및 해결 (PuLP)
# --------------------------------------------------
# 집합
I = ['FT1', 'FT2']                     # 소방차 집합
J = ['FF1', 'FF2', 'FF3']              # 소방 대원 집합
N = ['site1', 'site2', 'site3']        # 화재 지점 집합

# 파라미터 (예시값)
tp  = {'FT1': 5,    'FT2': 6}          # 수송 가능 인원
fe  = {'FT1': 8,    'FT2': 7}          # 연비 (km/L)
vs  = {'FT1': 60,   'FT2': 55}         # 속도 (km/h)
fc  = {'FT1': 1500, 'FT2': 1400}       # 연료비 (원/L)
d   = {'site1': 4, 'site2': 3, 'site3': 2}  
dist= {('FT1','site1'):10,('FT1','site2'):20,('FT1','site3'):15,
       ('FT2','site1'):12,('FT2','site2'):18,('FT2','site3'):14}
md  = {'FF1':2,'FF2':1,'FF3':1}        # 최대 출동 횟수
T_max = 1.5                            # 최대 허용 시간(h)
M      = 1e5                          # Big‐M

# 문제 정의
prob = pulp.LpProblem("Fire_Transport", pulp.LpMinimize)

# 결정변수
x = pulp.LpVariable.dicts("truck", 
                          [(i,n) for i in I for n in N],
                          cat='Binary')
y = pulp.LpVariable.dicts("ff", 
                          [(j,n) for j in J for n in N],
                          cat='Binary')

# 목적함수: 연료비 최소화
prob += pulp.lpSum((dist[(i,n)]/fe[i]) * fc[i] * x[(i,n)]
                   for i in I for n in N)

# 제약조건
for i in I:
    prob += pulp.lpSum(x[(i,n)] for n in N) <= 1

for j in J:
    prob += pulp.lpSum(y[(j,n)] for n in N) <= md[j]

for n in N:
    # 소방차 인원+대원 수요 충족
    prob += (pulp.lpSum(tp[i]*x[(i,n)] for i in I)
             + pulp.lpSum(y[(j,n)] for j in J)
             ) >= d[n]

for j in J:
    for n in N:
        # 대원은 차 배치 있어야만
        prob += pulp.lpSum(x[(i,n)] for i in I) >= y[(j,n)]

for n in N:
    # 수송인원 초과 금지
    prob += pulp.lpSum(y[(j,n)] for j in J) <= \
            pulp.lpSum(tp[i]*x[(i,n)] for i in I)

for i in I:
    for n in N:
        # 시간 제약
        prob += (dist[(i,n)]/vs[i]
                 <= T_max + M*(1 - x[(i,n)]))

# 해결
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# 최적 배치 지점 추출
assigned_sites = set()
for i, n in x:
    if pulp.value(x[(i,n)]) > 0.5:
        assigned_sites.add(n)

# --------------------------------------------------
# 2. 그리드 생성 (GeoPandas)
# --------------------------------------------------
# 지도 영역(경도/위도)
min_lat, max_lat = 35.0, 35.3  # 약 30 km
min_lon, max_lon = 127.9, 128.3
cell_size = 0.01  # 약 1km×1km

polygons = []
cell_ids = []
lons = np.arange(min_lon, max_lon, cell_size)
lats = np.arange(min_lat, max_lat, cell_size)
idx = 0
for lon in lons:
    for lat in lats:
        poly = Polygon([
            (lon, lat),
            (lon+cell_size, lat),
            (lon+cell_size, lat+cell_size),
            (lon, lat+cell_size),
        ])
        polygons.append(poly)
        cell_ids.append(idx)
        idx += 1

grid = gpd.GeoDataFrame({'cell_id': cell_ids,
                         'geometry': polygons},
                        crs='EPSG:4326')

# 화재 지점 위/경도 정의 (예시)
site_coords = {
    'site1': (128.09318849813317 , 35.156378294637506), # 진주시 진주대로 501 (가좌캠퍼스 ict융합센터)
    'site2': (128.0934006050697, 35.18042313594802), # 진주시 동진로 33 (칠암캠퍼스)
    'site3': (128.08239486362245, 35.156781657111466), # 진주시 내동면 내동로 139 (내동캠퍼스)
}
# GeoDataFrame으로 변환
sites = gpd.GeoDataFrame(
    {'site': list(site_coords.keys())},
    geometry=[Point(lon, lat) for lon, lat in site_coords.values()],
    crs='EPSG:4326'
)

# --------------------------------------------------
# 3. 배치 결과를 그리드 속성으로 매핑
# --------------------------------------------------
# 각 셀에 'assigned' 속성 추가: 소방차 배치 대상 셀 여부
grid['assigned'] = 0
# 각 지점을 포함하는 셀에 표시
joined = gpd.sjoin(grid, sites, how='left')
for idx, row in joined.iterrows():
    if row['site'] in assigned_sites:
        grid.at[idx, 'assigned'] = 1

# --------------------------------------------------
# 4. Folium으로 지도 시각화
# --------------------------------------------------
m = folium.Map(location=[(min_lat+max_lat)/2,
                         (min_lon+max_lon)/2],
               zoom_start=11)

def style_fn(feat):
    assigned = feat['properties']['assigned']
    return {
        'fillColor': 'red'   if assigned==1 else 'green',
        'color':     'gray',
        'weight':    0.5,
        'fillOpacity': 0.4,
    }

folium.GeoJson(
    grid.to_json(),
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(fields=['cell_id','assigned'])
).add_to(m)

# 화재 지점 마커 추가
for site, (lon, lat) in site_coords.items():
    folium.CircleMarker(
        location=(lat, lon),
        radius=5, color='blue', fill=True,
        fill_opacity=1.0,
        tooltip=site
    ).add_to(m)

# 결과 저장
m.save('code/test/LinearProgramming/fire_transport_map.html')
print("최적화된 배치 지도가 fire_transport_map.html 로 생성되었습니다.")
