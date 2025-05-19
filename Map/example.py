from Map import WildfireMap

# 중심 위치 (진주시청)
center_lat = 35.180477
center_lon = 128.108823
wildfire_map = WildfireMap(center_lat, center_lon)

#  위험도 기반 화재 발생 위치 (index: (위도, 경도))
fire_locations = {
    0: (35.192631, 128.101987),     # 가장 위험한 지역
    1: (35.164399, 128.084513),    # 중간 위험
    2: (35.179789, 128.165785)     # 덜 위험
}


#  소방서 위치 리스트 (위도, 경도)
fire_stations = [
    (35.18035823746264, 128.11851962302458)
]

# 지도에 시각화 요소 추가
wildfire_map.add_fire_locations(fire_locations)
wildfire_map.add_fire_stations(fire_stations)
wildfire_map.add_response_arrows(fire_stations, fire_locations)

# 지도 실행
wildfire_map.show_map()
