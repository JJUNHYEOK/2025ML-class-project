# Map.py
import folium
import webbrowser
import os
from folium import PolyLine
from folium.plugins import PolyLineTextPath


class WildfireMap:
    def __init__(self, center_lat: float, center_lon: float, zoom: int = 13):
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    def add_fire_locations(self, fire_dict: dict, color='red'):
        """
        화재 위치 표시 (위험도 인덱스에 따라 원 크기 다르게)
        fire_dict: {index: (lat, lon)}
        """
        radius_by_index = {
            0: 300,
            1: 150,
            2: 75
        }
        for index in sorted(fire_dict):
            lat, lon = fire_dict[index]
            radius = radius_by_index.get(index, 50)
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                popup=f"화재 위험도 순위: {index}"
            ).add_to(self.map)

    def add_fire_stations(self, station_coords: list, radius=100, color='blue'):
        """
        소방서 위치 표시
        station_coords: [(lat, lon), ...]
        """
        for lat, lon in station_coords:
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.3,
                popup="소방서 위치"
            ).add_to(self.map)

    def add_response_arrows(self, station_coords: list, fire_dict: dict, color='black'):
        """
        각 소방서에서 화재 지점으로 향하는 직선 + 화살표 1개 표시
        """
        for station_lat, station_lon in station_coords:
            for index in sorted(fire_dict):
                fire_lat, fire_lon = fire_dict[index]

                # 직선 경로
                line = PolyLine(
                    locations=[(station_lat, station_lon), (fire_lat, fire_lon)],
                    color=color,
                    weight=2,
                    opacity=0.8
                )
                self.map.add_child(line)

                # 선 중간에 단일 화살표 기호 표시
                arrow = PolyLineTextPath(
                    line,
                    '→',
                    repeat=False,
                    offset='50%',
                    attributes={
                        'fill': color,
                        'font-weight': 'bold',
                        'font-size': '18'
                    }
                )
                self.map.add_child(arrow)

    def show_map(self, filename='temp_wildfire_map.html'):
        """
        지도를 HTML로 저장하고 브라우저로 자동 열기
        """
        self.map.save(filename)
        webbrowser.open('file://' + os.path.realpath(filename))
