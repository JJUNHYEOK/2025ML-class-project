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

    def add_resource_allocations(self, base_station: dict, resources: list):
        """
        자원 배치 결과를 지도에 표시
        base_station: {'latitude': float, 'longitude': float}
        resources: [{'type': str, 'location': str, 'quantity': int, 'latitude': float, 'longitude': float}, ...]
        """

        radius_by_index = {
            0: 300,
            1: 150,
            2: 75
        }

        # 기준 소방서 위치 표시
        folium.Circle(
            location=[base_station['latitude'], base_station['longitude']],
            radius=200,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            popup=f"기준 소방서<br>위치: ({base_station['latitude']:.6f}, {base_station['longitude']:.6f})"
        ).add_to(self.map)

        # 자원 배치 위치 표시
        for resource in resources:
            # 자원 타입에 따른 색상 설정
            color = 'red' if resource['resource_type'] == 'truck' else 'green'
            
            # 자원 위치에 원 표시
            popup_content = f"""
                <b>자원 정보</b><br>
                유형: {resource['resource_type']}<br>
                종류: {resource['type']}<br>
                수량: {resource['quantity']}대<br>
                위치: ({resource['latitude']:.6f}, {resource['longitude']:.6f})<br>
                거리: {resource['distance']:.1f}km<br>
                화재 위험도 순위: {resources.index(resource)}
            """
            
            folium.Circle(
                location=[resource['latitude'], resource['longitude']],
                radius = radius_by_index.get(resources.index(resource), 50),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.3,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)

            # 기준 소방서에서 자원 위치까지의 경로 표시
            line = PolyLine(
                locations=[(base_station['latitude'], base_station['longitude']), 
                          (resource['latitude'], resource['longitude'])],
                color="black",
                weight=2,
                opacity=0.8
            )
            self.map.add_child(line)

            # 거리 정보를 포함한 화살표 표시
            distance = resource['distance']
            arrow = PolyLineTextPath(
                line,
                f'→ {distance:.1f}km',
                repeat=False,
                offset='50%',
                attributes={
                    'fill': color,
                    'font-weight': 'bold',
                    'font-size': '14',
                    'stroke': 'white',
                    'stroke-width': '3',
                    'paint-order': 'stroke'
                }
            )
            self.map.add_child(arrow)

    def show_map(self, filename='temp_wildfire_map.html'):
        """
        지도를 HTML로 저장하고 브라우저로 자동 열기
        """
        self.map.save(f"scenario/{filename}")
        webbrowser.open('file://' + os.path.realpath(f"scenario/{filename}"))
