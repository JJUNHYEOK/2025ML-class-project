import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import requests
import folium
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QProgressBar,
    QHeaderView, QLineEdit, QComboBox, QDateEdit, QSpinBox, QFileDialog, QStatusBar
)
from PyQt5.QtCore import Qt, QUrl, QDate
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEngineProfile
import PyQt5
from PyQt5.QtGui import *
from typing import Dict

from code.test.LinearProgramming.respondFireConfigure import main as run_optimization
from code.test.LinearProgramming.respondFireConfigure import load_and_preprocess_data
from code.test.LinearProgramming.respondFireConfigure import generate_scenarios_from_data
from code.test.LinearProgramming.respondFireConfigure import ResourceAllocator

from code.Map.Map import WildfireMap
from code.Risk.RiskCalculator import RiskCalculator
from code.Front.key import key
from code.Front.index_popup import IndexPopup

#주소 찾기 코드 - 비동기
import asyncio
import aiohttp



# 상수 정의
MAP_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'maps')
os.makedirs(MAP_DIR, exist_ok=True)

def find_qtwebengine_process():
    # PyQt5 설치 경로 찾기
    pyqt_path = os.path.dirname(PyQt5.__file__)
    
    # 가능한 경로들
    possible_paths = [
        os.path.join(pyqt_path, 'Qt5', 'bin', 'QtWebEngineProcess.exe'),
        os.path.join(pyqt_path, 'Qt', 'bin', 'QtWebEngineProcess.exe'),
        os.path.join(os.path.dirname(pyqt_path), 'PyQt5', 'Qt5', 'bin', 'QtWebEngineProcess.exe'),
        os.path.join(os.path.dirname(pyqt_path), 'PyQt5', 'Qt', 'bin', 'QtWebEngineProcess.exe'),
    ]
    
    # 현재 실행 파일 위치도 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths.append(os.path.join(current_dir, 'QtWebEngineProcess.exe'))
    
    # 각 경로 확인
    for path in possible_paths:
        if os.path.exists(path):
            print(f"QtWebEngineProcess.exe를 찾았습니다: {path}")
            return path
    
    print("경고: QtWebEngineProcess.exe를 찾을 수 없습니다.")
    print("검색한 경로들:")
    for path in possible_paths:
        print(f"- {path}")
    return None

class Messenger:
    def __init__(self, scenario, parent=None):
        self.scenario = scenario
        self.parent = parent  # 부모 윈도우 저장
        self.popup = None    # 팝업 객체 초기화
        self.show_popup()

    def show_popup(self):
        self.popup = IndexPopup(self.scenario, parent=self.parent)
        self.popup.show()  # 비모달로 표시

class FireGuardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIWRS 대시보드")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('code\Front\icon.png'))
        self.popups = []
        self.initUI()

    def initUI(self):
        # WebEngine 설정 초기화
        QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.status_bar = self.statusBar()
        self.video_tab = VideoTab(status_bar=self.status_bar)

        self.dashboard_tab = DashboardTab()
        self.resource_tab = ResourceManagementTab()
        self.history_tab = HistoryTab()

        # 탭들 서로 연결
        self.history_tab.connect_dashboard(self.dashboard_tab)
        self.resource_tab.connect_dashboard(self.dashboard_tab)

        optimize_button = QPushButton("자원 최적화 실행")
        optimize_button.clicked.connect(self.run_fire_optimization_and_show_map)
        self.tabs.setCornerWidget(optimize_button)


        self.tabs.addTab(self.dashboard_tab, "실시간 상황")
        self.tabs.addTab(self.resource_tab, "자원 관리")
        self.tabs.addTab(self.history_tab, "기록 조회")
        self.tabs.addTab(self.video_tab, "영상 분석")

    def run_fire_optimization_and_show_map(self):

        # 데이터 로드 및 전처리
        features_processed, target_processed = load_and_preprocess_data()
        if features_processed is None or target_processed is None:
            print("데이터 로드 오류")
            return

        # 시나리오 생성
        scenarios = generate_scenarios_from_data(features_processed, target_processed)
        print(f"scenarios type: {type(scenarios)}, length: {len(scenarios)}, content: {scenarios}")
        if not scenarios:
            print("시나리오가 생성되지 않음")
            return
        allocator = ResourceAllocator()

        # 자원 관리 탭의 모든 설정을 반영
        truck_settings, personnel_settings = self.resource_tab.get_all_resource_settings()
        for truck_type, qty in truck_settings.items():
            allocator.set_resource_deployment('truck', truck_type, qty)
        for personnel_type, qty in personnel_settings.items():
            allocator.set_resource_deployment('firefighter', personnel_type, qty)

        # 자원 현황 업데이트
        self.dashboard_tab.update_resource_status()
        
        for scenario in scenarios:
            results, cost = allocator.optimize_single_scenario(scenario)
            if results:
                # Map.py의 WildfireMap으로 지도 생성
                map_obj = WildfireMap(
                    center_lat=scenario.base_station['latitude'],
                    center_lon=scenario.base_station['longitude'],
                    zoom=12
                )
                map_obj.add_resource_allocations(scenario.base_station, results)
                
                # 지도 파일 저장
                map_path = os.path.abspath(os.path.join(MAP_DIR, f'scenario_{scenario.id}_map.html'))
                map_obj.show_map(map_path)
                print(f"지도 파일이 저장되었습니다: {map_path}")
                
                # 생성된 지도를 MapTab에 표시
                self.dashboard_tab.map_widget.load_scenario_map(scenario.id)
                
                # 화재 위협 정보 업데이트
                self.dashboard_tab.fire_count_label.setText(str(len(scenario.sites)))
                
                # 위협 목록 업데이트
                threat_list = []
                for site_id, site_info in scenario.sites.items():
                    risk_factors = site_info['risk_factors']
                    risk_score = self.dashboard_tab.risk_calculator.calculate_risk_score(risk_factors)
                    risk_level = self.dashboard_tab.risk_calculator.get_risk_level(risk_score)
                    threat_list.append(f"위치 {site_id}: {risk_level} ({risk_score}%)")
                
                self.dashboard_tab.threat_list.setText("\n".join(threat_list))
                
                # 위험도 평가 업데이트
                # 모든 사이트의 평균 위험도 계산
                avg_risk_factors = {
                    'wind_speed': sum(site['risk_factors']['wind_speed'] for site in scenario.sites.values()) / len(scenario.sites),
                    'humidity': sum(site['risk_factors']['humidity'] for site in scenario.sites.values()) / len(scenario.sites),
                    'fuel_type': max(site['risk_factors']['fuel_type'] for site in scenario.sites.values()),
                    'slope': sum(site['risk_factors']['slope'] for site in scenario.sites.values()) / len(scenario.sites),
                    'damage_class': max(site['risk_factors']['damage_class'] for site in scenario.sites.values())
                }
                self.dashboard_tab.update_risk_assessment(avg_risk_factors)
                
                # 자원 배치 알림 추가
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                deployment_log = f"[{timestamp}] 🔥 화재 대응 자원 배치 완료\n"
                deployment_log += f"- 배치된 소방차: {sum(1 for r in results if r['resource_type'] == 'truck')}대\n"
                deployment_log += f"- 배치된 도보 인력: {sum(1 for r in results if r['resource_type'] == 'firefighter')}명\n"
                deployment_log += f"- 예상 비용: {cost:,.0f}원\n"
                
                self.dashboard_tab.alert_text.append(deployment_log)
                
                # 기록 탭에 시나리오 정보 추가
                scenario_log = f"[{timestamp}] 시나리오 {scenario.id} 생성\n"
                scenario_log += f"- 화재 발생 지점: {len(scenario.sites)}개\n"
                scenario_log += f"- 평균 위험도: {self.dashboard_tab.risk_calculator.calculate_risk_score(avg_risk_factors)}%\n"
                scenario_log += f"- 배치된 자원: {len(results)}개\n"
                scenario_log += f"- 예상 비용: {cost:,.0f}원\n"
                
                self.dashboard_tab.fire_logs.append(scenario_log)
                self.history_tab.log_view.append(scenario_log)
                
                # 자원 현황 업데이트
                self.dashboard_tab.update_resource_status()
                
                # 자원 탭의 위치 정보 업데이트
                self.resource_tab.update_resource_locations(results)

                index = 0
                print(f"Using index: {index}, type: {type(index)}")  # 인덱스 타입 확인
                try:
                    m = Messenger(avg_risk_factors, parent=self)
                    self.popups.append(m)
                except TypeError as e:
                    print(f"TypeError: {e} - Check if index is float or scenarios is not a list")

                break  # 첫 번째 시나리오만 표시


class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self.fire_logs = []
        self.risk_calculator = RiskCalculator()
        self.initUI()
        self.update_weather_data()
        self.update_resource_status()

    def initUI(self):
        # --- 좌측 상단 네모들 ---
        main_layout = QHBoxLayout()

        fire_threat_box = QGroupBox("현재 화재 위협")
        fire_threat_box.setFixedSize(280, 280)
        fire_threat_layout = QVBoxLayout()
        self.fire_count_label = QLabel("0")
        self.fire_count_label.setAlignment(Qt.AlignCenter)
        self.fire_count_label.setStyleSheet("font-size: 48px; font-weight: bold; color: red;")
        fire_threat_layout.addWidget(self.fire_count_label)

        self.threat_list = QLabel("현재 위협 없음")
        self.threat_list.setAlignment(Qt.AlignCenter)
        fire_threat_layout.addWidget(self.threat_list)
        fire_threat_box.setLayout(fire_threat_layout)

        risk_box = QGroupBox("위험 예측")
        risk_box.setFixedSize(280, 280)
        risk_layout = QVBoxLayout()
        self.risk_level_label = QLabel("위험 수준 0%")
        self.risk_level_label.setAlignment(Qt.AlignCenter)
        self.risk_level_label.setStyleSheet("font-size: 32px; color: darkred; font-weight: bold;")
        risk_layout.addWidget(self.risk_level_label)

        self.risk_bar = QProgressBar()
        self.risk_bar.setValue(0)
        risk_layout.addWidget(self.risk_bar)

        self.risk_factors_label = QLabel("주요 위험 요인:\n- 높은 기온\n- 강한 바람\n- 가뭄 상태\n- 건조한 식생")
        self.risk_factors_label.setAlignment(Qt.AlignCenter)
        risk_layout.addWidget(self.risk_factors_label)
        risk_box.setLayout(risk_layout)

        # 대응 자원 현황 박스, 4등분 그리드로 수정 및 구급차 추가
        resources_box = QGroupBox("대응 자원 현황")
        resources_box.setFixedSize(280, 280)
        resources_layout = QVBoxLayout()

        grid = QGridLayout()
        # 2x2 그리드로 소방차, 항공 지원, 인력, 구급차 배치
        grid.addWidget(QLabel("소방차"), 0, 0, alignment=Qt.AlignCenter)
        self.truck_status_label = QLabel("15/15 대기 중")
        self.truck_status_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.truck_status_label, 1, 0, alignment=Qt.AlignCenter)

        grid.addWidget(QLabel("인력"), 2, 0, alignment=Qt.AlignCenter)
        self.personnel_status_label = QLabel("120/120 대기 중")
        self.personnel_status_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.personnel_status_label, 3, 0, alignment=Qt.AlignCenter)

        resources_layout.addLayout(grid)

        # 전반적인 준비도는 대응 자원 현황 박스 내부 하단에 배치 (고정 크기 유지)
        self.overall_readiness_label = QLabel("전반적인 준비도")
        self.overall_readiness_label.setAlignment(Qt.AlignCenter)
        self.overall_readiness_bar = QProgressBar()
        self.overall_readiness_bar.setValue(100)
        resources_layout.addWidget(self.overall_readiness_label)
        resources_layout.addWidget(self.overall_readiness_bar)

        resources_box.setLayout(resources_layout)

        self.weather_box = QGroupBox("기상 정보 (진주시 가좌동)")
        self.weather_box.setFixedSize(280, 280)
        self.weather_layout = QGridLayout()

        self.weather_labels = {}
        weather_keys = ["풍향", "풍속", "온도", "습도", "강수량", "대기질"]
        for i, key in enumerate(weather_keys):
            key_label = QLabel(key)
            val_label = QLabel("불러오는 중...")
            self.weather_layout.addWidget(key_label, i, 0)
            self.weather_layout.addWidget(val_label, i, 1)
            self.weather_labels[key] = val_label

        self.weather_box.setLayout(self.weather_layout)

        # --- 지도와 실시간 알림 레이아웃 변경 ---
        bottom_layout = QHBoxLayout()

        # 지도 크게 (실시간 상황 탭 내에 포함)
        self.map_widget = MapTab()
        self.map_widget.setFixedSize(700, 480)  # 지도 크기 조절

        # 실시간 알림은 지도 옆에 작게 세로 배치
        alert_box = QGroupBox("실시간 알림")
        alert_box.setFixedSize(280, 480)
        alert_layout = QVBoxLayout()
        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setText("현재 알림이 없습니다.")
        alert_layout.addWidget(self.alert_text)
        alert_box.setLayout(alert_layout)

        bottom_layout.addWidget(self.map_widget)
        bottom_layout.addWidget(alert_box)

        main_layout.addWidget(fire_threat_box)
        main_layout.addWidget(risk_box)
        main_layout.addWidget(resources_box)
        main_layout.addWidget(self.weather_box)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(bottom_layout)  # 지도+알림 배치

        self.setLayout(final_layout)

    def update_risk_assessment(self, risk_factors: Dict):
        """위험도 평가 업데이트"""
        # 위험도 점수 계산
        risk_score = self.risk_calculator.calculate_risk_score(risk_factors)
        risk_level = self.risk_calculator.get_risk_level(risk_score)
        
        # 위험도 표시 업데이트
        self.risk_level_label.setText(f"위험 수준 {risk_score}%")
        self.risk_bar.setValue(int(risk_score))
        
        # 위험도에 따른 색상 변경
        if risk_score >= 80:
            color = "red"
        elif risk_score >= 60:
            color = "orange"
        elif risk_score >= 40:
            color = "yellow"
        elif risk_score >= 20:
            color = "lightgreen"
        else:
            color = "green"
        
        self.risk_level_label.setStyleSheet(f"font-size: 32px; color: {color}; font-weight: bold;")
        
        # 위험 요인 설명 업데이트
        risk_factors_desc = self.risk_calculator.get_risk_factors_description(risk_factors)
        if risk_factors_desc:
            self.risk_factors_label.setText("주요 위험 요인:\n" + "\n".join(f"- {desc}" for desc in risk_factors_desc))
        else:
            self.risk_factors_label.setText("현재 특별한 위험 요인이 없습니다.")
        
        # 위험도가 높은 경우 로그에 기록
        if risk_score >= 60:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] ⚠️ 위험도 {risk_score}% ({risk_level}) - {', '.join(risk_factors_desc)}"
            self.fire_logs.append(log_entry)
            
            # 실시간 알림 업데이트
            self.alert_text.append(log_entry)
            self.alert_text.setStyleSheet("color: red; font-weight: bold;")

    def update_weather_data(self):
        api_key = key
        lat, lon = 35.1767, 128.1035
        url = (f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=kr")

        try:
            response = requests.get(url)
            response.raise_for_status()  # HTTP 오류 확인
            data = response.json()
            
            # API 응답 검증
            if 'main' not in data or 'wind' not in data:
                raise ValueError("필수 날씨 데이터가 누락되었습니다.")

            # 날씨 데이터 추출
            wind_speed = data.get('wind', {}).get('speed', 0)
            wind_deg = data.get('wind', {}).get('deg', 0)
            temp = data.get('main', {}).get('temp', 0)
            humidity = data.get('main', {}).get('humidity', 0)
            rain = data.get('rain', {}).get('1h', 0) if 'rain' in data else 0
            air_quality = "좋음"  # 기본값

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"API 오류 발생: {str(e)}")
            print("더미 데이터를 사용합니다.")
            # 더미 데이터 사용
            wind_speed = 5.2
            wind_deg = 180
            temp = 25.6
            humidity = 65
            rain = 0
            air_quality = "좋음"

        # 풍향 계산
        wind_dir = self.deg_to_direction(wind_deg)

        # UI 업데이트
        self.weather_labels["풍향"].setText(f"{wind_dir}")
        self.weather_labels["풍속"].setText(f"{wind_speed} m/s")
        self.weather_labels["온도"].setText(f"{temp} °C")
        self.weather_labels["습도"].setText(f"{humidity}%")
        self.weather_labels["강수량"].setText(f"{rain} mm")
        self.weather_labels["대기질"].setText(air_quality)

        # 위험도 평가 업데이트
        risk_factors = {
            'wind_speed': wind_speed,
            'humidity': humidity,
            'fuel_type': 3,  # 기본값
            'slope': 15,     # 기본값
            'damage_class': 2  # 기본값
        }
        self.update_risk_assessment(risk_factors)

    def deg_to_direction(self, deg):
        directions = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
        ix = int((deg + 22.5) / 45) % 8
        return directions[ix]

    def update_resource_status(self):
        """자원 현황을 업데이트하는 메서드"""
        from code.test.LinearProgramming.respondFireConfigure import ResourceAllocator
        
        # ResourceAllocator 인스턴스 생성
        allocator = ResourceAllocator()
        
        # 소방차 현황 업데이트
        total_trucks = len(allocator.truck_types) * 2  # 각 타입별 최대 2대
        available_trucks = total_trucks  # 현재는 모든 차량이 대기 중이라고 가정
        self.truck_status_label.setText(f"{available_trucks}/{total_trucks} 대기 중")
        
        # 인력 현황 업데이트
        total_personnel = len(allocator.firefighter_types) * 3  # 각 타입별 최대 3명
        available_personnel = total_personnel  # 현재는 모든 인력이 대기 중이라고 가정
        self.personnel_status_label.setText(f"{available_personnel}/{total_personnel} 대기 중")
        
        # 전반적인 준비도 업데이트
        readiness = int((available_trucks / total_trucks + available_personnel / total_personnel) / 2 * 100)
        self.overall_readiness_bar.setValue(readiness)


class MapTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        m = folium.Map(location=[35.1767, 128.1035], zoom_start=14)
        folium.Marker(location=[35.1767, 128.1035], tooltip="가좌동").add_to(m)
        folium.CircleMarker(location=[35.18, 128.10], radius=30, color='red', fill=True, fill_opacity=0.5, popup="화재 위험 지역").add_to(m)

        map_path = os.path.abspath(os.path.join(MAP_DIR, "map.html"))
        m.save(map_path)
        print(f"기본 지도 파일이 저장되었습니다: {map_path}")

        self.web_view = QWebEngineView()
        self.web_view.load(QUrl.fromLocalFile(map_path))
        layout.addWidget(self.web_view)

        self.setLayout(layout)

    def load_scenario_map(self, scenario_id):
        map_path = os.path.abspath(os.path.join(MAP_DIR, f"scenario_{scenario_id}_map.html"))
        if os.path.exists(map_path):
            self.web_view.load(QUrl.fromLocalFile(map_path))
            print(f"시나리오 지도를 로드했습니다: {map_path}")
        else:
            print(f"지도 파일을 찾을 수 없습니다: {map_path}")


class ResourceManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_current_resources()

    def initUI(self):
        layout = QVBoxLayout()

        # 자원 설정 그룹
        resource_group = QGroupBox("자원 설정")
        resource_layout = QGridLayout()

        # 소방차 설정
        resource_layout.addWidget(QLabel("소방차 타입"), 0, 0)
        self.truck_type_combo = QComboBox()
        self.truck_type_combo.addItems(['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6'])
        resource_layout.addWidget(self.truck_type_combo, 0, 1)

        resource_layout.addWidget(QLabel("배치 수량"), 0, 2)
        self.truck_quantity_spin = QSpinBox()
        self.truck_quantity_spin.setRange(0, 2)
        self.truck_quantity_spin.setValue(2)
        self.truck_quantity_spin.setSpecialValueText("대기중")  # 0일 때 "대기중" 표시
        resource_layout.addWidget(self.truck_quantity_spin, 0, 3)

        # 인력 설정
        resource_layout.addWidget(QLabel("인력 타입"), 1, 0)
        self.personnel_type_combo = QComboBox()
        self.personnel_type_combo.addItems(['FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6'])
        resource_layout.addWidget(self.personnel_type_combo, 1, 1)

        resource_layout.addWidget(QLabel("배치 수량"), 1, 2)
        self.personnel_quantity_spin = QSpinBox()
        self.personnel_quantity_spin.setRange(0, 3)
        self.personnel_quantity_spin.setValue(3)
        self.personnel_quantity_spin.setSpecialValueText("대기중")  # 0일 때 "대기중" 표시
        resource_layout.addWidget(self.personnel_quantity_spin, 1, 3)

        # 설정 버튼
        apply_button = QPushButton("설정 적용")
        apply_button.clicked.connect(self.apply_resource_settings)
        resource_layout.addWidget(apply_button, 2, 0, 1, 4)

        resource_group.setLayout(resource_layout)
        layout.addWidget(resource_group)

        # 현재 자원 현황 테이블
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["자원명", "상태", "위치"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)


    async def get_road_address_from_coords(self, lon, lat):
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 18,
            "accept-language": "ko"
        }
        headers = {"User-Agent": "AIWRS/Beta1.0 (moongijun967@gmail.com)"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        address = data.get('address', {})
                        road = address.get('road', '')
                        suburb = address.get('suburb', '')
                        city = address.get('city', '')
                        country = address.get('country', '')
                        return f"{country} {city} {suburb} {road}" if road else "주소 없음"
                    else:
                        return f"API 오류: {response.status}"
        except Exception as e:
            return f"오류: {str(e)}"



    def load_current_resources(self):
        """현재 자원 설정을 로드"""
        from code.test.LinearProgramming.respondFireConfigure import ResourceAllocator
        allocator = ResourceAllocator()
        
        self.table.setRowCount(0)
        
        # 소방차 현황 추가
        for truck_type in allocator.truck_types:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"소방차 {truck_type}"))
            self.table.setItem(row, 1, QTableWidgetItem("대기 중"))
            self.table.setItem(row, 2, QTableWidgetItem("기지"))
        
        # 인력 현황 추가
        for personnel_type in allocator.firefighter_types:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"인력 {personnel_type}"))
            self.table.setItem(row, 1, QTableWidgetItem("대기 중"))
            self.table.setItem(row, 2, QTableWidgetItem("기지"))

    def apply_resource_settings(self):
        # 자원 최적화 실행 버튼 클릭 시 드롭박스 비활성화
        self.truck_type_combo.setEnabled(False)
        self.truck_quantity_spin.setEnabled(False)
        self.personnel_type_combo.setEnabled(False)
        self.personnel_quantity_spin.setEnabled(False)

        """자원 설정을 적용하고 대시보드 업데이트"""
        truck_type = self.truck_type_combo.currentText()
        truck_quantity = self.truck_quantity_spin.value()
        personnel_type = self.personnel_type_combo.currentText()
        personnel_quantity = self.personnel_quantity_spin.value()

        # ResourceAllocator에 설정 적용
        from code.test.LinearProgramming.respondFireConfigure import ResourceAllocator
        allocator = ResourceAllocator()
        allocator.set_resource_deployment('truck', truck_type, truck_quantity)
        allocator.set_resource_deployment('firefighter', personnel_type, personnel_quantity)

        # 테이블 업데이트
        for row in range(self.table.rowCount()):
            resource_name = self.table.item(row, 0).text()
            if f"소방차 {truck_type}" in resource_name:
                status = "대기중" if truck_quantity == 0 else f"배치 중 ({truck_quantity}대)"
                self.table.item(row, 1).setText(status)
            elif f"인력 {personnel_type}" in resource_name:
                status = "대기중" if personnel_quantity == 0 else f"배치 중 ({personnel_quantity}명)"
                self.table.item(row, 1).setText(status)

        # 대시보드 업데이트
        if hasattr(self, 'dashboard_tab'):
            self.dashboard_tab.update_resource_status()

    def update_resource_locations(self, results):
        """자원 배치 결과에 따라 위치 정보 업데이트 (동기 함수)"""
        # 이벤트 루프 생성 및 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_update_resource_locations(results))
        loop.close()

    async def _async_update_resource_locations(self, results):
        """자원 배치 결과에 따라 위치 정보 업데이트"""
        # 모든 좌표에 대한 주소 요청 태스크 생성
        tasks = []
        for result in results:
            lon = result['longitude']
            lat = result['latitude']
            tasks.append(self.get_road_address_from_coords(lon, lat))
    
        # 모든 주소를 병렬로 조회
        addresses = await asyncio.gather(*tasks)
    
        # 테이블 업데이트
        address_idx = 0
        for address_idx, result in enumerate(results):
            for row in range(self.table.rowCount()):
                resource_name = self.table.item(row, 0).text()
                # 소방차 처리
                if f"소방차 {result['type']}" in resource_name:
                    self.table.setItem(row, 1, QTableWidgetItem(f"배치 완료 ({result['quantity']}대)"))
                    self.table.setItem(row, 2, QTableWidgetItem(addresses[address_idx]))
                    address_idx += 1
                # 인력 처리
                elif f"인력 {result['type']}" in resource_name:
                    self.table.setItem(row, 1, QTableWidgetItem(f"배치 완료 ({result['quantity']}명)"))
                    self.table.setItem(row, 2, QTableWidgetItem(addresses[address_idx]))
                    address_idx += 1



    def connect_dashboard(self, dashboard_tab):
        """대시보드 탭과 연결"""
        self.dashboard_tab = dashboard_tab

    def get_all_resource_settings(self):
        """모든 자원(소방차, 인력)의 배치 수량을 딕셔너리로 반환"""
        truck_settings = {}
        personnel_settings = {}
        for row in range(self.table.rowCount()):
            resource_name = self.table.item(row, 0).text()
            status = self.table.item(row, 1).text()
            # 소방차
            if resource_name.startswith("소방차"):
                truck_type = resource_name.split()[1]
                if "배치 중" in status:
                    # 예: '배치 중 (2대)'
                    try:
                        qty = int(status.split("(")[1].split("대")[0])
                    except:
                        qty = 0
                else:
                    qty = 0
                truck_settings[truck_type] = qty
            # 인력
            elif resource_name.startswith("인력"):
                personnel_type = resource_name.split()[1]
                if "배치 중" in status:
                    # 예: '배치 중 (3명)'
                    try:
                        qty = int(status.split("(")[1].split("명")[0])
                    except:
                        qty = 0
                else:
                    qty = 0
                personnel_settings[personnel_type] = qty
        return truck_settings, personnel_settings


class HistoryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.date_filter = QDateEdit()
        self.date_filter.setCalendarPopup(True)
        self.date_filter.setDate(QDate.currentDate())

        self.severity_filter = QComboBox()
        self.severity_filter.addItems(["전체", "낮음", "보통", "높음", "심각"])

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("날짜:"))
        filter_layout.addWidget(self.date_filter)
        filter_layout.addWidget(QLabel("위험도:"))
        filter_layout.addWidget(self.severity_filter)

        self.refresh_button = QPushButton("기록 조회")
        self.refresh_button.clicked.connect(self.load_logs)

        self.layout.addLayout(filter_layout)
        self.layout.addWidget(self.log_view)
        self.layout.addWidget(self.refresh_button)
        self.setLayout(self.layout)

    def connect_dashboard(self, dashboard: DashboardTab):
        self.dashboard = dashboard

    def load_logs(self):
        if hasattr(self, "dashboard"):
            selected_date = self.date_filter.date().toString("yyyy-MM-dd")
            selected_level = self.severity_filter.currentText()

            filtered_logs = []
            for log in self.dashboard.fire_logs:
                date_match = selected_date in log
                level_match = True
                if selected_level != "전체":
                    level_match = selected_level in log
                if date_match and level_match:
                    filtered_logs.append(log)

            self.log_view.setPlainText("\n".join(filtered_logs))
        else:
            self.log_view.setPlainText("대시보드 연결 실패")

import cv2
class VideoTab(QWidget):
    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.layout = QVBoxLayout()

        self.video_player = QLabel(parent=self)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(stretch=1)  # 오른쪽 정렬

        btn_layout_v = QVBoxLayout()
        btn_layout_v.addStretch(stretch=10)

        self.analyze_btn = QPushButton("영상 분석")
        self.analyze_btn.clicked.connect(self.analyze_video)
        btn_layout.addWidget(self.analyze_btn)

        self.delete_btn = QPushButton("영상 삭제")
        self.delete_btn.clicked.connect(self.delete_video)
        btn_layout.addWidget(self.delete_btn)

        self.layout.addLayout(btn_layout)
        self.layout.addLayout(btn_layout_v)
        self.setLayout(self.layout)

    def analyze_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if not fname:
            self.status_bar.showMessage("영상 파일을 선택해주세요.")
            return
        
        self.cap = cv2.VideoCapture(fname)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_player.resize(int(width), int(height))

        if self.cap is None:
            self.status_bar.showMessage("영상 파일 읽기 실패")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qImg = QImage(img.data, w, h, ch*w, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qImg)
            self.video_player.setPixmap(self.pixmap)
            cv2.waitKey(int(self.cap.get(cv2.CAP_PROP_FPS)))

        self.cap.release()

    def delete_video(self):
        self.cap.release()
        self.video_player.clear()
        self.status_bar.showMessage("영상 삭제 완료")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Qt WebEngine 설정
    qtwebengine_process = find_qtwebengine_process()
    if qtwebengine_process:
        os.environ['QTWEBENGINEPROCESS_PATH'] = qtwebengine_process
        os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--single-process'
        
        # WebEngine 프로필 설정
        profile = QWebEngineProfile.defaultProfile()
        profile.setPersistentStoragePath(os.path.join(MAP_DIR, 'webengine_data'))
        profile.setCachePath(os.path.join(MAP_DIR, 'webengine_cache'))
    
    mainWin = FireGuardApp()
    mainWin.show()
    sys.exit(app.exec_())
