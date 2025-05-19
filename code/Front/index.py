import sys
import os
import requests
import folium
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QProgressBar,
    QHeaderView, QLineEdit, QComboBox, QDateEdit
)
from PyQt5.QtCore import Qt, QUrl, QDate
from PyQt5.QtWebEngineWidgets import QWebEngineView

os.environ["QTWEBENGINEPROCESS_PATH"] = r"code/Front/QtWebEngineProcess.exe"


class FireGuardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("파이어가드AI 대시보드")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard_tab = DashboardTab()
        # 지도 탭 제거
        # self.map_tab = MapTab()
        self.resource_tab = ResourceManagementTab()
        self.history_tab = HistoryTab()

        self.history_tab.connect_dashboard(self.dashboard_tab)

        self.tabs.addTab(self.dashboard_tab, "실시간 상황")
        # 지도 탭 제거
        # self.tabs.addTab(self.map_tab, "지도")
        self.tabs.addTab(self.resource_tab, "자원 관리")
        self.tabs.addTab(self.history_tab, "기록 조회")


class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self.fire_logs = []

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

        risk_factors = QLabel("주요 위험 요인:\n- 높은 기온\n- 강한 바람\n- 가뭄 상태\n- 건조한 식생")
        risk_factors.setAlignment(Qt.AlignCenter)
        risk_layout.addWidget(risk_factors)
        risk_box.setLayout(risk_layout)

        # 대응 자원 현황 박스, 4등분 그리드로 수정 및 구급차 추가
        resources_box = QGroupBox("대응 자원 현황")
        resources_box.setFixedSize(280, 280)
        resources_layout = QVBoxLayout()

        grid = QGridLayout()
        # 2x2 그리드로 소방차, 항공 지원, 인력, 구급차 배치
        grid.addWidget(QLabel("소방차"), 0, 0, alignment=Qt.AlignCenter)
        grid.addWidget(QLabel("15/15 대기 중"), 1, 0, alignment=Qt.AlignCenter)

        grid.addWidget(QLabel("항공 지원"), 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(QLabel("4/4 대기 중"), 1, 1, alignment=Qt.AlignCenter)

        grid.addWidget(QLabel("인력"), 2, 0, alignment=Qt.AlignCenter)
        grid.addWidget(QLabel("120/120 대기 중"), 3, 0, alignment=Qt.AlignCenter)

        grid.addWidget(QLabel("구급차"), 2, 1, alignment=Qt.AlignCenter)
        grid.addWidget(QLabel("8/8 대기 중"), 3, 1, alignment=Qt.AlignCenter)

        # 행간 조정 및 공간 확보를 위해 setRowMinimumHeight 등 필요하면 추가 가능

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

        self.update_weather_data()

    def update_weather_data(self):
        api_key = "여기에_발급받은_API_키_넣기"
        lat, lon = 35.1767, 128.1035
        url = (f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=kr")

        try:
            response = requests.get(url)
            data = response.json()

            wind_deg = data["wind"]["deg"]
            wind_speed = data["wind"]["speed"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            rain = data.get("rain", {}).get("1h", 0)
            air_quality = "좋음"

            wind_dir = self.deg_to_direction(wind_deg)

            self.weather_labels["풍향"].setText(f"{wind_dir} 🌬️")
            self.weather_labels["풍속"].setText(f"{wind_speed} m/s")
            self.weather_labels["온도"].setText(f"{temp} °C 🌡️")
            self.weather_labels["습도"].setText(f"{humidity}%")
            self.weather_labels["강수량"].setText(f"{rain} mm 🌧️")
            self.weather_labels["대기질"].setText(air_quality)

            # 자동 화재 발생 기록 (예시: 위험 수준이 70% 이상일 때 기록)
            fire_count = 3
            risk_level = 75
            if risk_level >= 70:
                self.fire_logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 화재 감지 - 위험도: {risk_level}%")

        except Exception as e:
            for label in self.weather_labels.values():
                label.setText("데이터 불러오기 실패")
            print("날씨 데이터 로드 실패:", e)

    def deg_to_direction(self, deg):
        directions = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
        ix = int((deg + 22.5) / 45) % 8
        return directions[ix]


class MapTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        m = folium.Map(location=[35.1767, 128.1035], zoom_start=14)
        folium.Marker(location=[35.1767, 128.1035], tooltip="가좌동").add_to(m)
        folium.CircleMarker(location=[35.18, 128.10], radius=30, color='red', fill=True, fill_opacity=0.5, popup="화재 위험 지역").add_to(m)

        map_path = os.path.abspath("map.html")
        m.save(map_path)

        self.web_view = QWebEngineView()
        self.web_view.load(QUrl.fromLocalFile(map_path))
        layout.addWidget(self.web_view)

        self.setLayout(layout)


class ResourceManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["자원명", "상태", "위치"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        add_box = QHBoxLayout()
        self.name_input = QLineEdit()
        self.status_input = QComboBox()
        self.status_input.addItems(["대기 중", "사용 중"])
        self.loc_input = QLineEdit()
        add_btn = QPushButton("추가")
        add_btn.clicked.connect(self.add_resource)
        add_box.addWidget(self.name_input)
        add_box.addWidget(self.status_input)
        add_box.addWidget(self.loc_input)
        add_box.addWidget(add_btn)

        layout.addLayout(add_box)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_resource(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(self.name_input.text()))
        self.table.setItem(row, 1, QTableWidgetItem(self.status_input.currentText()))
        self.table.setItem(row, 2, QTableWidgetItem(self.loc_input.text()))
        self.name_input.clear()
        self.loc_input.clear()


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = FireGuardApp()
    mainWin.show()
    sys.exit(app.exec_())
