from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys

# 필요한 모듈 임포트 (기존 코드와 동일)
from code.test.fireSpread.fireSpread import FireSpreadSimulator
from code.test.fireSpread.forest import *
from haversine import haversine
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

STEP = 5

class IndexPopup(QDialog):
    def __init__(self, scenario, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AIWRS 화재확산 시뮬레이터")
        self.scenario = scenario
        self.result = result[0]
        self.setupUi(self)
        self.setModal(False)
        self.history = None
        self.sim = None
        self.timer = QTimer(self)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(600, 500)

        layout = QVBoxLayout()

        self.btn_simulate = QPushButton("시뮬레이션 실행")
        self.btn_simulate.clicked.connect(self.simulate)
        layout.addWidget(self.btn_simulate)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        Dialog.setLayout(layout)

    def simulate(self):
        """
        수목 데이터 연동 로직을 통합하여 시뮬레이션을 실행하는 메인 함수.
        """
        try:
            # 1. 시나리오에서 환경 변수 및 화재 위치 추출
            fire_lat = self.result['latitude']
            fire_lon = self.result['longitude']
            wind_speed = self.scenario['wind_speed']
            wind_direction = self.scenario.get('wind_direction', (0, 1)) # 기본값은 북풍
            fuel_moisture = 1 - (self.scenario['humidity'] / 100.0)
            slope = self.scenario['slope']

            # 2. 수목 데이터를 이용한 연료 그리드(fuel_grid) 강화
            fuel_grid = None
            # df_forest가 forest 모듈에서 전역적으로 로드되었다고 가정
            df_forest['distance'] = df_forest.apply(
                lambda row: haversine((fire_lat, fire_lon), (row['latitude'], row['longitude'])),
                axis=1
            )
            closest_point = df_forest.loc[df_forest['distance'].idxmin()]

            if closest_point['distance'] < 15.0:  # 15km 이내에 수목 데이터가 있을 경우
                print(f"인근 수목 데이터 발견: {closest_point['교목우점_species']} (거리: {closest_point['distance']:.2f}km)")
                dominant_fuel_type = closest_point['fuel_type']
                fuel_grid = np.full((100, 100), dominant_fuel_type, dtype=int)
            else:
                print("인근 수목 데이터 없음. 기본 시뮬레이션 실행.")

            # 3. 강화된 파라미터로 시뮬레이터 초기화
            self.sim = FireSpreadSimulator(
                grid_size=100,
                resolution=30,
                burn_time=3,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                fuel_moisture=fuel_moisture,
                slope=slope,
                fuel_grid=fuel_grid  # 강화된 연료 그리드를 전달
            )

            # 4. 시뮬레이션 실행 및 시각화
            self.history = self.sim.run(steps=STEP)

            if not isinstance(self.history, list) or not self.history:
                raise ValueError("시뮬레이션 결과(history)가 비어있거나 유효하지 않습니다.")

            self.visualize()
            return self.history

        except KeyError as e:
            QMessageBox.critical(self, "입력 오류", f"시나리오 데이터가 누락되었습니다: {str(e)} 키가 필요합니다.")
            return None
        except Exception as e:
            QMessageBox.critical(self, "실행 오류", f"시뮬레이션 중 오류 발생: {str(e)}")
            return None

    def visualize(self):
        try:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.colorbar = None

            if not self.history:
                raise ValueError("시각화할 history 데이터가 없습니다.")

            self.current_step = 0
            # 기존 타이머가 실행 중이면 중지
            if self.timer.isActive():
                self.timer.stop()
            
            self.timer.timeout.connect(self.update_visualization)
            self.timer.start(500)  # 0.5초 간격으로 업데이트

            # 첫 프레임 즉시 표시
            self.update_visualization()

        except Exception as e:
            QMessageBox.critical(self, "시각화 오류", f"시각화 중 오류 발생: {str(e)}")

    def update_visualization(self):
        try:
            if self.current_step < len(self.history):
                t = self.current_step
                grid = self.history[t]

                # 첫 스텝에서만 이미지와 컬러바 생성
                if self.current_step == 0:
                    self.im = self.ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=3)
                    self.colorbar = self.figure.colorbar(self.im, ax=self.ax)
                    self.colorbar.set_ticks([0, 1, 2, 3])
                    self.colorbar.set_ticklabels(['UNBURNED', 'BURNING', 'BURNED', 'SUPPRESSED'])
                else:
                    self.im.set_data(grid)

                self.ax.set_title(f"Fire Spread Simulation (Time: {t * 6} Hours)")
                self.ax.set_xlabel("X-axis")
                self.ax.set_ylabel("Y-axis")
                self.canvas.draw()
                
                self.current_step += 1
            else:
                self.timer.stop()
        except Exception as e:
            if self.timer.isActive():
                self.timer.stop()
            # 오류 메시지는 한 번만 표시되도록 조치 가능 (예: 플래그 사용)
            QMessageBox.critical(self, "업데이트 오류", f"시각화 업데이트 중 오류 발생: {str(e)}")

