from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys

from code.test.fireSpread.fireSpread import FireSpreadSimulator
from code.Front.index import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

STEP = 5

class IndexPopup(QDialog):
    def __init__(self, scenario, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AIWRS 화재확산 시뮬레이터")
        self.scenario = scenario
        self.setupUi(self)
        self.setModal(False)  # 비모달 다이얼로그로 설정
        self.history = None  # 시뮬레이션 결과를 저장할 변수

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(600, 500)  # 창 크기 조정

        # 레이아웃 생성
        layout = QVBoxLayout()

        # 시뮬레이션 버튼
        self.btn_simulate = QPushButton("시뮬레이션 실행")
        self.btn_simulate.clicked.connect(self.simulate)
        layout.addWidget(self.btn_simulate)

        # Matplotlib 캔버스 추가
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # 다이얼로그에 레이아웃 설정
        Dialog.setLayout(layout)

    def simulate(self):
        try:
            # FireScenario 객체에서 cluster_stats에서 데이터 추출
            wind_speed = self.scenario['wind_speed']
            fuel_moisture = 1 - self.scenario['humidity'] / 100
            slope = self.scenario['slope']

            self.sim = FireSpreadSimulator(
                grid_size=100,
                resolution=30,
                burn_time=3,
                wind_speed=wind_speed,
                wind_direction=(0, 1),
                fuel_moisture=fuel_moisture,
                slope=slope
            )
            self.history = self.sim.run(steps=STEP)

            if not isinstance(self.history, list) or len(self.history) == 0:
                raise ValueError(f"self.history is empty or not a list: {type(self.history)}, length: {len(self.history)}")

            self.visualize()
            return self.history
        except Exception as e:
            QMessageBox.critical(self, "에러", f"시뮬레이션 중 오류 발생: {str(e)}")
            return None

    def visualize(self):
        try:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.colorbar = None

            if not self.history or len(self.history) == 0:
                raise ValueError("self.history is empty")

            self.current_step = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_visualization)
            self.timer.start(500)  # 500ms 간격

            t = self.current_step
            grid = self.history[t]
            self.im = self.ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(grid) if np.max(grid) > 0 else 1)
            self.colorbar = self.figure.colorbar(self.im, ax=self.ax)
        except Exception as e:
            QMessageBox.critical(self, "에러", f"시각화 중 오류 발생: {str(e)}")

    def update_visualization(self):
        try:
            if self.current_step < len(self.history):
                t = self.current_step
                grid = self.history[t]

                # self.ax.clear()는 사용하지 않음
                self.im.set_data(grid)
                self.im.set_clim(0, np.max(grid) if np.max(grid) > 0 else 1)
                if self.colorbar:
                    self.colorbar.update_normal(self.im)
                self.ax.set_title(f"Fire Spread Simulation - Time Step {t * 6}")
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.canvas.draw()
                self.current_step += 1
            else:
                self.timer.stop()
        except Exception as e:
            QMessageBox.critical(self, "에러", f"시각화 중 오류 발생: {str(e)}")
            self.timer.stop()
