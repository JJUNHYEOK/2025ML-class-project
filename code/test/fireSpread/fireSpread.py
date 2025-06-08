import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
import random
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

class FireCell:
    STATES = ['UNBURNED', 'BURNING', 'BURNED', 'SUPPRESSED']
    
    def __init__(self, fuel_type, moisture):
        self.state = 'UNBURNED'
        self.fuel_type = fuel_type
        self.moisture = moisture
        self.burn_time = 0

class FireSpreadSimulator:
    def __init__(self, grid_size=100, resolution=30, burn_time=3, 
                 wind_speed=2.0, wind_direction=(0,1), fuel_moisture=0.1,
                 slope=0, ignition_points=[(50,50)], fuel_grid=None):
        
        # 시뮬레이션 파라미터
        self.grid_size = grid_size    # 격자 크기 (N x N)
        self.resolution = resolution  # 셀 해상도 (미터)
        self.burn_time = burn_time    # 셀 연소 지속시간(타임스텝)
        
        # 환경 조건
        self.wind_speed = wind_speed          # 풍속 (m/s)
        self.wind_direction = wind_direction  # 풍향 (x,y 벡터)
        self.fuel_moisture = fuel_moisture     # 연료 수분(0-1)
        self.slope = slope                    # 경사도(도)
        
        # 초기화재 위치 설정
        self.ignition_points = ignition_points
        self.time_step = 0

        # 격자 초기화: fuel_grid가 있으면 사용, 없으면 기존 방식대로
        if fuel_grid is not None:
            self.grid = [[FireCell(fuel_grid[i][j], fuel_moisture) for j in range(grid_size)] 
                        for i in range(grid_size)]
        else:
            # 기존 코드 (균일한 연료)
            self.grid = [[FireCell(1, fuel_moisture) for _ in range(grid_size)] 
                        for _ in range(grid_size)]
        
        # 초기 점화
        for x, y in ignition_points:
            self.grid[x][y].state = 'BURNING'
            self.grid[x][y].burn_time = burn_time

    def calculate_spread_prob(self, from_cell, to_cell):
        """Rothermel 모델 기반 확산 확률 계산 (단순화 버전)"""
        dx = to_cell[0] - from_cell[0]
        dy = to_cell[1] - from_cell[1]
        
        # 풍향 영향
        wind_effect = (dx*self.wind_direction[0] + dy*self.wind_direction[1]) 
        wind_effect *= self.wind_speed * 0.1
        
        # 연료 수분 영향
        moisture_effect = exp(-2 * self.fuel_moisture)
        
        # 경사 영향 
        slope_effect = 1 + 0.05 * self.slope
        
        # 기본 확률 + 풍향 영향
        base_prob = 0.3 + 0.2 * wind_effect
        final_prob = base_prob * moisture_effect * slope_effect

                # 목표 셀의 연료 타입 확인
        to_cell_i, to_cell_j = to_cell
        target_fuel_type = self.grid[to_cell_i][to_cell_j].fuel_type
        fuel_effect = 1.0 + (target_fuel_type * 0.1) #가중치 조절해봐야함

        base_prob = 0.3 + 0.2 * wind_effect
        final_prob = base_prob * moisture_effect * slope_effect * fuel_effect
        
        return np.clip(final_prob, 0, 1)

    def propagate_fire(self):
        """1타임스텝 화재 확산 시뮬레이션"""
        new_burning = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j].state == 'BURNING':
                    # 연소 시간 감소
                    self.grid[i][j].burn_time -= 1
                    if self.grid[i][j].burn_time <= 0:
                        self.grid[i][j].state = 'BURNED'
                    
                    # 8방향 이웃으로 확산
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue  # 자기 자신 제외
                                
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                                neighbor = self.grid[ni][nj]
                                if neighbor.state == 'UNBURNED':
                                    prob = self.calculate_spread_prob((i,j), (ni,nj))
                                    if random.random() < prob:
                                        new_burning.append((ni, nj))
                                        
        # 새로운 연소 셀 업데이트
        for x, y in new_burning:
            self.grid[x][y].state = 'BURNING'
            self.grid[x][y].burn_time = self.burn_time
            
        self.time_step += 1

    def run(self, steps=20):
        """지정된 시간 동안 화재 확산 시뮬레이션 실행"""
        history = []
        for _ in range(steps):
            self.propagate_fire()
            history.append(np.array([[self._cell_to_value(cell) 
                                    for cell in row] 
                                    for row in self.grid]))
        return history

    def get_burned_area(self):
        """전체 연소 면적 계산"""
        burned = 0
        for row in self.grid:
            for cell in row:
                if cell.state == 'BURNED':
                    burned += 1
        return burned * (self.resolution ** 2)

    def get_fire_perimeter(self):
        """화재 둘레 계산"""
        perimeter = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j].state == 'BURNING':
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+dx, j+dy
                        if not (0 <= ni < self.grid_size and 0 <= nj < self.grid_size):
                            perimeter += 1
                        elif self.grid[ni][nj].state not in ['BURNING', 'BURNED']:
                            perimeter += 1
        return perimeter * self.resolution

    def _cell_to_value(self, cell):
        """시각화를 위한 셀 상태 매핑"""
        return {
            'UNBURNED': 0,
            'BURNING': 1,
            'BURNED': 2,
            'SUPPRESSED': 3
        }[cell.state]

    def visualize(self, grid, timestep):
        """화재 확산 시각화"""
        plt.figure(figsize=(8,8))
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.title(f'화재 확산 (시간: {timestep}시간)')
        plt.colorbar(label='화재 강도')
        plt.axis('off')
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 시뮬레이션 파라미터 설정
    sim = FireSpreadSimulator(
        grid_size=100,
        resolution=30,  # 30m/셀
        wind_speed=5.0, 
        wind_direction=(1,0),  # 동풍
        ignition_points=[(50,50)]
    )
    
    # 24시간(4타임스텝) 시뮬레이션 실행
    history = sim.run(steps=4)
    
    # 결과 시각화
    for t, grid in enumerate(history):
        sim.visualize(grid, t*6)
