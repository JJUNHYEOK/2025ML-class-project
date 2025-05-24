## 화재 확산 시뮬레이션

셀룰러 오토마타 알고리즘 활용. (8방향 그리드 탐색->동기처리)
```python
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
```
