import pulp

# 테스트용 데이터 정의
truck_types = ['TypeA', 'TypeB']
firefighter_types = ['FF1', 'FF2']
scenario = type('FireScenario', (), {
    'id': 1,
    'sites': {
        'Site1': {'distance': {'TypeA': 10, 'TypeB': 15}, 'demand': 5},
        'Site2': {'distance': {'TypeA': 20, 'TypeB': 25}, 'demand': 3}
    }
})
truck_capabilities = {
    'TypeA': {'cost': 100, 'fuel_efficiency': 5, 'speed': 50, 'personnel': 2},
    'TypeB': {'cost': 150, 'fuel_efficiency': 4, 'speed': 40, 'personnel': 3}
}
firefighter_capabilities = {
    'FF1': {'cost': 50, 'max_deployments': 5},
    'FF2': {'cost': 60, 'max_deployments': 4}
}
truck_deployments = {'TypeA': 0, 'TypeB': 0}
firefighter_deployments = {'FF1': 0, 'FF2': 0}

# 모델 정의
model = pulp.LpProblem(f"Fire_Resource_Allocation_Scenario_{scenario.id}", pulp.LpMinimize)

# 결정 변수 정의
x = pulp.LpVariable.dicts("truck_assign", 
                          [(i, n) for i in truck_types for n in scenario.sites.keys()], 
                          cat='Integer', lowBound=0, upBound=2)
y = pulp.LpVariable.dicts("ff_assign", 
                          [(j, n) for j in firefighter_types for n in scenario.sites.keys()], 
                          cat='Integer', lowBound=0, upBound=3)
z = pulp.LpVariable.dicts("truck_location", 
                          [(i, n) for i in truck_types for n in scenario.sites.keys()], 
                          cat='Binary')

# 목적 함수
model += pulp.lpSum(
    truck_capabilities[i]['cost'] * x[(i, n)] +
    firefighter_capabilities[j]['cost'] * y[(j, n)] +
    (scenario.sites[n]['distance'][i] / truck_capabilities[i]['fuel_efficiency']) * 1000 * x[(i, n)]
    for i in truck_types 
    for j in firefighter_types 
    for n in scenario.sites.keys()
)

# 제약 조건 (간략화된 버전)
for i in truck_types:
    model += pulp.lpSum(x[(i, n)] for n in scenario.sites.keys()) <= 2 - truck_deployments[i]
    model += pulp.lpSum(x[(i, n)] for n in scenario.sites.keys()) <= 1
    for n in scenario.sites.keys():
        model += x[(i, n)] <= 2 * z[(i, n)]
        model += x[(i, n)] >= z[(i, n)]

for j in firefighter_types:
    model += pulp.lpSum(y[(j, n)] for n in scenario.sites.keys()) <= \
             (firefighter_capabilities[j]['max_deployments'] - firefighter_deployments[j])

for n in scenario.sites.keys():
    model += pulp.lpSum(truck_capabilities[i]['personnel'] * x[(i, n)] 
                        for i in truck_types) + \
             pulp.lpSum(y[(j, n)] for j in firefighter_types) >= \
             max(1, scenario.sites[n]['demand'] - 1)

# 최적화 실행
model.solve()

# 결과 출력
print("Status:", pulp.LpStatus[model.status])
print("Objective Value (Cost):", pulp.value(model.objective))
for v in model.variables():
    if pulp.value(v) > 0:
        print(f"{v.name}: {pulp.value(v)}")
