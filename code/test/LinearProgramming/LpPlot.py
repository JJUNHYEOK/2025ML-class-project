import seaborn as sns
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 집합 정의
I = ['FT1', 'FT2']                     # 소방차 집합
J = ['FF1', 'FF2', 'FF3']              # 소방 대원 집합
N = ['site1', 'site2', 'site3']        # 화재 지점 집합

# 파라미터 정의 (예시 값)
cap = {'FT1': 1000, 'FT2': 1200}       # 물탱크 용량 (l)
tp  = {'FT1': 5,    'FT2': 6}          # 수송 가능 인원 (명)
fe  = {'FT1': 8,    'FT2': 7}          # 연비 (km/l)
vs  = {'FT1': 60,   'FT2': 55}         # 평균속도 (km/h)
fc  = {'FT1': 1500, 'FT2': 1400}       # 연료비 (원/l)

d    = {'site1': 4, 'site2': 1, 'site3': 2}  
dist = {('FT1','site1'):10, ('FT1','site2'):20, ('FT1','site3'):15,
        ('FT2','site1'):12, ('FT2','site2'):18, ('FT2','site3'):14}

md    = {'FF1': 2, 'FF2': 5, 'FF3': 1}  # 최대 출동 횟수
T_max = 1.5                             # 최대 허용 시간 (h)
M     = 1e5                             # Big‐M

# 문제 정의
prob = pulp.LpProblem("Fire_Transport", pulp.LpMinimize)

# 결정변수
x = pulp.LpVariable.dicts("truck_assign", 
                          [(i,n) for i in I for n in N], 
                          cat='Binary')
y = pulp.LpVariable.dicts("ff_assign", 
                          [(j,n) for j in J for n in N], 
                          cat='Binary')

# 목적함수: 연료비 최소화
prob += pulp.lpSum((dist[(i,n)]/fe[i]) * fc[i] * x[(i,n)]
                   for i in I for n in N)

# 제약조건
# 소방차 1대당 하나의 지점만 배치
for i in I:
    prob += pulp.lpSum(x[(i,n)] for n in N) <= 1

# 대원별 최대 출동 횟수
for j in J:
    prob += pulp.lpSum(y[(j,n)] for n in N) <= md[j]

# 지점별 인원 수요 충족
for n in N:
    prob += pulp.lpSum(tp[i] * x[(i,n)] + y[(j,n)] for i in I) >= d[n] #소방차량 및 인원 배치까지 전부 고려

# 대원은 배치된 소방차가 있어야 이동 가능
for j in J:
    for n in N:
        prob += pulp.lpSum(x[(i,n)] for i in I) >= y[(j,n)]

# 소방차별 수송인원 상한
for n in N:
    prob += pulp.lpSum(y[(j,n)] for j in J) <= pulp.lpSum(tp[i]*x[(i,n)] for i in I)

# 시간 제약: 도착 시간 ≤ T_max
for i in I:
    for n in N:
        prob += (dist[(i,n)]/vs[i]) <= T_max + M*(1 - x[(i,n)])

# 문제 해결
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# 결과 출력
print("=== 소방차 배치 ===")
for i in I:
    for n in N:
        if pulp.value(x[(i,n)]) > 0.5:
            print(f"  {i} → {n}")

print("\n=== 소방대원 배치 ===")
for j in J:
    for n in N:
        if pulp.value(y[(j,n)]) > 0.5:
            print(f"  {j} → {n}")
        else :
            pass
            #print("장비 외의 인력은 배치하지 않는게 좋습니다.")

def heatmap():
    # 소방차 배치 행렬
    truck_matrix = pd.DataFrame(0.0, index=I, columns=N)
    for i in I:
        for n in N:
            truck_matrix.loc[i, n] = pulp.value(x[(i, n)])

    # 소방대원 배치 행렬
    ff_matrix = pd.DataFrame(0.0, index=J, columns=N)
    for j in J:
        for n in N:
            ff_matrix.loc[j, n] = pulp.value(y[(j, n)])

    # 히트맵 그리기
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(truck_matrix, annot=True, cmap='Blues', cbar=False)
    plt.title("소방차 배치")
    plt.xlabel("화재 지점")
    plt.ylabel("소방차")

    plt.subplot(1, 2, 2)
    sns.heatmap(ff_matrix, annot=True, cmap='Greens', cbar=False)
    plt.title("소방대원 배치")
    plt.xlabel("화재 지점")
    plt.ylabel("소방대원")

    plt.tight_layout()
    plt.show()

def NetworkGraph():
    # 그래프 생성
    G = nx.DiGraph()

    # 노드 추가 (소방차, 소방대원, 화재 지점)
    G.add_nodes_from(I, bipartite=0, color='red', label='Truck')  # 소방차
    G.add_nodes_from(J, bipartite=1, color='blue', label='Firefighter')  # 소방대원
    G.add_nodes_from(N, bipartite=2, color='green', label='Site')  # 화재 지점

    # 엣지 추가 (배치 결과 기반)
    edges = []
    for i in I:
        for n in N:
            if pulp.value(x[(i, n)]) > 0.5:
                edges.append((i, n))  # 소방차 -> 화재 지점
    for j in J:
        for n in N:
            if pulp.value(y[(j, n)]) > 0.5:
                edges.append((j, n))  # 소방대원 -> 화재 지점
    G.add_edges_from(edges)

    # 레이아웃 설정
    pos = nx.spring_layout(G)

    # 노드 색상 및 라벨 설정
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    node_labels = {node: node for node in G.nodes}

    # 그래프 그리기
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, 
            node_size=800, font_size=10, font_color='white', edge_color='gray')
    plt.title("소방차 및 소방대원 배치 네트워크")
    plt.show()



heatmap()
NetworkGraph()