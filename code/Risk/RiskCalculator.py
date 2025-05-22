from typing import Dict, List, Tuple

class RiskCalculator:
    def __init__(self):
        self.risk_factors = {
            'wind_speed': {
                'weight': 0.3,
                'thresholds': [(0, 0.2), (5, 0.4), (10, 0.6), (15, 0.8), (20, 1.0)]
            },
            'humidity': {
                'weight': 0.2,
                'thresholds': [(80, 0.2), (60, 0.4), (40, 0.6), (20, 0.8), (0, 1.0)]
            },
            'fuel_type': {
                'weight': 0.2,
                'values': {
                    1: 0.2,  # 낮은 연료량
                    2: 0.4,
                    3: 0.6,
                    4: 0.8,
                    5: 1.0   # 높은 연료량
                }
            },
            'slope': {
                'weight': 0.15,
                'thresholds': [(0, 0.2), (10, 0.4), (20, 0.6), (30, 0.8), (45, 1.0)]
            },
            'damage_class': {
                'weight': 0.15,
                'values': {
                    1: 0.2,  # 낮은 피해 등급
                    2: 0.4,
                    3: 0.6,
                    4: 0.8,
                    5: 1.0   # 높은 피해 등급
                }
            }
        }

    def calculate_risk_score(self, risk_factors: Dict) -> float:
        """위험 요소들을 기반으로 위험도 점수 계산 (0-100)"""
        total_score = 0
        total_weight = 0

        for factor, value in risk_factors.items():
            if factor in self.risk_factors:
                factor_info = self.risk_factors[factor]
                weight = factor_info['weight']
                
                if 'thresholds' in factor_info:
                    # 연속형 변수 처리 (풍속, 습도, 경사도)
                    score = self._calculate_continuous_score(value, factor_info['thresholds'])
                else:
                    # 이산형 변수 처리 (연료 유형, 피해 등급)
                    score = factor_info['values'].get(value, 0.5)
                
                total_score += score * weight
                total_weight += weight

        # 가중 평균 계산 및 100점 만점으로 변환
        final_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
        return round(final_score, 1)

    def _calculate_continuous_score(self, value: float, thresholds: List[Tuple[float, float]]) -> float:
        """연속형 변수의 점수 계산"""
        for i in range(len(thresholds) - 1):
            if thresholds[i][0] <= value < thresholds[i + 1][0]:
                # 선형 보간
                x1, y1 = thresholds[i]
                x2, y2 = thresholds[i + 1]
                return y1 + (y2 - y1) * (value - x1) / (x2 - x1)
        
        # 범위를 벗어난 경우
        if value < thresholds[0][0]:
            return thresholds[0][1]
        return thresholds[-1][1]

    def get_risk_level(self, score: float) -> str:
        """위험도 점수를 기반으로 위험 수준 반환"""
        if score >= 80:
            return "심각"
        elif score >= 60:
            return "높음"
        elif score >= 40:
            return "보통"
        elif score >= 20:
            return "낮음"
        else:
            return "매우 낮음"

    def get_risk_factors_description(self, risk_factors: Dict) -> List[str]:
        """위험 요소들의 설명 반환"""
        descriptions = []
        
        if risk_factors.get('wind_speed', 0) >= 15:
            descriptions.append("강한 바람")
        if risk_factors.get('humidity', 100) <= 30:
            descriptions.append("건조한 날씨")
        if risk_factors.get('fuel_type', 1) >= 4:
            descriptions.append("높은 연료량")
        if risk_factors.get('slope', 0) >= 30:
            descriptions.append("가파른 경사")
        if risk_factors.get('damage_class', 1) >= 4:
            descriptions.append("높은 피해 등급")
            
        return descriptions 