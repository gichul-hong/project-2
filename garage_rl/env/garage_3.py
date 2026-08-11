"""
문제 3 — 평가용 레벨.

  도착      : 간격 없음 (매 틱 도착)
  인내도    : 30 틱
  정비 시간 : A 34 ~ 36 틱, 손상도가 클수록 오래 걸림, 모든 연식 허용
              B 15 ~ 37 틱, 연식 >= 20 인 차량은 30% 감소
              C 12 ~ 37 틱, 차체 크기 <= 4.0 인 차량은 50% 감소

문제 2 와 달리 정비소별 조건이 두 개이고, 인내도가 30 틱뿐입니다.
정비 시간이 인내도보다 길기 때문에 대기열이 밀리면 차가 떠납니다.
규칙 기반 베이스라인은 여기서 100대 중 21대를 놓칩니다.

student/solution.py 를 그대로 적용해 학습하고 제출하세요.

[중요] 이 파일은 수정하지 마세요. 채점 시 해시를 검증합니다.
"""

import os
import sys

# python env/garage_N.py 로 직접 실행할 때도 동작하도록 저장소 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env._core import GarageEnvBase, run_baseline

DAMAGE_ALPHA = 1.5   # A 정비소에서 손상도 1.0 이 늘리는 비율 (x2.5)


class GarageEnv_3(GarageEnvBase):
    PATIENCE = 30
    ARRIVAL_MAX = 0

    def get_repair_time(self, car, station):
        """
        정비소마다 보는 속성이 하나씩 다릅니다.

        A 정비소 : 15 ~ 19 틱  x (1 + 1.5 x 손상도)   손상도가 낮을수록 빠름
        B 정비소 : 26 ~ 36 틱, 연식 >= 20 이면  x 0.5  (오래된 차에 유리)
        C 정비소 : 26 ~ 36 틱, 크기 <= 4.0 이면  x 0.35 (작은 차에 유리)

        A 만 연속적으로 변합니다 (손상도 0.0 이면 15~19, 1.0 이면 38~48).
        B 와 C 는 조건을 넘느냐 마느냐로 갈립니다.
        """
        if station == 'A':
            t = self.rng.randint(15, 19) * (1.0 + DAMAGE_ALPHA * car.damage)
        elif station == 'B':
            t = self.rng.randint(26, 36)
            if car.year >= 20:
                t *= 0.5
        else:  # 'C'
            t = self.rng.randint(26, 36)
            if car.size <= 4.0:
                t *= 0.35

        return max(1, int(round(t)))


if __name__ == "__main__":
    run_baseline(GarageEnv_3, episodes=100)
