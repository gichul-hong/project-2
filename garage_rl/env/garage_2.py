"""
문제 2 — 본 과제.

  도착      : 간격 없음 (매 틱 도착, 대기 공간이 빌 때마다 즉시 채워짐)
  인내도    : 100 틱
  정비 시간 : A 22 ~ 23 틱
              B 21 ~ 24 틱
              C 10 ~ 27 틱, 단 차체 크기 <= 4.0 인 차량은 50% 감소

정비소마다 잘 맞는 차량이 다릅니다. 어떤 차량을 어느 정비소에 넣는지가
총 소요 시간을 바꾸므로, 여기서는 배치 정책이 실제로 성능을 좌우합니다.

[중요] 이 파일은 수정하지 마세요. python verify_env.py 로 확인합니다.
"""

import os
import sys

# python env/garage_N.py 로 직접 실행할 때도 동작하도록 저장소 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env._core import GarageEnvBase, run_baseline


class GarageEnv_2(GarageEnvBase):
    PATIENCE = 100
    ARRIVAL_MAX = 0

    def get_repair_time(self, car, station):
        """
        A 정비소 : 22 ~ 23 틱 (랜덤)
        B 정비소 : 21 ~ 24 틱 (랜덤)
        C 정비소 : 10 ~ 27 틱 (랜덤), 차체 크기 <= 4.0 이면 50% 감소
        """
        if station == 'A':
            base = self.rng.randint(22, 23)
        elif station == 'B':
            base = self.rng.randint(21, 24)
        else:  # 'C'
            base = self.rng.randint(10, 27)
            if car.size <= 4.0:
                base = int(base * 0.5)
        return max(1, base)


if __name__ == "__main__":
    run_baseline(GarageEnv_2, episodes=100)
