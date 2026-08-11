"""
문제 1 — 배선 확인용 레벨.

  도착      : 랜덤 간격 0 ~ 30 틱
  인내도    : 30 틱
  정비 시간 : A / B / C 모두 20 ~ 25 틱 (차량 속성과 무관)

세 정비소가 완전히 같으므로 어디에 배치해도 결과가 같습니다.
따라서 이 레벨의 목표는 점수 경쟁이 아니라
"내가 만든 관측과 보상으로 학습이 실제로 돌아가는가" 를 확인하는 것입니다.
베이스라인 점수 근처에 도달하면 통과입니다.

[중요] 이 파일은 수정하지 마세요. python verify_env.py 로 확인합니다.
"""

import os
import sys

# python env/garage_N.py 로 직접 실행할 때도 동작하도록 저장소 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env._core import GarageEnvBase, run_baseline


class GarageEnv_1(GarageEnvBase):
    PATIENCE = 30
    ARRIVAL_MAX = 30

    def get_repair_time(self, car, station):
        """A, B, C 정비소 모두 20 ~ 25 틱 (랜덤)."""
        return self.rng.randint(20, 25)


if __name__ == "__main__":
    run_baseline(GarageEnv_1, episodes=100)
