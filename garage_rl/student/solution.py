"""
=======================================================================
  여러분이 작성하는 유일한 파일입니다.
  env/ 아래의 파일은 수정하지 마세요 (원본 여부를 해시로 확인합니다).
=======================================================================

두 문제(garage_1 / garage_2)가 모두 이 파일 하나를 사용합니다.
따라서 문제마다 다른 값을 하드코딩하면 안 됩니다.
예를 들어 인내도는 문제 1 이 30, 문제 2 가 100 이므로
반드시 env.max_patience 로 나누어 정규화하세요.


환경에서 읽을 수 있는 정보
-------------------------------------------------------------------
  env.max_waiting          대기 공간 최대 수용 대수 (= 3)
  env.max_patience         이 레벨의 인내도 최댓값 (정규화에 사용)
  env.current_time         현재 틱

  env.waiting_area         대기 중인 Car 리스트 (0 ~ 3 대, 앞이 먼저 온 차)
      car.size             차체 크기  3.0 ~ 5.0
      car.year             연식       10 ~ 25
      car.damage           손상도     0.0 ~ 1.0
      car.id               차량 고유 번호
  env.car_patience[car.id] 그 차량의 남은 인내도 (틱)

  env.repair_status[st]    st 는 'A', 'B', 'C'
      None 이면 비어 있음
      (car, 남은틱, 할당틱) 이면 정비 중
      -> 진행률은 (할당틱 - 남은틱) / 할당틱


행동(action) 정의 — 참고용, 여기서 정하는 것이 아닙니다
-------------------------------------------------------------------
  0        대기 (아무것도 하지 않음)
  1, 2, 3  waiting_area[0] 을 각각 A, B, C 정비소에 배치
  4, 5, 6  waiting_area[1] 을 각각 A, B, C 정비소에 배치
  7, 8, 9  waiting_area[2] 을 각각 A, B, C 정비소에 배치

  이미 차량이 있는 정비소에 배치하려 하면 아무 일도 일어나지 않습니다.


평가 지표
-------------------------------------------------------------------
  점수 = 소요 틱 수 + 50 * 이탈 차량 수      (낮을수록 좋음)
"""

import numpy as np


# =====================================================================
# 관측 벡터 구성 (총 27차원) — 레벨 무관 (하드코딩 없음)
#   [0:3]   slot_occupied      대기 슬롯 i 에 차가 있는가
#   [3:6]   station_busy       정비소 A/B/C 가동 중인가
#   [6:9]   station_progress   (할당틱-남은틱)/할당틱, idle=0
#   [9:12]  station_rem_ticks  정비소별 절대 남은 틱 / T_NORM, idle=0
#                              (진행률만으로는 "몇 틱 뒤에 비는지"를 알 수 없음)
#   [12:24] per_car × 3 slots  [size, year, damage, patience] 정규화
#   [24:27] car_patience_abs   차량별 절대 남은 인내도 틱 / T_NORM
#                              (patience_norm 은 레벨 상대값이라
#                               station_rem_ticks 와 단위가 다름 —
#                               같은 틱/T_NORM 스케일로 맞춰 "정비소가 비기
#                               전에 차가 떠나는가"를 직접 비교 가능하게 함)
#
#   T_NORM = 50.0 : 시간 정규화 상수 (틱). 레벨 규칙이 아니라 관측
#   스케일링 상수이며, 이를 넘는 값은 1.0 으로 clip 된다.
# =====================================================================
T_NORM = 50.0

OBS_DIM = 3 + 3 + 3 + 3 + 3 * 4 + 3


def get_observation(env):
    vec = np.zeros(OBS_DIM, dtype=np.float32)

    n = len(env.waiting_area)
    for i in range(env.max_waiting):
        vec[i] = 1.0 if i < n else 0.0

    for j, st in enumerate(['A', 'B', 'C']):
        status = env.repair_status[st]
        if status is not None:
            car, remaining, assigned = status
            vec[3 + j] = 1.0
            vec[6 + j] = (assigned - remaining) / assigned
            vec[9 + j] = min(1.0, remaining / T_NORM)

    for i in range(env.max_waiting):
        if i < n:
            car = env.waiting_area[i]
            patience = env.car_patience[car.id]
            base = 12 + i * 4
            vec[base + 0] = (car.size - 3.0) / 2.0
            vec[base + 1] = (car.year - 10.0) / 15.0
            vec[base + 2] = car.damage
            vec[base + 3] = min(1.0, patience / env.max_patience)
            vec[24 + i] = min(1.0, patience / T_NORM)

    return vec


# =====================================================================
# 보상 설계 — 평가 지표(점수 = 틱 + 50×이탈)와 정렬
#   틱당 페널티 : 이탈 페널티 = 1 : 50 비율을 유지해
#   보상 최대화가 곧 점수 최소화가 되게 한다.
#   done/assigned 보너스는 사용하지 않는다
#   (done 은 상수 신호, assigned 는 오배치를 부추기는 보상 해킹 위험).
# =====================================================================
R_SCALE       = 0.01   # 전체 보상 스케일 (틱당 페널티)
R_EXPIRE_MULT = 50.0   # 이탈 가중 배율 (지표의 50 에 대응)
R_INVALID     = 0.02   # 점유된 정비소 배치 시도 억제 (소량)
R_FINISH      = 0.05   # 정비 완료 보너스 (throughput 유도 — 과도한 대기 억제)


def compute_reward(env, event) -> float:
    reward = -R_SCALE                                     # 시간 압박
    reward -= event['expired'] * R_SCALE * R_EXPIRE_MULT  # 이탈 = 점수 +50 에 대응
    if event['invalid']:
        reward -= R_INVALID
    reward += event['finished'] * R_FINISH
    return reward
