"""
정비소 시뮬레이터 공통 코어.

[중요] 이 파일과 garage_1/2.py 는 수정하지 마세요. 고치면 문제 자체가
       달라져서 다른 사람 결과와 비교할 수 없습니다.
       여러분이 작성하는 파일은 student/solution.py 하나입니다.
       python verify_env.py 로 원본과 같은지 확인할 수 있습니다.

관측(observation)과 보상(reward)은 student/solution.py 에 위임됩니다.
시뮬레이션 동역학(차량 도착, 정비 시간, 인내도, 종료 조건)은 고정입니다.
"""

import random
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TICK = 1                 # step() 한 번이 1틱
SLEEP_TIME = 0.001       # debug 시각화용 sleep (초)
MAX_CAR = 100            # 한 에피소드에 방문하는 총 차량 수
MAX_TICKS = 50_000       # 안전 상한 (초과 시 truncated=True)

STATIONS = ('A', 'B', 'C')


class Car:
    """대기 공간에 들어온 차량 한 대."""

    def __init__(self, car_id, patience, rng):
        self.id = car_id
        self.size = round(rng.uniform(3.0, 5.0), 2)    # 차체 크기 3.0 ~ 5.0
        self.year = rng.randint(10, 25)                # 연식 10 ~ 25
        self.damage = round(rng.uniform(0.0, 1.0), 2)  # 손상도 0.0 ~ 1.0
        self.patience = patience                       # 입장 시 인내도 (틱)

    def __repr__(self):
        return (f"Car(id={self.id}, size={self.size}, "
                f"year={self.year}, damage={self.damage}, pat={self.patience})")


class GarageEnvBase(gym.Env):
    """
    레벨 공통 환경. 하위 클래스가 다음 세 가지만 정의합니다.

        PATIENCE     : 차량 입장 시 인내도 (틱)
        ARRIVAL_MAX  : 다음 배치 도착까지의 간격을 rng.randint(0, ARRIVAL_MAX) 로 뽑음
                       0 이면 매 틱 도착 (간격 없음)
        get_repair_time(car, station) -> int
    """

    PATIENCE = 30
    ARRIVAL_MAX = 30

    def __init__(self, debug=False, seed=None, need_obs=True):
        """
        need_obs=False 로 만들면 student/solution.py 를 쓰지 않습니다.
        규칙 기반 베이스라인은 관측이 필요 없으므로, TODO 를 채우기 전에도
        python env/garage_N.py 로 기준 점수를 확인할 수 있습니다.
        """
        super().__init__()
        self.debug = debug

        # --- 시뮬레이션 파라미터 (학생이 참조 가능) ---
        self.max_waiting = 3                  # 대기 공간 최대 수용 대수
        self.max_patience = self.PATIENCE     # 인내도 정규화에 사용하세요
        self.max_car = MAX_CAR

        # --- 난수 생성기: 환경마다 독립 (전역 random 을 오염시키지 않음) ---
        self.rng = random.Random(seed)

        self._reset_state()

        # --- 액션 공간: 0 = 대기, 1..9 = (차량 순번 * 3 + 정비소 번호) ---
        self.action_space = spaces.Discrete(self.max_waiting * 3 + 1)

        # --- 관측 공간: 크기는 student/solution.py 의 OBS_DIM 이 결정 ---
        self._solution = None
        if need_obs:
            from student import solution
            obs_dim = int(solution.OBS_DIM)
            if obs_dim <= 0:
                raise ValueError(
                    "student/solution.py 의 OBS_DIM 이 아직 0 입니다 (TODO 1). "
                    "관측 벡터의 길이를 정한 뒤 다시 실행하세요.\n"
                    "규칙 기반 베이스라인 점수만 보려면 python env/garage_N.py "
                    "를 실행하세요 (TODO 를 채우지 않아도 동작합니다)."
                )
            self._solution = solution
        else:
            obs_dim = 1

        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        if self._solution is not None:
            self.get_observation()      # 첫 관측을 만들어 형식을 확인한다

    # ------------------------------------------------------------------ #
    # 내부 상태
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.total_cars_created = 0
        self.car_id_counter = 0
        self.success_count = 0
        self.removed_count = 0
        self.waiting_area: list = []                     # List[Car]
        self.car_patience: dict = {}                     # {car.id: 남은 인내도}
        self.repair_status: dict = {s: None for s in STATIONS}   # {station: (car, 남은틱, 할당틱) | None}
        self.current_time = 0
        self.arrival_timer = self.rng.randint(0, self.ARRIVAL_MAX)

    def _check_observation(self, obs):
        """
        학생이 흔히 겪는 실수를 알아보기 쉬운 오류로 바꿔 줍니다.

        매 스텝 호출합니다. 대기 공간이 빈 첫 관측만 검사하면
        "인내도를 100 으로 하드코딩" 같은 실수를 놓치기 때문입니다
        (문제 1 은 인내도가 30 이라 값이 1 을 넘습니다).
        비용은 500k 스텝 학습에 0.6초 (0.5%) 수준입니다.
        """
        obs = np.asarray(obs)
        expected = self.observation_space.shape[0]
        if obs.shape != (expected,):
            raise ValueError(
                f"get_observation() 이 길이 {obs.shape} 벡터를 돌려줬는데 "
                f"OBS_DIM 은 {expected} 입니다. 둘을 맞춰 주세요."
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError("관측 벡터에 nan 또는 inf 가 있습니다.")
        lo, hi = float(obs.min()), float(obs.max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f"관측 값이 [0, 1] 범위를 벗어났습니다 "
                f"(최소 {lo:.3f}, 최대 {hi:.3f}, {self.current_time}틱). "
                f"정규화가 필요합니다. 인내도는 env.max_patience "
                f"(이 문제는 {self.max_patience}), 연식은 10~25, "
                f"차체 크기는 3.0~5.0 범위입니다."
            )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        if self.debug:
            print('\n--- Simulation reset ---')
        self._reset_state()
        return self.get_observation(), {}

    def step(self, action):
        event = {
            'action': int(action),
            'assigned': False,    # 이번 스텝에 차량을 정비소에 넣었는가
            'invalid': False,     # 이미 점유된 정비소에 넣으려 했는가
            'expired': 0,         # 이번 스텝에 인내도가 0이 되어 떠난 차량 수
            'finished': 0,        # 이번 스텝에 정비가 끝난 차량 수
            'done': False,        # 이번 스텝에 에피소드가 끝났는가
        }

        # 1) 액션 처리
        if action != 0:
            idx = (action - 1) // 3
            station = STATIONS[(action - 1) % 3]
            if idx < len(self.waiting_area):
                car = self.waiting_area[idx]
                if self.repair_status[station] is not None:
                    event['invalid'] = True
                    if self.debug:
                        print(f"[tick {self.current_time}] \033[91m"
                              f"!! Invalid assign: station {station} occupied\033[0m")
                else:
                    repair_time = self.get_repair_time(car, station)
                    self.repair_status[station] = (car, repair_time, repair_time)
                    self.waiting_area.pop(idx)
                    del self.car_patience[car.id]
                    event['assigned'] = True
                    if self.debug:
                        print(f"[tick {self.current_time}] \033[96m"
                              f"-> Assigned {car} to {station} "
                              f"(t={repair_time} ticks)\033[0m")

        # 2) 도착 타이머 감소 및 배치 생성
        self.arrival_timer -= 1
        if self.arrival_timer <= 0:
            if len(self.waiting_area) < self.max_waiting:
                self._spawn_batch()
            self.arrival_timer = self.rng.randint(0, self.ARRIVAL_MAX)

        # 3) 인내도 감소 및 만료 처리
        for cid in list(self.car_patience):
            self.car_patience[cid] -= 1
            if self.car_patience[cid] <= 0:
                self._remove_car(cid)
                event['expired'] += 1
                if self.debug:
                    print(f"[tick {self.current_time}] \033[35m"
                          f"-- Car {cid} patience expired, removed\033[0m")

        # 4) 정비소 진행
        for st, status in list(self.repair_status.items()):
            if status is not None:
                car, rem, assigned = status
                rem -= 1
                if rem <= 0:
                    event['finished'] += 1
                    self._finish_repair(st)
                else:
                    self.repair_status[st] = (car, rem, assigned)

        # 5) 시간 갱신 및 종료 판정
        self.current_time += 1
        terminated = (
            self.total_cars_created >= MAX_CAR
            and not self.waiting_area
            and all(s is None for s in self.repair_status.values())
        )
        truncated = (not terminated) and self.current_time >= MAX_TICKS
        event['done'] = bool(terminated)

        obs = self.get_observation()
        if self._solution is None:
            reward = 0.0
        else:
            reward = float(self._solution.compute_reward(self, event))
        info = {'serviced': self.success_count, 'removed': self.removed_count}
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        if self._solution is None:
            return np.zeros(1, dtype=np.float32)
        obs = np.asarray(self._solution.get_observation(self), dtype=np.float32)
        self._check_observation(obs)
        return obs

    # ------------------------------------------------------------------ #
    # 동역학 (고정)
    # ------------------------------------------------------------------ #
    def _spawn_batch(self):
        slots = self.max_waiting - len(self.waiting_area)
        if slots <= 0:
            return
        batch = min(self.rng.randint(1, slots), MAX_CAR - self.total_cars_created)
        for _ in range(batch):
            car = Car(self.car_id_counter, self.PATIENCE, self.rng)
            self.waiting_area.append(car)
            self.car_patience[car.id] = car.patience
            self.total_cars_created += 1
            self.car_id_counter += 1
            if self.debug:
                print(f"[tick {self.current_time}] \033[93m+ Car arrived: {car} "
                      f"(waiting={len(self.waiting_area)}/{self.max_waiting})\033[0m")

    def _remove_car(self, cid):
        self.waiting_area = [c for c in self.waiting_area if c.id != cid]
        del self.car_patience[cid]
        self.removed_count += 1

    def _finish_repair(self, station):
        car, _, _ = self.repair_status[station]
        self.repair_status[station] = None
        self.success_count += 1
        if self.debug:
            print(f"[tick {self.current_time}] \033[92m<- Car {car.id} done at "
                  f"{station} (total serviced={self.success_count})\033[0m")

    def get_repair_time(self, car, station):
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# 규칙 기반 베이스라인 (python env/garage_N.py 로 실행)
# ---------------------------------------------------------------------- #
def baseline_action(env):
    """빈 정비소가 있으면 대기열 맨 앞(idx=0) 차량을 A, B, C 순으로 배치."""
    if env.waiting_area:
        for station_idx, st in enumerate(STATIONS):
            if env.repair_status[st] is None:
                return station_idx + 1
    return 0


def run_baseline(env_cls, episodes=100, debug=False):
    """베이스라인 점수를 출력합니다. 점수 = 소요 틱 + 50 * 이탈 차량 수 (낮을수록 좋음)."""
    scores = []
    for i in range(episodes):
        env = env_cls(debug=debug, seed=i, need_obs=False)
        env.reset(seed=i)
        total_reward = 0.0
        step = 0
        while True:
            _, reward, terminated, truncated, info = env.step(baseline_action(env))
            total_reward += reward
            step += 1
            if terminated or truncated:
                break
            if debug:
                time.sleep(SLEEP_TIME)
        scores.append(step + 50 * info['removed'])
        if debug or episodes == 1:
            print("\n\033[48;5;208m\033[1m\033[38;5;0m", end="")
            print("=== Simulation Finished ===")
            print(f" Steps taken : {step}")
            print(f" Total reward: {total_reward:.2f}")
            print(f" Serviced    : {info['serviced']}")
            print(f" Removed     : {info['removed']}")
            print("\033[0m")

    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    print(f"\n[{env_cls.__name__}] baseline over {episodes} episodes")
    print(f"  score (steps + 50*removed) : {mean:.1f}  (sd {var ** 0.5:.1f})")
    print(f"  best / worst               : {min(scores)} / {max(scores)}")
    return mean
