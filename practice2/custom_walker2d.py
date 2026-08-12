import numpy as np
import gymnasium as gym
import os
from collections import deque

# 위상 시프트 대칭 보상의 반주기 (스텝 단위, dt=0.02s -> 15스텝 = 0.3s)
GAIT_HALF_PERIOD = 15

# v9: 범프 정보는 XML에서 자동 파싱 (맵 교체 시 코드 수정 불필요)
# 관측/보상에서 "실질 장애물"로 취급하는 최소 높이 (이하는 잔범프)
BIG_BUMP_HEIGHT = 0.2
# 관측에 포함하는 "다음 범프" 개수 (맵 불변 관측)
OBS_BUMP_K = 2
# 종료 조건용 torso 높이 하한 (무릎 보행 차단). env healthy_z_range는 원본값 유지.
MIN_TORSO_Z = 0.9


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False, xml_file=None):
        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        self.is_bump_env = bump_practice or bump_challenge
        self.bumps = []

        if bump_challenge:
            # xml_file로 커리큘럼 단계(낮은 범프) XML을 지정할 수 있음.
            # v9: 범프 개수/위치/폭이 달라도 관측 차원은 고정(K개 슬롯)
            xml = xml_file or (os.getcwd() + "/asset/custom_walker2d_bumps.xml")
            env = gym.make(
                "Walker2d-v5",
                xml_file=xml,
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))  # 스켈레톤 원본값 유지 (README 제약)
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))  # 스켈레톤 원본값 유지 (README 제약)
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10)

        super().__init__(env)
        self.prev_x = 0.0
        self.passed_bumps = []

        # --- Phase 1: XML geom에서 범프 자동 파싱 (reset보다 먼저) ---
        if self.is_bump_env:
            self.bumps = self._parse_bumps()

        # 마지막 범프 너머에도 전진 유인을 유지하기 위한 1회성 마일스톤
        if self.bumps:
            last_x = max(b["back_x"] for b in self.bumps)
            self.goal_milestones = [(last_x + 3.0, 25.0), (last_x + 5.0, 50.0)]
        else:
            self.goal_milestones = []

        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)

    def _parse_bumps(self):
        """XML의 name이 'bump'로 시작하는 box geom을 읽어 범프 목록 생성."""
        model = self.env.unwrapped.model
        bumps = []
        for i in range(model.ngeom):
            name = model.geom(i).name
            if not name.startswith("bump"):
                continue
            pos = model.geom(i).pos     # [x, y, z]
            size = model.geom(i).size   # [half_w, half_d, half_h]
            x = float(pos[0])
            half_width = float(size[0])
            height = float(pos[2]) + float(size[2])  # 상단 높이 (pos_z=0 기준)
            bumps.append(dict(name=name, x=x, half_width=half_width,
                              height=height,
                              front_x=x - half_width, back_x=x + half_width,
                              big=height >= BIG_BUMP_HEIGHT))
        bumps.sort(key=lambda b: b["x"])

        # 계단 판정: 직전 범프와 간격 < 1.0m이고 직전이 더 낮으면
        # 그 상단에서 도약하므로 기준 높이를 직전 상단으로 잡음
        for i, b in enumerate(bumps):
            base = 0.0
            if i > 0:
                prev = bumps[i - 1]
                if (b["front_x"] - prev["back_x"] < 1.0
                        and prev["height"] < b["height"]):
                    base = prev["height"]
            b["base_height"] = base
            b["rel_height"] = b["height"] - base
        return bumps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.passed_bumps = [False] * len(self.bumps)
        self.face_bonus = [False] * len(self.bumps)
        self.jump_bonus = [False] * len(self.bumps)
        self.height_bonus = [False] * len(self.bumps)
        self.max_x = obs[0]
        self.stall_steps = 0
        self.right_leg_hist = deque(maxlen=GAIT_HALF_PERIOD)
        self.goal_bonus = [False] * len(self.goal_milestones)
        return self.custom_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, action, reward)
        self.prev_x = obs[0]
        terminated = self.custom_terminated(terminated, obs)
        truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, terminated, truncated, info

    def custom_terminated(self, terminated, obs):
        # 무릎 보행(torso z~0.75-0.85) 차단: 기존에는 env의 healthy_z_range로 막았으나
        # README 제약("__init__의 환경 파라미터 수정 금지")에 맞춰 종료 조건으로 이전.
        # env는 원본값 (0.5, 10.0)을 사용하고, 여기서 0.9 하한을 적용한다.
        if obs[1] < MIN_TORSO_Z:
            return True
        # 정지(standing) 국소최적 차단: 일정 시간 전진이 없으면 에피소드 종료.
        # terminated로 끊어야 가치 부트스트랩이 없어 "서서 버티기"의 기대가치가 낮아짐.
        if obs[0] > self.max_x + 0.05:
            self.max_x = obs[0]
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        if self.stall_steps >= 500:  # dt=0.02s -> 10초간 진전 없으면 종료 (계단 등반 시도 시간)
            return True
        return terminated

    def custom_truncated(self, truncated):
        return truncated

    def custom_observation(self, obs):
        """v9: 맵 불변 관측.

        base 18차원 + (다음 K개 범프 × [거리/10, 높이, 반폭]) + (torso_z - 1.25)
        미도달 = torso_x < bump.back_x. K개보다 적으면 [2.0, 0.0, 0.0] 패딩.
        """
        torso_x = obs[0]
        torso_z = obs[1]
        upcoming = [b for b in self.bumps if torso_x < b["back_x"]][:OBS_BUMP_K]
        extra = []
        for b in upcoming:
            extra.extend([(b["x"] - torso_x) / 10.0, b["height"], b["half_width"]])
        for _ in range(OBS_BUMP_K - len(upcoming)):
            extra.extend([2.0, 0.0, 0.0])
        extra.append(torso_z - 1.25)
        return np.concatenate([obs, np.array(extra, dtype=np.float64)])

    def custom_reward(self, obs, action, original_reward):
        # 부호 있는 전진 보상: 후진은 그대로 벌점이 되므로
        # 앞뒤 진동(제자리 호핑)으로 보상을 착취할 수 없음
        dx = obs[0] - self.prev_x  # frame_skip=10 기준 1 step 이동 거리
        torso_z = obs[1]
        z_vel = obs[10]
        torso_angle = obs[2]
        torso_ang_vel = obs[11]
        thigh_vel_r = obs[12]
        thigh_vel_l = obs[15]

        reward = 1.0  # healthy

        # 전진 보상 (후진 시 페널티). dt=0.02s이므로 1 m/s 보행 시 dx=0.02
        # -> 60*dx = 1.2/step으로 healthy(1.0)보다 크도록 스케일 조정.
        # 높이 게이트: 몸을 낮추고(무릎 보행) 전진하면 전진 보상을 못 받음.
        # torso_z >= 1.1이면 1.0, 0.9 이하면 0 (0.9 미만은 어차피 종료됨)
        height_factor = np.clip((torso_z - 0.9) / 0.2, 0.0, 1.0)
        reward += 60.0 * dx * height_factor

        # 낮은 자세 페널티: 서있는 높이(~1.25)보다 낮게 웅크리면 벌점
        reward -= 1.0 * max(0.0, 1.1 - torso_z)

        # 자세 안정화 페널티 (정상 보행 시 스텝 보상이 양수를 유지하도록 약하게)
        # z_vel 페널티는 "아직 안 넘은 큰 범프 근처"에서는 끔: 범프를 넘으려면
        # 큰 상승 속도가 필요한데 이걸 벌점 주면 발목 통통 점프로 수렴함
        # (v9: 잔범프는 대상 제외 - 잔범프에서 대칭 보상이 꺼지면 안 됨)
        near_unpassed_bump = any(
            b["big"] and not self.passed_bumps[i]
            and b["front_x"] - 2.5 <= obs[0] <= b["back_x"] + 1.0
            for i, b in enumerate(self.bumps))
        # 착지 구간: 높은 범프(h>=0.4) 통과 직후 2m는 z_vel 벌점 해제
        in_landing_zone = any(
            b["height"] >= 0.4 and self.passed_bumps[i]
            and b["back_x"] <= obs[0] <= b["back_x"] + 2.0
            for i, b in enumerate(self.bumps))
        if not near_unpassed_bump and not in_landing_zone:
            reward -= 0.3 * (z_vel ** 2)
        elif near_unpassed_bump:
            # 범프 근처에서는 대칭 보상이 꺼지므로 한발 점프로 치우치기 쉬움.
            # 양발목 동시 push-off 유도: 두 발목 토크 중 "작은 쪽"에 비례 보상
            # -> 한쪽만 쓰면 0, 양쪽을 같이 써야 최대 +0.2
            reward += 0.2 * min(abs(float(action[2])), abs(float(action[5])))
        reward -= 0.3 * (torso_angle ** 2)
        reward -= 0.05 * (torso_ang_vel ** 2)

        # 교대 보행 보상: 양쪽 허벅지가 "반대 방향으로 움직일 때"만 보상.
        # 정적 자세나 두 발 동시 호핑으로는 충족 불가.
        swing_product = -thigh_vel_r * thigh_vel_l  # 반대 방향 스윙이면 양수
        alternation = np.tanh(max(0.0, swing_product))
        reward += 0.3 * alternation

        # 위상 시프트 대칭 보상 (강의 Tip: Solution #2):
        # 오른다리 상태(각도 + 스케일된 각속도)를 반주기 전에 저장해두고,
        # 왼다리가 그 상태를 재현하면 보상 -> 주기적 교대 gait 유도.
        # 미통과 범프 근처에서는 비대칭 등반 동작이 필요하므로 끔.
        # 전진 속도 게이트: 정지 자세도 자명하게 대칭이므로 (left_now = right_past)
        # 걷고 있을 때만 지급 -> "서서 대칭 보너스 수확" exploit 차단.
        right_state = np.array([obs[3], obs[4], obs[5],
                                0.1 * obs[12], 0.1 * obs[13], 0.1 * obs[14]])
        if len(self.right_leg_hist) == GAIT_HALF_PERIOD and not near_unpassed_bump:
            left_state = np.array([obs[6], obs[7], obs[8],
                                   0.1 * obs[15], 0.1 * obs[16], 0.1 * obs[17]])
            l2 = float(np.sum((left_state - self.right_leg_hist[0]) ** 2))
            speed_gate = np.clip(obs[9], 0.0, 1.0)  # 1 m/s 이상에서 만점
            reward += 0.5 * np.exp(-2.0 * l2) * speed_gate
        self.right_leg_hist.append(right_state)

        if self.is_bump_env:
            for i, b in enumerate(self.bumps):
                # v9: 점프/높이 보너스는 큰 범프(h>=0.2)에만. 잔범프 헛점프 방지.
                # 계단(bump7->bump8)은 직전 상단(base_height) 기준 상대값 사용.
                if b["big"] and not self.passed_bumps[i] \
                        and b["front_x"] - 2.0 <= obs[0] <= b["back_x"]:
                    if not self.jump_bonus[i] and \
                            z_vel > min(1.5, 3.0 * b["rel_height"]):
                        reward += 10.0  # 도약 임펄스 (높이 비례, 최대 1.5 m/s)
                        self.jump_bonus[i] = True
                    if not self.height_bonus[i] and torso_z > b["height"] + 1.05:
                        reward += 15.0  # 체공/등반 높이 (평지 보행으론 불가)
                        self.height_bonus[i] = True
                # 범프 전면 도달 마일스톤(1회성 +15, 큰 범프만): "거의 넘음"과
                # "접근 못함" 사이에 보상 기울기를 만들어 등반 시도를 유도.
                # x 기준 단조 마일스톤이라 진동으로 착취 불가.
                if b["big"] and not self.face_bonus[i] \
                        and obs[0] > b["front_x"] - 0.4:
                    reward += 15.0
                    self.face_bonus[i] = True
                # 통과 판정은 범프 뒷면 기준. 잔범프는 보상 인플레 방지로 +10
                if not self.passed_bumps[i] and obs[0] > b["back_x"]:
                    reward += 50.0 if b["big"] else 10.0
                    self.passed_bumps[i] = True

            # 마지막 범프 너머 마일스톤: "범프3 통과 후 멈춤" 방지용 전진 유인
            for i, (mx, bonus) in enumerate(self.goal_milestones):
                if not self.goal_bonus[i] and obs[0] > mx:
                    reward += bonus
                    self.goal_bonus[i] = True

        return reward


if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
