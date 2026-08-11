import numpy as np
import gymnasium as gym
import os
from collections import deque

# 위상 시프트 대칭 보상의 반주기 (스텝 단위, dt=0.02s -> 15스텝 = 0.3s)
GAIT_HALF_PERIOD = 15

BUMP_CONFIGS = {
    "bump_practice": [
        dict(name="bump1", x=6.0, width=0.3, height=0.2),
        dict(name="bump2", x=10.0, width=0.6, height=0.45),
    ],
    "bump_challenge": [
        dict(name="bump1", x=6.0, width=0.3, height=0.3),
        dict(name="bump2", x=10.0, width=0.7, height=0.6),
        dict(name="bump3", x=15.0, width=0.5, height=0.5),
    ],
}


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False, xml_file=None):
        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        self.is_bump_env = bump_practice or bump_challenge
        self.bumps = []

        if bump_challenge:
            self.bumps = BUMP_CONFIGS["bump_challenge"]
            # xml_file로 커리큘럼 단계(낮은 범프) XML을 지정할 수 있음.
            # 범프 위치/개수는 동일해야 함 (관측 차원 유지)
            xml = xml_file or (os.getcwd() + "/asset/custom_walker2d_bumps.xml")
            env = gym.make(
                "Walker2d-v5",
                xml_file=xml,
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.9, 10.0))  # 무릎 보행(torso z~0.75-0.85) 차단
        elif bump_practice:
            self.bumps = BUMP_CONFIGS["bump_practice"]
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.9, 10.0))  # 무릎 보행(torso z~0.75-0.85) 차단
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10)

        super().__init__(env)
        self.prev_x = 0.0
        self.passed_bumps = []
        # 마지막 범프 너머에도 전진 유인을 유지하기 위한 1회성 마일스톤
        if self.bumps:
            last_x = max(b["x"] for b in self.bumps)
            self.goal_milestones = [(last_x + 3.0, 25.0), (last_x + 5.0, 50.0)]
        else:
            self.goal_milestones = []

        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)

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
        # 정지(standing) 국소최적 차단: 일정 시간 전진이 없으면 에피소드 종료.
        # terminated로 끊어야 가치 부트스트랩이 없어 "서서 버티기"의 기대가치가 낮아짐.
        if obs[0] > self.max_x + 0.05:
            self.max_x = obs[0]
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        if self.stall_steps >= 300:  # dt=0.02s -> 6초간 진전 없으면 종료 (범프 등반 시도 시간 확보)
            return True
        return terminated

    def custom_truncated(self, truncated):
        return truncated

    def custom_observation(self, obs):
        torso_x = obs[0]
        torso_z = obs[1]
        extra = []
        for b in self.bumps:
            extra.append((b["x"] - torso_x) / 20.0)
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
        # z_vel 페널티는 "아직 안 넘은 범프 근처"에서는 끔: 범프를 넘으려면
        # 큰 상승 속도가 필요한데 이걸 벌점 주면 발목 통통 점프로 수렴함
        near_unpassed_bump = any(
            not self.passed_bumps[i] and b["x"] - 2.5 <= obs[0] <= b["x"] + 1.0
            for i, b in enumerate(self.bumps))
        if not near_unpassed_bump:
            reward -= 0.3 * (z_vel ** 2)
        else:
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
                # 명시적 점프 유도 (1회성이라 제자리 점프 농사 불가):
                # 접근 구간에서 (1) 도약 임펄스, (2) 몸 띄우기를 단계적으로 보상
                if not self.passed_bumps[i] and b["x"] - 2.0 <= obs[0] <= b["x"] + 0.7:
                    if not self.jump_bonus[i] and z_vel > 1.5:
                        reward += 10.0  # 도약 임펄스 (상승 속도 1.5 m/s)
                        self.jump_bonus[i] = True
                    if not self.height_bonus[i] and torso_z > 1.55:
                        reward += 15.0  # 체공/등반 높이 (평지 보행으론 불가)
                        self.height_bonus[i] = True
                # 범프 전면 도달 마일스톤(1회성 +15): "거의 넘음"과 "접근 못함"
                # 사이에 보상 기울기를 만들어 등반 시도를 유도.
                # x 기준 단조 마일스톤이라 진동으로 착취 불가.
                if not self.face_bonus[i] and obs[0] > b["x"] - 0.4:
                    reward += 15.0
                    self.face_bonus[i] = True
                if not self.passed_bumps[i] and obs[0] > b["x"]:
                    reward += 50.0
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
