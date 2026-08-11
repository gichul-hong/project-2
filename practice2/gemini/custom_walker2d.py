import numpy as np
import gymnasium as gym
import os

# Observation (exclude_current_positions_from_observation=False -> 18 dims):
# | 0  | x-coordinate of the torso        (rootx, m)     |
# | 1  | z-coordinate of the torso        (rootz, m)     |
# | 2  | angle of the torso               (rooty, rad)   |
# | 3  | right thigh joint angle          (rad)          |
# | 4  | right leg joint angle            (rad)          |
# | 5  | right foot joint angle           (rad)          |
# | 6  | left thigh joint angle           (rad)          |
# | 7  | left leg joint angle             (rad)          |
# | 8  | left foot joint angle            (rad)          |
# | 9  | torso x velocity                 (m/s)          |
# | 10 | torso z velocity                 (m/s)          |
# | 11 | torso angular velocity           (rad/s)        |
# | 12..14 | right thigh/leg/foot angular velocities     |
# | 15..17 | left  thigh/leg/foot angular velocities     |
#
# Extra features appended by this wrapper (indices 18..21):
# | 18 | bump1_rel_x      = (6.0  - torso_x) / 10.0     |
# | 19 | bump2_rel_x      = (10.0 - torso_x) / 10.0     |
# | 20 | torso_height_dev = torso_z - 1.25               |
# | 21 | next_bump_h_norm = 다음 bump 높이 (정규화)       |
#
# Action: Box(-1, 1, (6,)) torques for
# [thigh, leg, foot, thigh_left, leg_left, foot_left]

BUMP1_X = 6.0
BUMP1_H = 0.2
BUMP2_X = 10.0
BUMP2_H = 0.45
BUMP_APPROACH_DIST = 2.0   # bump 앞 2m부터 접근 보상 활성화
BUMP_PASS_BONUS = 5.0      # 일회성 통과 보너스 (O(1) 스케일)


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        self.is_bump_env = bump_practice or bump_challenge

        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10)

        super().__init__(env)

        self.prev_x = 0.0
        self.passed_bump1 = False
        self.passed_bump2 = False

        # Automatically resize the observation space to match custom_observation.
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.passed_bump1 = False
        self.passed_bump2 = False
        return self.custom_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, action, reward)
        self.prev_x = obs[0]
        terminated = self.custom_terminated(terminated, obs)
        truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def custom_terminated(self, terminated, obs):
        return terminated

    def custom_truncated(self, truncated):
        return truncated

    # ------------------------------------------------------------------ #
    def custom_observation(self, obs):
        """Append task-relevant, non-redundant features.

        bump1_rel_x / bump2_rel_x: 장애물까지 상대 거리
        torso_height_dev: 상체 높이 편차 (점프/웅크림 감지)
        next_bump_h_norm: 다음 미통과 bump의 높이 (정규화)
          → bump마다 높이가 다르므로 policy가 대응을 달리할 수 있게 함
        """
        torso_x = obs[0]
        torso_z = obs[1]

        # 다음 미통과 bump의 정규화 높이 (0.0 = bump 없음)
        if self.is_bump_env:
            if torso_x < BUMP1_X:
                next_bump_h = BUMP1_H / BUMP2_H  # 0.2/0.45 ≈ 0.44
            elif torso_x < BUMP2_X:
                next_bump_h = 1.0                 # 0.45/0.45 = 1.0
            else:
                next_bump_h = 0.0                 # 이미 다 지남
        else:
            next_bump_h = 0.0  # flat ground → bump 정보 불필요

        extra = np.array([
            (BUMP1_X - torso_x) / 10.0,   # bump1_rel_x
            (BUMP2_X - torso_x) / 10.0,   # bump2_rel_x
            torso_z - 1.25,               # torso_height_dev
            next_bump_h,                  # next bump 높이 (정규화)
        ], dtype=np.float64)
        return np.concatenate([obs, extra])

    # ------------------------------------------------------------------ #
    def custom_reward(self, obs, action, original_reward):
        torso_x = obs[0]
        torso_z = obs[1]
        torso_angle = obs[2]
        thigh_r, leg_r, foot_r = obs[3], obs[4], obs[5]
        thigh_l, leg_l, foot_l = obs[6], obs[7], obs[8]
        forward_vel = obs[9]
        z_vel = obs[10]
        torso_ang_vel = obs[11]
        thigh_vel_r = obs[12]
        thigh_vel_l = obs[15]

        # ========== 기본 보상: O(1) scale ========== #
        healthy_reward = 1.0
        ctrl_cost = float(np.sum(np.square(action)))
        # forward_vel 가중치 1.0: hopping은 높은 순간속도를 생산하므로
        # 속도 보상을 낮춰 gait quality 보상에 상대적 여유를 줌
        reward = healthy_reward + 1.0 * forward_vel - 0.001 * ctrl_cost

        # ========== Task 2: 상체 안정성 ========== #
        # 상체 기울기 및 흔들림 억제 (hopping은 torso recoil에 의존)
        reward -= 0.8 * (torso_angle ** 2)
        reward -= 0.2 * (torso_ang_vel ** 2)

        # [핵심 NEW] 수직 바운싱 억제
        # hopping은 반드시 큰 z_vel 진동을 동반.
        # 걷기에서도 약간의 z_vel은 있지만 hopping보다 훨씬 작음
        reward -= 0.3 * (z_vel ** 2)

        # ========== Task 2: 교대 보행 유도 ========== #
        # (a) 다리 벌림(scissoring) 보상 — 교대 보행 시 양 다리가 벌어짐
        alternation = min(abs(thigh_r - thigh_l), 1.0)
        reward += 0.4 * alternation

        # (b) [핵심 NEW] 다리 각속도 부호 반대 보상
        #     걷기: 한 다리는 swing(전진), 다른 다리는 stance(후진)
        #     → thigh 각속도의 부호가 반대여야 함
        #     hopping: 양쪽 동시 이동 → 같은 부호 → 이 보상 못받음
        if thigh_vel_r * thigh_vel_l < 0:  # 부호 반대
            reward += 0.2

        # (c) 다리 속도 균형 — 한쪽만 과도하게 사용 억제
        vel_imbalance = abs(abs(thigh_vel_r) - abs(thigh_vel_l))
        reward -= 0.1 * min(vel_imbalance, 3.0)

        # (d) 무릎/발목 대칭 (절대값 기준)
        sym = (0.5 * abs(abs(leg_r) - abs(leg_l))
               + 0.5 * abs(abs(foot_r) - abs(foot_l)))
        reward -= 0.2 * sym

        # ========== Task 3: bump traversal ========== #
        if self.is_bump_env:
            near_bump1 = (not self.passed_bump1) and (
                BUMP1_X - BUMP_APPROACH_DIST <= torso_x <= BUMP1_X)
            near_bump2 = (not self.passed_bump2) and (
                BUMP2_X - BUMP_APPROACH_DIST <= torso_x <= BUMP2_X)

            if near_bump1 or near_bump2:
                # 상향 속도 장려 (점프 준비)
                reward += 0.3 * max(0.0, z_vel)
                # 다리 들기 장려
                leg_lift = max(0.0, -thigh_r) + max(0.0, -thigh_l)
                reward += 0.1 * min(leg_lift, 2.0)
                # [NEW] 적절한 접근 속도 유도 (너무 빠르면 걸려 넘어짐)
                desired_vel = 1.5
                vel_diff = abs(forward_vel - desired_vel)
                reward -= 0.2 * min(vel_diff, 2.0)

            # 일회성 통과 보너스
            if not self.passed_bump1 and torso_x > BUMP1_X:
                self.passed_bump1 = True
                reward += BUMP_PASS_BONUS
            if not self.passed_bump2 and torso_x > BUMP2_X:
                self.passed_bump2 = True
                reward += BUMP_PASS_BONUS

        return reward


## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
