import numpy as np
import gymnasium as gym
import os

BUMP_PRACTICE_CONFIG = [
    dict(name="bump1", x=6.0, size=(0.3, 2.0, 0.2)),
    dict(name="bump2", x=10.0, size=(0.6, 2.0, 0.45)),
]
BUMP_CHALLENGE_CONFIG = [
    dict(name="bump1", x=6.0, size=(0.3, 2.0, 0.3)),
    dict(name="bump2", x=10.0, size=(0.7, 2.0, 0.6)),
    dict(name="bump3", x=15.0, size=(0.5, 2.0, 0.5)),
]

BUMP_APPROACH_DIST = 2.0


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        self.is_bump_env = bump_practice or bump_challenge

        if bump_challenge:
            self.bumps = BUMP_CHALLENGE_CONFIG
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))
        elif bump_practice:
            self.bumps = BUMP_PRACTICE_CONFIG
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0))
        else:
            self.bumps = []
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10)

        super().__init__(env)

        self.prev_x = 0.0
        self.passed_bumps = []

        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.passed_bumps = [False] * len(self.bumps)
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
        return terminated

    def custom_truncated(self, truncated):
        return truncated

    def custom_observation(self, obs):
        torso_x = obs[0]
        torso_z = obs[1]
        bump_features = []
        for b in self.bumps:
            rel_x = (b["x"] - torso_x) / 20.0
            bump_features.append(rel_x)
        height_dev = torso_z - 1.25
        extra = np.array(bump_features + [height_dev], dtype=np.float64)
        return np.concatenate([obs, extra])

    def custom_reward(self, obs, action, original_reward):
        torso_x = obs[0]
        torso_angle = obs[2]
        thigh_r, leg_r, foot_r = obs[3], obs[4], obs[5]
        thigh_l, leg_l, foot_l = obs[6], obs[7], obs[8]
        forward_vel = obs[9]
        z_vel = obs[10]
        torso_ang_vel = obs[11]
        thigh_vel_r = obs[12]
        thigh_vel_l = obs[15]

        healthy_reward = 1.0
        ctrl_cost = float(np.sum(np.square(action)))
        reward = healthy_reward + 1.0 * forward_vel - 0.001 * ctrl_cost

        reward -= 0.8 * (torso_angle ** 2)
        reward -= 0.2 * (torso_ang_vel ** 2)

        reward -= 0.5 * (z_vel ** 2)

        thigh_vel_r_sign = np.sign(thigh_vel_r)
        thigh_vel_l_sign = np.sign(thigh_vel_l)
        if thigh_vel_r_sign * thigh_vel_l_sign < 0:
            reward += 0.5

        alternation = min(abs(thigh_r - thigh_l), 1.0)
        reward += 0.3 * alternation

        vel_imbalance = abs(abs(thigh_vel_r) - abs(thigh_vel_l))
        reward -= 0.1 * min(vel_imbalance, 3.0)

        sym = (0.5 * abs(abs(leg_r) - abs(leg_l))
               + 0.5 * abs(abs(foot_r) - abs(foot_l)))
        reward -= 0.2 * sym

        if self.is_bump_env:
            near_any_bump = False
            for i, b in enumerate(self.bumps):
                if self.passed_bumps[i]:
                    continue
                near = (b["x"] - BUMP_APPROACH_DIST <= torso_x <= b["x"])
                if near:
                    near_any_bump = True
                    reward += 0.3 * max(0.0, z_vel)
                    leg_lift = max(0.0, -thigh_r) + max(0.0, -thigh_l)
                    reward += 0.1 * min(leg_lift, 2.0)
                    reward -= 0.3 * max(0.0, forward_vel - 1.0)
                if torso_x > b["x"]:
                    reward += 50.0
                    self.passed_bumps[i] = True
            if not near_any_bump:
                reward += 2.0 * max(0.0, forward_vel)

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
