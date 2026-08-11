import numpy as np
import gymnasium as gym
import os

# The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
# | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
# | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
# | 0   | x-coordinate of the torso                          | -Inf | Inf | rootz                            | slide | position (m)             |
# | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
# | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
# | 3   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
# | 4   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
# | 5   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
# | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
# | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
# | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
# | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
# | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
# | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
# | 12  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
# | 13  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
# | 14  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
# | 15  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
# | 16  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
# | 17  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

# The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
# | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
# |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
# | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
# | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
# | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
# | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
# | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10)
        
        super().__init__(env)
        
        self.prev_x = 0.0
        self.passed_bump1 = False
        self.passed_bump2 = False
        
        ## change observation space according to the new observation
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.passed_bump1 = False
        self.passed_bump2 = False
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, reward)
        self.prev_x = obs[0]
        custom_terminated = self.custom_terminated(terminated, obs)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_terminated(self, terminated, obs):
        return terminated
    
    def custom_truncated(self, truncated):
        return truncated

    def custom_observation(self, obs):
        torso_x = obs[0]
        torso_z = obs[1]
        extra = np.array([
            (6.0 - torso_x) / 10.0,   # bump1_rel_x
            (10.0 - torso_x) / 10.0,  # bump2_rel_x
            torso_z - 1.25,           # torso_height_dev
        ], dtype=np.float64)
        return np.concatenate([obs, extra])

    def custom_reward(self, obs, original_reward):
        reward = original_reward
        torso_angle = obs[2]
        torso_ang_vel = obs[11]
        z_vel = obs[10]
        thigh_r, leg_r, foot_r = obs[3], obs[4], obs[5]
        thigh_l, leg_l, foot_l = obs[6], obs[7], obs[8]
        thigh_vel_r, thigh_vel_l = obs[12], obs[15]

        reward -= 0.8 * (torso_angle ** 2)
        reward -= 0.2 * (torso_ang_vel ** 2)
        alternation = min(abs(thigh_r - thigh_l), 1.0)
        reward += 0.3 * alternation
        vel_imbalance = abs(abs(thigh_vel_r) - abs(thigh_vel_l))
        reward -= 0.1 * min(vel_imbalance, 3.0)
        sym = (0.5 * abs(abs(leg_r) - abs(leg_l)) + 0.5 * abs(abs(foot_r) - abs(foot_l)))
        reward -= 0.2 * sym

        if self.bump_practice:
            bump1_x = 6.0
            bump2_x = 10.0
            near_bump1 = (not self.passed_bump1) and (bump1_x - 1.0 <= obs[0] <= bump1_x)
            near_bump2 = (not self.passed_bump2) and (bump2_x - 1.0 <= obs[0] <= bump2_x)
            if near_bump1 or near_bump2:
                reward += 0.3 * max(0.0, z_vel)
                leg_lift = max(0.0, -thigh_r) + max(0.0, -thigh_l)
                reward += 0.1 * min(leg_lift, 2.0)
            if not self.passed_bump1 and obs[0] > bump1_x:
                reward += 50.0
                self.passed_bump1 = True
            if not self.passed_bump2 and obs[0] > bump2_x:
                reward += 50.0
                self.passed_bump2 = True
        return reward

## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()
