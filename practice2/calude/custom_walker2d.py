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
# Extra features appended by this wrapper (indices 18..20):
# | 18 | bump1_rel_x     = (6.0  - torso_x) / 10.0       |
# | 19 | bump2_rel_x     = (10.0 - torso_x) / 10.0       |
# | 20 | torso_height_dev = torso_z - 1.25               |
#
# Action: Box(-1, 1, (6,)) torques for
# [thigh, leg, foot, thigh_left, leg_left, foot_left]

BUMP1_X = 6.0
BUMP2_X = 10.0
BUMP_APPROACH_DIST = 1.0  # start "prepare to jump" shaping this far before a bump


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
        # Keep the default healthy-range termination.
        return terminated

    def custom_truncated(self, truncated):
        return truncated

    # ------------------------------------------------------------------ #
    def custom_observation(self, obs):
        """Append task-relevant, non-redundant features.

        bump1_rel_x / bump2_rel_x tell the agent where the obstacles are;
        torso_height_dev makes it easy to sense crouch/jump state.
        (These are computed for every task so a single policy/architecture
        works everywhere; on flat ground they are simply smooth functions
        of x and z that the network can ignore.)
        """
        torso_x = obs[0]
        torso_z = obs[1]
        extra = np.array([
            (BUMP1_X - torso_x) / 10.0,   # bump1_rel_x
            (BUMP2_X - torso_x) / 10.0,   # bump2_rel_x
            torso_z - 1.25,               # torso_height_dev
        ], dtype=np.float64)
        return np.concatenate([obs, extra])

    # ------------------------------------------------------------------ #
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

        # ---------- base reward: O(1) scale ----------
        healthy_reward = 1.0
        ctrl_cost = float(np.sum(np.square(action)))
        # NOTE: deviating slightly from the original 1.5x spec -> 1.2x.
        # At 1.5x, raw forward speed dominated the gait/symmetry terms and
        # the policy converged to single-leg hopping (high forward_vel,
        # cheap to produce). 1.2x still strongly rewards progress (Task 1)
        # while giving the alternation/symmetry/posture terms enough
        # relative weight to shape a two-legged gait (Task 2).
        reward = healthy_reward + 1.2 * forward_vel - 0.001 * ctrl_cost

        # ---------- Task 2: upright torso (anti-hopping: strong pitch damping) ----------
        # Hopping relies on torso recoil; heavier pitch/pitch-rate penalties
        # make single-leg hopping unprofitable.
        reward -= 0.8 * (torso_angle ** 2)
        reward -= 0.2 * (torso_ang_vel ** 2)

        # ---------- Task 2: gait symmetry & leg alternation ----------
        # NOTE: thigh joints are limited to [-150, 0] deg, so BOTH thigh
        # angles are always <= 0 (same sign). Sign-based crossing checks
        # (thigh_r * thigh_l < 0) never fire in this env. Instead we use:
        #
        # (a) Alternation bonus: reward thigh separation |thigh_r - thigh_l|.
        #     During alternating gait the legs are scissored apart; during
        #     hopping they stay nearly parallel (separation ~ 0).
        alternation = min(abs(thigh_r - thigh_l), 1.0)
        reward += 0.3 * alternation

        # (b) Leg activity balance: hopping drives one leg hard while the
        #     other stays idle. Penalize the mismatch in thigh angular
        #     speed magnitudes (clipped to stay O(1)).
        vel_imbalance = abs(abs(thigh_vel_r) - abs(thigh_vel_l))
        reward -= 0.1 * min(vel_imbalance, 3.0)

        # (c) Knee/ankle magnitude symmetry across legs (smooth L1).
        sym = (0.5 * abs(abs(leg_r) - abs(leg_l))
               + 0.5 * abs(abs(foot_r) - abs(foot_l)))
        reward -= 0.2 * sym

        # ---------- Task 3: bump traversal ----------
        if self.is_bump_env:
            near_bump1 = (not self.passed_bump1) and (BUMP1_X - BUMP_APPROACH_DIST <= torso_x <= BUMP1_X)
            near_bump2 = (not self.passed_bump2) and (BUMP2_X - BUMP_APPROACH_DIST <= torso_x <= BUMP2_X)
            if near_bump1 or near_bump2:
                # encourage upward velocity (jump prep) and leg lift
                reward += 0.3 * max(0.0, z_vel)
                leg_lift = max(0.0, -thigh_r) + max(0.0, -thigh_l)
                reward += 0.1 * min(leg_lift, 2.0)

            # one-time pass bonuses
            if not self.passed_bump1 and torso_x > BUMP1_X:
                self.passed_bump1 = True
                reward += 2.0
            if not self.passed_bump2 and torso_x > BUMP2_X:
                self.passed_bump2 = True
                reward += 2.0

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
