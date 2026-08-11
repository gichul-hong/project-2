"""
Headless evaluation for the Walker2D custom env.

Reports:
  - avg distance travelled, avg episode reward/length
  - hopping diagnostic: RMS angular velocity of each thigh joint and the
    ratio between them. A ratio close to 1.0 means both legs are used
    roughly equally (healthy alternating gait); a ratio close to 0 (one
    leg's RMS >> the other) indicates single-leg hopping.
  - alternation diagnostic: mean |thigh_r - thigh_l| separation. Larger
    separation means the legs scissor apart (alternating gait); near-zero
    separation with low variance indicates the legs move together (hop).
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker2d import CustomEnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--vecnormalize", type=str, default=None,
                    help="Path to the matching VecNormalize .pkl saved during training. "
                         "Required if the model was trained with observation normalization.")
parser.add_argument("--bump_practice", action="store_true")
parser.add_argument("--bump_challenge", action="store_true")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1000)
args = parser.parse_args()

env = CustomEnvWrapper(render_mode=None, bump_practice=args.bump_practice,
                       bump_challenge=args.bump_challenge)
model = PPO.load(args.model)

vecnorm = None
if args.vecnormalize:
    vecnorm = VecNormalize.load(args.vecnormalize, DummyVecEnv([lambda: env]))
    vecnorm.training = False
    vecnorm.norm_reward = False

for ep in range(args.episodes):
    raw_obs, _ = env.reset()
    thigh_vel_r_hist, thigh_vel_l_hist = [], []
    thigh_sep_hist = []
    total_reward = 0.0
    steps = 0
    reached_end = False
    passed_b1 = passed_b2 = False
    for t in range(args.max_steps):
        model_obs = vecnorm.normalize_obs(raw_obs) if vecnorm is not None else raw_obs
        action, _ = model.predict(model_obs, deterministic=True)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        thigh_vel_r_hist.append(raw_obs[12])
        thigh_vel_l_hist.append(raw_obs[15])
        thigh_sep_hist.append(abs(raw_obs[3] - raw_obs[6]))
        if env.passed_bump1:
            passed_b1 = True
        if env.passed_bump2:
            passed_b2 = True
        if terminated or truncated:
            break
    torso_x = env.env.unwrapped.data.qpos[0]
    if torso_x >= 40.0:
        reached_end = True

    rms_r = float(np.sqrt(np.mean(np.square(thigh_vel_r_hist))))
    rms_l = float(np.sqrt(np.mean(np.square(thigh_vel_l_hist))))
    ratio = min(rms_r, rms_l) / max(rms_r, rms_l, 1e-6)
    mean_sep = float(np.mean(thigh_sep_hist))

    print(f"[ep {ep}] steps={steps} reward={total_reward:.1f} "
         f"torso_x={torso_x:.2f} reached_end={reached_end} "
         f"passed_bump1={passed_b1} passed_bump2={passed_b2}")
    print(f"         thigh_vel_rms: R={rms_r:.3f} L={rms_l:.3f} "
         f"balance_ratio={ratio:.3f} (1.0=balanced, ~0=hopping)")
    print(f"         mean |thigh_r - thigh_l| separation = {mean_sep:.3f} "
         f"(higher => more scissoring/alternation)")
