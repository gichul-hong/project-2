"""
Headless evaluation for Walker2D — hopping 진단 + bump 통과율 측정.

Reports:
  - avg distance, avg episode reward/length
  - hopping diagnostic: thigh 각속도 RMS 비율 (1.0 = balanced, ~0 = hopping)
  - alternation diagnostic: mean |thigh_r - thigh_l| separation
  - bump 통과 여부 (bump 환경인 경우)
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker2d import CustomEnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--vecnormalize", type=str, default=None,
                    help="Path to the matching VecNormalize .pkl")
parser.add_argument("--bump_practice", action="store_true")
parser.add_argument("--bump_challenge", action="store_true")
parser.add_argument("--episodes", type=int, default=10)
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

# 전체 통계
all_x, all_steps, all_rewards = [], [], []
all_b1, all_b2 = 0, 0
all_balance = []

print(f"\n{'='*70}")
print(f"  Evaluating: {args.model}")
print(f"  Episodes: {args.episodes} | Bump: {args.bump_practice or args.bump_challenge}")
print(f"{'='*70}\n")

for ep in range(args.episodes):
    raw_obs, _ = env.reset()
    thigh_vel_r_hist, thigh_vel_l_hist = [], []
    thigh_sep_hist = []
    total_reward = 0.0
    steps = 0
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
    rms_r = float(np.sqrt(np.mean(np.square(thigh_vel_r_hist))))
    rms_l = float(np.sqrt(np.mean(np.square(thigh_vel_l_hist))))
    ratio = min(rms_r, rms_l) / max(rms_r, rms_l, 1e-6)
    mean_sep = float(np.mean(thigh_sep_hist))

    all_x.append(torso_x)
    all_steps.append(steps)
    all_rewards.append(total_reward)
    all_balance.append(ratio)
    if passed_b1: all_b1 += 1
    if passed_b2: all_b2 += 1

    bump_str = f" bump1={'✓' if passed_b1 else '✗'} bump2={'✓' if passed_b2 else '✗'}" if (args.bump_practice or args.bump_challenge) else ""
    print(f"  [ep {ep:2d}] x={torso_x:6.1f}  steps={steps:4d}  reward={total_reward:7.1f}"
          f"  balance={ratio:.3f}  sep={mean_sep:.3f}{bump_str}")

# 요약
print(f"\n{'─'*70}")
print(f"  SUMMARY ({args.episodes} episodes)")
print(f"{'─'*70}")
print(f"  avg_x:       {np.mean(all_x):.1f} ± {np.std(all_x):.1f}")
print(f"  avg_steps:   {np.mean(all_steps):.0f} ± {np.std(all_steps):.0f}")
print(f"  avg_reward:  {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
print(f"  avg_balance: {np.mean(all_balance):.3f}  (1.0=balanced, ~0=hopping)")
if args.bump_practice or args.bump_challenge:
    print(f"  bump1 pass:  {all_b1}/{args.episodes}")
    print(f"  bump2 pass:  {all_b2}/{args.episodes}")

# 판정
avg_bal = np.mean(all_balance)
if avg_bal > 0.7:
    print(f"\n  ✅ 양발 균형 양호 (balance={avg_bal:.3f} > 0.7)")
elif avg_bal > 0.4:
    print(f"\n  ⚠️  양발 불균형 경향 (balance={avg_bal:.3f})")
else:
    print(f"\n  ❌ 심각한 hopping (balance={avg_bal:.3f} < 0.4)")
print()
