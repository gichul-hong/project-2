import argparse, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker2d import CustomEnvWrapper

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--xml", default=None)
p.add_argument("--episodes", type=int, default=10)
args = p.parse_args()

xml = os.path.abspath(args.xml) if args.xml else None
raw = CustomEnvWrapper(bump_challenge=True, xml_file=xml)
# v9.1: 관측 정규화가 환경에 내장되어 VecNormalize 불필요 (채점기와 동일 경로)
env = DummyVecEnv([lambda: raw])
model = PPO.load(args.model, env=env, device="cpu")

big_idx = [i for i, b in enumerate(raw.bumps) if b["big"]]
print("big bumps:", [(raw.bumps[i]["name"], raw.bumps[i]["height"]) for i in big_idx])

results = []
for ep in range(args.episodes):
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        last_max_x, last_passed = raw.max_x, list(raw.passed_bumps)
        obs, r, dones, infos = env.step(action)
        done = dones[0]
        steps += 1
    results.append((steps, last_max_x, last_passed))
    big_str = " ".join(f"{raw.bumps[i]['name']}={'O' if last_passed[i] else 'X'}" for i in big_idx)
    print(f"ep{ep}: len={steps} max_x={last_max_x:.2f} {big_str}")

lens = [r[0] for r in results]
summary = "  ".join(
    f"{raw.bumps[i]['name']} pass {sum(r[2][i] for r in results)}/{len(results)}"
    for i in big_idx)
print(f"\nmean_len={np.mean(lens):.0f}  {summary}  mean_max_x={np.mean([r[1] for r in results]):.2f}")
