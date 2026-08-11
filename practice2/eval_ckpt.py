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
vec_path = args.model.replace(".zip", "").replace("walker_model_", "walker_model_vecnormalize_") + ".pkl"
env = VecNormalize.load(vec_path, DummyVecEnv([lambda: raw]))
env.training = False
env.norm_reward = False
model = PPO.load(args.model, env=env)

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
    print(f"ep{ep}: len={steps} max_x={last_max_x:.2f} passed={last_passed}")

lens = [r[0] for r in results]
b2 = sum(r[2][1] for r in results)
b3 = sum(r[2][2] for r in results)
print(f"\nmean_len={np.mean(lens):.0f}  bump2 pass {b2}/{len(results)}  bump3 pass {b3}/{len(results)}  mean_max_x={np.mean([r[1] for r in results]):.2f}")
