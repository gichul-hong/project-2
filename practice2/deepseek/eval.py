import sys, os
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from custom_walker2d import CustomEnvWrapper
import numpy as np

def evaluate(model_path, bump=False, nep=10):
    model = PPO.load(model_path)
    xs, steps_list, surv = [], [], 0
    bump_passed = [0, 0]
    for ep in range(nep):
        env = CustomEnvWrapper(render_mode='rgb_array', bump_practice=bump)
        obs, _ = env.reset()
        s = 0
        for s in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        xs.append(obs[0])
        steps_list.append(s)
        if s >= 1000:
            surv += 1
        if bump:
            bump_passed[0] += 1 if env.passed_bump1 else 0
            bump_passed[1] += 1 if env.passed_bump2 else 0
    if bump:
        print(f'  avg_x={np.mean(xs):.1f} avg_steps={np.mean(steps_list):.0f} survived={surv}/{nep} bump1={bump_passed[0]}/{nep} bump2={bump_passed[1]}/{nep}')
    else:
        print(f'  avg_x={np.mean(xs):.1f} avg_steps={np.mean(steps_list):.0f} survived={surv}/{nep}')
    return np.mean(xs), np.mean(steps_list), surv

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--bump', action='store_true')
    parser.add_argument('--nep', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.model, bump=args.bump, nep=args.nep)