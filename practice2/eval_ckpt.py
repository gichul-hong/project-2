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
if os.path.exists(vec_path):
    env = VecNormalize.load(vec_path, DummyVecEnv([lambda: raw]))
    env.training = False
else:
    # 수술본 등 통계가 없는 경우: 롤아웃 중 통계를 갱신하며 평가 (참고용)
    print(f"[warn] {vec_path} 없음 -> 새 VecNormalize (통계 미학습, 성능 저평가 가능)")
    env = VecNormalize(DummyVecEnv([lambda: raw]), norm_obs=True, norm_reward=False, clip_obs=10.0)
env.norm_reward = False
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
