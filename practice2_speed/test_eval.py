import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker2d import CustomEnvWrapper

def main():
    parser = argparse.ArgumentParser(description="Walker2D Policy Evaluation & Visual Inspection Script")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip file")
    parser.add_argument("--mode", type=str, choices=["flat", "bump_practice", "bump_challenge"], default="flat",
                        help="Environment mode to test: 'flat' (Task 1/2), 'bump_practice' (Task 3 2-bumps), or 'bump_challenge' (3-bumps)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true", help="Enable GUI visualization window")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"[Error] Model file not found at '{model_path}'")
        sys.exit(1)

    # 1. 환경 인스턴스 생성
    bump_practice = (args.mode == "bump_practice")
    bump_challenge = (args.mode == "bump_challenge")
    render_mode = "human" if args.render else None

    raw_env = CustomEnvWrapper(render_mode=render_mode, bump_practice=bump_practice, bump_challenge=bump_challenge)

    # 2. VecNormalize 정규화 파일 자동 탐색 및 로드
    dir_name = os.path.dirname(model_path)
    base_name = os.path.basename(model_path).replace(".zip", "")
    vec_path_1 = os.path.join(dir_name, f"{base_name.replace('walker_model_', 'walker_model_vecnormalize_')}.pkl")
    vec_path_2 = os.path.join(dir_name, f"{base_name}_vecnormalize.pkl")

    vec_path = vec_path_1 if os.path.exists(vec_path_1) else (vec_path_2 if os.path.exists(vec_path_2) else None)

    if vec_path:
        print(f"[Info] Loaded VecNormalize stats from: {vec_path}")
        vec_env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize.load(vec_path, vec_env)
        env.training = False
        env.norm_reward = False
        is_vec_env = True
    else:
        print("[Info] No VecNormalize stats found. Evaluating raw environment.")
        env = raw_env
        is_vec_env = False

    # 3. PPO 모델 로드
    model = PPO.load(model_path)
    print(f"[Info] Successfully loaded PPO model from: {model_path}")
    print(f"[{'=' * 65}]")
    print(f"  Mode: {args.mode.upper()} | Episodes: {args.episodes} | Render: {args.render}")
    print(f"[{'=' * 65}]\n")

    # 4. 에피소드 루프 실행
    x_dists = []
    step_counts = []
    rewards = []
    balance_ratios = []
    bump1_passes = 0
    bump2_passes = 0

    for ep in range(args.episodes):
        if is_vec_env:
            obs = env.reset()
        else:
            obs, _ = env.reset()

        ep_reward = 0.0
        steps = 0
        passed_b1, passed_b2 = False, False
        thigh_vel_r, thigh_vel_l = [], []
        thigh_seps = []

        while True:
            action, _ = model.predict(obs, deterministic=True)

            if is_vec_env:
                obs, reward, done, info = env.step(action)
                step_reward = reward[0]
                terminated = done[0]
                truncated = False
                current_obs = env.get_original_obs()[0]
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                step_reward = reward
                current_obs = obs

            ep_reward += step_reward
            steps += 1

            # 보행 진단 데이터 수집 (Obs indices: 3=thigh_r, 6=thigh_l, 12=thigh_vel_r, 15=thigh_vel_l)
            thigh_seps.append(abs(current_obs[3] - current_obs[6]))
            thigh_vel_r.append(current_obs[12])
            thigh_vel_l.append(current_obs[15])

            if getattr(raw_env, 'passed_bump1', False): passed_b1 = True
            if getattr(raw_env, 'passed_bump2', False): passed_b2 = True

            if args.render:
                raw_env.render()

            if terminated or truncated:
                break

        # 에피소드 결과 집계
        final_x = raw_env.env.unwrapped.data.qpos[0]
        rms_r = float(np.sqrt(np.mean(np.square(thigh_vel_r))))
        rms_l = float(np.sqrt(np.mean(np.square(thigh_vel_l))))
        balance_ratio = min(rms_r, rms_l) / max(rms_r, rms_l, 1e-6)

        x_dists.append(final_x)
        step_counts.append(steps)
        rewards.append(ep_reward)
        balance_ratios.append(balance_ratio)
        if passed_b1: bump1_passes += 1
        if passed_b2: bump2_passes += 1

        bump_info = f" | Bumps: B1={'✓' if passed_b1 else '✗'}, B2={'✓' if passed_b2 else '✗'}" if bump_practice or bump_challenge else ""
        print(f"  [Ep {ep+1:02d}/{args.episodes:02d}] Distance(x): {final_x:6.2f}m | Steps: {steps:4d} | Reward: {ep_reward:7.1f} | Leg Balance: {balance_ratio:.3f}{bump_info}")

    # 5. 요약 리포트 출력
    print(f"\n[{'=' * 65}]")
    print(f"  EVALUATION SUMMARY ({args.mode.upper()})")
    print(f"[{'=' * 65}]")
    print(f"  • Avg Distance (x)  : {np.mean(x_dists):.2f} m ± {np.std(x_dists):.2f}")
    print(f"  • Avg Steps Alive   : {np.mean(step_counts):.1f} steps (Max 1000)")
    print(f"  • Avg Episode Reward: {np.mean(rewards):.1f}")
    print(f"  • Leg Balance Ratio : {np.mean(balance_ratios):.3f} (1.0 = Ideal Gait, <0.4 = Hopping)")

    if bump_practice or bump_challenge:
        print(f"  • Bump 1 Clear Rate : {bump1_passes}/{args.episodes} ({bump1_passes/args.episodes*100:.1f}%)")
        print(f"  • Bump 2 Clear Rate : {bump2_passes}/{args.episodes} ({bump2_passes/args.episodes*100:.1f}%)")
    
    avg_bal = np.mean(balance_ratios)
    if avg_bal > 0.7:
        print("\n  [Gait Status]: ✅ Excellent Bipedal Alternating Gait (자연스러운 교대 보행)")
    elif avg_bal > 0.4:
        print("\n  [Gait Status]: ⚠️ Moderate Asymmetry (약간의 불균형 보행)")
    else:
        print("\n  [Gait Status]: ❌ Hopping Detected (깡총깡총 뛰어가는 상태)")
    print(f"[{'=' * 65}]\n")

    env.close()

if __name__ == "__main__":
    main()
