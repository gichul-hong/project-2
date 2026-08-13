from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_walker2d import CustomEnvWrapper
import torch

N_ENVS = 5

def make_env(bump_practice=False, bump_challenge=False, xml_file=None):
    def _init():
        return CustomEnvWrapper(render_mode=None, bump_practice=bump_practice, bump_challenge=bump_challenge, xml_file=xml_file)
    return _init

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    log_std_init=-1.0,
)

import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--bump_practice", action="store_true")
parser.add_argument("--bump_challenge", action="store_true")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint .zip to resume training from")
parser.add_argument("--xml", type=str, default=None, help="Override bump XML (curriculum stage)")
args = parser.parse_args()

if __name__ == "__main__":
    xml_file = os.path.abspath(args.xml) if args.xml else None
    # v10-speed: DummyVecEnv는 단일 프로세스라 12코어를 못 쓴다.
    # 채점에는 영향이 없고 학습 처리량만 오르므로 항상 SubprocVecEnv를 쓴다.
    VecEnv = SubprocVecEnv
    env = VecEnv([make_env(bump_practice=args.bump_practice, bump_challenge=args.bump_challenge, xml_file=xml_file) for _ in range(N_ENVS)])
    env = VecMonitor(env)

    # v9.1: 관측 정규화는 custom_walker2d.py에 고정 통계로 내장됨 (공식 채점기가
    # VecNormalize pkl을 읽지 않으므로). VecNormalize는 이중 정규화가 되어 미사용.
    if args.resume:
        cand = args.resume.replace(".zip", "").replace(
            "walker_model_", "walker_model_vecnormalize_") + ".pkl"
        if os.path.exists(cand):
            print(f"[info] {cand} 무시 — 관측 정규화는 환경에 내장됨")

    if args.bump_practice:
        folder_name = "bump_practice"
    elif args.bump_challenge:
        folder_name = "bump_challenge"
    else:
        folder_name = "walker_model"

    save_path = f'./checkpoints/{folder_name}/'

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # v11-bounding: 10000->20000 (N_ENVS=5 → 100k스텝마다 저장)
        save_path=save_path,
        name_prefix="walker_model",
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log="./logs/", device="cpu")
        model.verbose = 1
        model.ent_coef = 0.05  # v11-bounding: 0.03->0.05. bounding은 희귀 이벤트, 탐색 강화
        # "안정적으로 좋은 평균"보다 "체크포인트 중 최고 1개"가 중요하다.
        # 탐색을 키워 체크포인트 간 다양성(variation)을 늘린다.
        model.target_kl = 0.08  # 0.05->0.08. 정책이 더 크게 움직이도록 허용
        with torch.no_grad():
            # v11-bounding: log_std 하한 -0.3. bounding 도약을 위한 행동 노이즈 확대.
            # 학습된 각 관절 상대 크기(부호/차이)는 유지, 하한만 상향.
            model.policy.log_std.data.clamp_(min=-0.3)
        model.learn(total_timesteps=10000000000, callback=checkpoint_callback, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu",
            learning_rate=3e-4, n_steps=2048, batch_size=256, gamma=0.995, ent_coef=0.005, target_kl=0.03)
        model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
    model.save("ppo_custom_walker2d_parallel")
