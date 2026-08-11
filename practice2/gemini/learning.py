import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_walker2d import CustomEnvWrapper

# 병렬 환경 수 (CPU 코어에 맞게 조정)
N_ENVS = 8


def make_env(bump_practice=False, bump_challenge=False):
    def _init():
        return CustomEnvWrapper(render_mode=None,
                                bump_practice=bump_practice,
                                bump_challenge=bump_challenge)
    return _init


# 충분한 표현력 확보를 위한 네트워크 구조
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    log_std_init=-1.0,
)

parser = argparse.ArgumentParser()
parser.add_argument("--bump_practice", action="store_true",
                    help="Train on the 2-bump practice track (Task 3)")
parser.add_argument("--bump_challenge", action="store_true",
                    help="Train on the bump challenge track")
parser.add_argument("--init_model", type=str, default=None,
                    help="Warm-start from a saved .zip (e.g. flat-ground checkpoint)")
parser.add_argument("--init_vecnormalize", type=str, default=None,
                    help="Matching VecNormalize stats (.pkl) for --init_model")
args = parser.parse_args()

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(bump_practice=args.bump_practice,
                                  bump_challenge=args.bump_challenge)
                         for _ in range(N_ENVS)])
    env = VecMonitor(env)

    if args.init_vecnormalize:
        # Warm-start: 기존 obs 정규화 통계 재사용
        env = VecNormalize.load(args.init_vecnormalize, env)
    else:
        # norm_reward=False: 수작업 O(1) 스케일 보상을 보존
        # norm_obs=True: position/angle/velocity의 상이한 스케일을 정규화
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.bump_practice:
        folder_name = "bump_practice"
    elif args.bump_challenge:
        folder_name = "bump_challenge"
    else:
        folder_name = "walker_model"

    save_path = f'./checkpoints/{folder_name}/'

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="walker_model",
        save_vecnormalize=True,  # VecNormalize 통계도 함께 저장
    )

    if args.init_model:
        print(f"Warm-starting from {args.init_model}")
        model = PPO.load(args.init_model, env=env, device="cpu",
                          tensorboard_log="./logs/")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            policy_kwargs=policy_kwargs,
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,    # 걷기 탐색을 위한 적절한 entropy
        )

    model.learn(total_timesteps=10_000_000_000, callback=checkpoint_callback,
               reset_num_timesteps=(args.init_model is None))
    model.save("ppo_custom_walker2d_parallel")
    env.save(f"{save_path}vecnormalize_final.pkl")
