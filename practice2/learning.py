from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_walker2d import CustomEnvWrapper

N_ENVS = 10

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
    env = SubprocVecEnv([make_env(bump_practice=args.bump_practice, bump_challenge=args.bump_challenge, xml_file=xml_file) for _ in range(N_ENVS)])
    env = VecMonitor(env)

    vecnorm_path = None
    if args.resume:
        ckpt = args.resume.replace(".zip", "")
        cand = ckpt.replace("walker_model_", "walker_model_vecnormalize_") + ".pkl"
        if os.path.exists(cand):
            vecnorm_path = cand

    if vecnorm_path:
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.bump_practice:
        folder_name = "bump_practice"
    elif args.bump_challenge:
        folder_name = "bump_challenge"
    else:
        folder_name = "walker_model"

    save_path = f'./checkpoints/{folder_name}/'

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix="walker_model",
        save_vecnormalize=True,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log="./logs/", device="cpu")
        model.ent_coef = 0.01  # 범프 등반 동작 탐색을 위해 상향
        model.learn(total_timesteps=10000000000, callback=checkpoint_callback, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu",
            learning_rate=3e-4, n_steps=2048, batch_size=256, gamma=0.995, ent_coef=0.005, target_kl=0.03)
        model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
    model.save("ppo_custom_walker2d_parallel")
    env.save(f"{save_path}vecnormalize_final.pkl")
