import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_walker2d import CustomEnvWrapper

# Number of parallel environments (SubprocVecEnv)
N_ENVS = 8


def make_env(bump_practice=False, bump_challenge=False):
    def _init():
        return CustomEnvWrapper(render_mode=None,
                                bump_practice=bump_practice,
                                bump_challenge=bump_challenge)
    return _init


# Policy/value networks with enough capacity for the richer observation.
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    log_std_init=-1.0,
)

parser = argparse.ArgumentParser()
parser.add_argument("--bump_practice", action="store_true", help="Train on the 2-bump practice track (Task 3)")
parser.add_argument("--bump_challenge", action="store_true", help="Train on the bump challenge track")
args = parser.parse_args()

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(bump_practice=args.bump_practice,
                                  bump_challenge=args.bump_challenge)
                         for _ in range(N_ENVS)])
    env = VecMonitor(env)

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
    )

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
        ent_coef=0.005,  # raised from 0.001: extra early exploration to escape single-leg hopping
    )

    model.learn(total_timesteps=10_000_000_000, callback=checkpoint_callback)
    model.save("ppo_custom_walker2d_parallel")
