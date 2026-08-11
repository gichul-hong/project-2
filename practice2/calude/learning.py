import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
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
parser.add_argument("--init_model", type=str, default=None,
                    help="Warm-start from a saved .zip (e.g. a flat-ground checkpoint) instead of a fresh policy")
parser.add_argument("--init_vecnormalize", type=str, default=None,
                    help="Matching VecNormalize stats (.pkl) for --init_model. Required to warm-start "
                         "correctly, otherwise the loaded policy sees renormalized observations it "
                         "was never trained on.")
args = parser.parse_args()

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(bump_practice=args.bump_practice,
                                  bump_challenge=args.bump_challenge)
                         for _ in range(N_ENVS)])
    env = VecMonitor(env)

    if args.init_vecnormalize:
        # Warm start: reuse the running obs statistics collected so far so
        # normalization stays consistent with what the loaded policy saw.
        env = VecNormalize.load(args.init_vecnormalize, env)
    else:
        # norm_reward=False: keep our hand-tuned O(1) reward scale intact,
        # only observations (which mix positions/angles/velocities of very
        # different magnitudes) are normalized for training stability.
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
        save_vecnormalize=True,  # persists VecNormalize running stats alongside the model
    )

    if args.init_model:
        # Warm-start: keep the gait learned on flat ground and continue
        # training in the target environment (e.g. bump_practice).
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
            ent_coef=0.005,   # extra early exploration to escape single-leg hopping
            # NOTE: target_kl was tried (0.03, then 0.1) as a stability guard but
            # backfired: it early-stopped training to ~0-2 minibatch updates per
            # rollout, leaving approx_kl stuck at 0.05-0.09 and ep_rew_mean flat
            # at -46 for 700k+ steps. Left unset (SB3 default = None / full
            # n_epochs) so PPO gets its full 10 epochs of updates per rollout.
        )

    model.learn(total_timesteps=10_000_000_000, callback=checkpoint_callback,
               reset_num_timesteps=(args.init_model is None))
    model.save("ppo_custom_walker2d_parallel")
    env.save(f"{save_path}vecnormalize_final.pkl")
