import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from custom_walker2d import CustomEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 4

def make_env():
    def _init():
        return CustomEnvWrapper(render_mode=None, bump_practice=True)
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecMonitor(env)

    save_path = './checkpoints/bump_practice/'
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, save_path=save_path, name_prefix="bump_v6"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[128,64,64], vf=[128,64,64]), log_std_init=-1.0),
        learning_rate=0.0003, ent_coef=0.0, gamma=0.995)

    model.learn(total_timesteps=3000000, callback=checkpoint_callback)
    model.save(f"{save_path}bump_v6_final")
    print("V6 DONE")