import argparse
import cv2
import numpy as np
import os  
from stable_baselines3 import PPO
from custom_walker2d import CustomEnvWrapper
import gymnasium as gym

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--bump_practice", action="store_true", help="Enable bumping") # For bump practice
parser.add_argument("--bump_challenge", action="store_true", help="Enable bumping") # For bump challenge
parser.add_argument("--record", action="store_true", help="Enable recording with R key toggle") # For recording
parser.add_argument("--xml", type=str, default=None, help="Override bump XML (curriculum stage)")
args = parser.parse_args()

render_mode = "rgb_array" if args.record else "human"
xml_file = os.path.abspath(args.xml) if args.xml else None
raw_env = CustomEnvWrapper(render_mode=render_mode, bump_practice=args.bump_practice, bump_challenge=args.bump_challenge, xml_file=xml_file)

is_vec_env = False
if args.model is not None:
    dir_name = os.path.dirname(args.model)
    base_name = os.path.basename(args.model).replace(".zip", "")
    vec_path = os.path.join(dir_name, f"{base_name.replace('walker_model_', 'walker_model_vecnormalize_')}.pkl")
    if not os.path.exists(vec_path):
        vec_path = os.path.join(dir_name, f"{base_name}_vecnormalize.pkl")
    
    if os.path.exists(vec_path):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        print(f"Loading VecNormalize stats from {vec_path}")
        vec_env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize.load(vec_path, vec_env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(args.model, env=env)
        is_vec_env = True
    else:
        env = raw_env
        model = PPO.load(args.model)
else:
    env = raw_env
    model = None

if is_vec_env:
    obs = env.reset()
else:
    obs, _ = env.reset()

video_writer = None
recording = False
frames = []

if args.record:
    print("Recording enabled. Press 'R' to start/stop recording, 'Q' to quit.")

while True:
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()
    
    if is_vec_env:
        obs, reward, done, info = env.step(action)
        terminated = done[0]
        truncated = False
    else:
        obs, reward, terminated, truncated, _ = env.step(action)

    if args.record:
        frame = raw_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if recording:
            frames.append(frame_bgr)

        cv2.imshow("Walker2D", frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                print("Recording")
                recording = True
                frames = []
            else:
                print("Recording stopped & saving...")
                recording = False
                if frames:
                    height, width, _ = frames[0].shape
                    file_name = f"recorded_{os.path.splitext(os.path.basename(args.model))[0]}.mp4"
                    video_writer = cv2.VideoWriter(file_name,
                                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                                   60, (width, height))
                    for f in frames:
                        video_writer.write(f)
                    video_writer.release()
                    print("recorded.mp4 saved successfully")

        if key == ord('q'):
            print("Exiting...")
            break

    if terminated or truncated:
        if is_vec_env:
            obs = env.reset()
        else:
            obs, _ = env.reset()
