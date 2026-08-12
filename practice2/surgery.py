"""v9 가중치 수술: 구 정책(관측 22차원, 구맵 하드코딩) -> 신 정책(관측 25차원, 맵 불변).

구 관측 배치:  [0..17]=base 18, [18..20]=범프1~3 거리(/20), [21]=torso_z-1.25
신 관측 배치:  [0..17]=base 18, [18,19,20]=다음범프1(거리/10, 높이, 반폭),
               [21,22,23]=다음범프2(거리/10, 높이, 반폭), [24]=torso_z-1.25

첫 Linear 레이어(in_features == obs_dim)만 열 매핑으로 이식하고 나머지는 그대로 복사.
거리 스케일이 /20 -> /10로 2배 커졌으므로 해당 열 가중치는 x0.5로 보정.

사용:
  python surgery.py --src checkpoints/bump_challenge_oldmap/walker_model_32200000_steps.zip \
                    --out checkpoints/bump_challenge/walker_model_surgery_0_steps.zip
"""
import argparse
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_walker2d import CustomEnvWrapper

# learning.py는 import 시 argparse를 실행하므로 값만 동일하게 복제
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    log_std_init=-1.0,
)

OLD_OBS_DIM = 22
# 신 열 -> (구 열, 스케일). 없는 열(높이/반폭)은 0 초기화.
COL_MAP = {**{i: (i, 1.0) for i in range(18)},
           18: (18, 0.5),   # 다음 범프 거리 <- 구 범프1 거리 (/20 -> /10)
           21: (19, 0.5),   # 둘째 범프 거리 <- 구 범프2 거리
           24: (21, 1.0)}   # torso_z offset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--xml", default=None)
    p.add_argument("--graft_vecnorm", action="store_true",
                   help="구 VecNormalize 통계도 열 매핑으로 이식 (검증용)")
    args = p.parse_args()

    xml = os.path.abspath(args.xml) if args.xml else None
    env = DummyVecEnv([lambda: CustomEnvWrapper(
        render_mode=None, bump_challenge=True, xml_file=xml)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    new_obs_dim = env.observation_space.shape[0]
    print(f"new obs dim = {new_obs_dim}")

    new_model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs,
                    device="cpu", learning_rate=3e-4, n_steps=2048,
                    batch_size=256, gamma=0.995, ent_coef=0.01, target_kl=0.03)
    old_model = PPO.load(args.src, device="cpu")

    new_sd = new_model.policy.state_dict()
    old_sd = old_model.policy.state_dict()

    print("--- policy state_dict ---")
    moved, grafted, skipped = [], [], []
    for key, new_w in new_sd.items():
        if key not in old_sd:
            skipped.append(f"{key} (구 모델에 없음)")
            continue
        old_w = old_sd[key]
        if old_w.shape == new_w.shape:
            new_sd[key] = old_w.clone()
            moved.append(key)
            continue
        # shape 불일치는 첫 레이어(in_features==obs_dim)의 weight만 허용
        if (new_w.dim() == 2 and new_w.shape[1] == new_obs_dim
                and old_w.shape[1] == OLD_OBS_DIM
                and old_w.shape[0] == new_w.shape[0]):
            w = torch.zeros_like(new_w)
            for new_col, (old_col, scale) in COL_MAP.items():
                w[:, new_col] = old_w[:, old_col] * scale
            new_sd[key] = w
            grafted.append(f"{key} {tuple(old_w.shape)} -> {tuple(new_w.shape)}")
        else:
            skipped.append(f"{key} shape {tuple(old_w.shape)} vs {tuple(new_w.shape)}")

    new_model.policy.load_state_dict(new_sd)
    print(f"copied ({len(moved)}): {moved}")
    print(f"grafted ({len(grafted)}): {grafted}")
    print(f"skipped ({len(skipped)}): {skipped}")
    if not grafted:
        raise SystemExit("첫 레이어 이식 실패 - 레이어 이름/shape 확인 필요")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_model.save(args.out)
    print(f"saved -> {args.out}")

    if args.graft_vecnorm:
        src_pkl = args.src.replace(".zip", "").replace(
            "walker_model_", "walker_model_vecnormalize_") + ".pkl"
        if not os.path.exists(src_pkl):
            raise SystemExit(f"{src_pkl} 없음")
        old_vn = pickle.loads(open(src_pkl, "rb").read())
        om, ov = old_vn.obs_rms.mean, old_vn.obs_rms.var
        # 신 관측값 = 구 관측값 x (1/scale). 거리 열은 /20 -> /10 이므로 2배.
        mean = np.zeros(new_obs_dim)
        var = np.ones(new_obs_dim)
        for new_col, (old_col, scale) in COL_MAP.items():
            inv = 1.0 / scale
            mean[new_col] = om[old_col] * inv
            var[new_col] = ov[old_col] * inv * inv
        for new_col, init_mean in ((19, 0.3), (20, 0.4), (22, 0.3), (23, 0.4)):
            mean[new_col] = init_mean
            var[new_col] = 1.0
        env.obs_rms.mean = mean
        env.obs_rms.var = var
        env.obs_rms.count = old_vn.obs_rms.count
        out_pkl = args.out.replace(".zip", "").replace(
            "walker_model_", "walker_model_vecnormalize_") + ".pkl"
        env.save(out_pkl)
        print(f"saved vecnormalize -> {out_pkl}")

    # 동작 점검: 수술본으로 몇 스텝 굴려 관측/행동에 NaN이 없는지 확인
    obs = env.reset()
    for _ in range(50):
        action, _ = new_model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if np.any(np.isnan(obs)) or np.any(np.isnan(action)):
            raise SystemExit("NaN 발생")
    print("smoke test ok (no NaN)")
    print("VecNormalize 통계는 이식하지 않음 (새로 시작) - 계획서 Phase 4 간단 대안")


if __name__ == "__main__":
    main()
