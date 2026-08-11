"""
학습 스크립트.

    python train.py --level 1
    python train.py --level 2

하이퍼파라미터는 config/ppo_config.json 에서 읽습니다.
모델은 model/level{N}/ppo.zip, 로그는 tensorboard/level{N}/ 에 저장됩니다.

이 파일은 수정하지 않아도 됩니다. 여러분이 고칠 것은
  - student/solution.py   (관측과 보상)
  - config/ppo_config.json (하이퍼파라미터)
두 개입니다.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

# 이 문제의 신경망은 매우 작아서(21 -> 64 -> 64 -> 10) 여러 스레드를 쓰면
# 분배 오버헤드가 이득을 넘어 오히려 느려집니다. 실측: 1스레드 127초,
# 8스레드 142초, 16스레드 168초 (500k timesteps 기준).
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env._core import baseline_action
from env.garage_1 import GarageEnv_1
from env.garage_2 import GarageEnv_2

ENVS = {1: GarageEnv_1, 2: GarageEnv_2}

# 문제 3 은 challenge 로 따로 배포합니다. 파일이 있을 때만 등록합니다.
try:
    from env.garage_3 import GarageEnv_3
    ENVS[3] = GarageEnv_3
except ImportError:
    pass

PEEK_SEED_BASE = 90_000     # 학습 중 중간 점검용 (평가 시드와 겹치지 않게)


def quick_score(env_cls, model, episodes=5):
    """학습 중간에 실제 지표(점수)를 싸게 엿본다. 5 에피소드에 약 0.2초."""
    scores = []
    for i in range(episodes):
        env = env_cls(debug=False, seed=PEEK_SEED_BASE + i)
        obs, _ = env.reset(seed=PEEK_SEED_BASE + i)
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            step += 1
            if terminated or truncated:
                break
        scores.append(step + 50 * info['removed'])
    return statistics.mean(scores)


class Progress(BaseCallback):
    """
    진행 상황을 stdout 한 줄로 찍고, 평가 지표를 tensorboard 에 기록한다.

    SB3 가 기본으로 남기는 `rollout/ep_rew_mean` 은 여러분이 설계한 보상이라
    보상 함수를 바꾸면 서로 비교할 수 없다. 그래서 실제 평가 지표
    (= 소요 틱 + 50 × 이탈)를 따로 기록한다.

      eval/score      결정적 정책으로 5 에피소드 실측 (평가와 같은 방식)
      eval/gain_pct   베이스라인 대비 개선폭 %
      rollout/score   학습 중 에피소드에서 계산한 점수 (공짜, 확률적 정책)
      rollout/removed 학습 중 에피소드의 평균 이탈 대수
    """

    def __init__(self, env_cls, total, baseline, print_every=10, score_every=10_240):
        super().__init__()
        self.env_cls = env_cls
        self.total = total
        self.baseline = baseline
        self.print_mark = max(1, total // print_every)
        self.next_print = self.print_mark
        self.score_every = score_every
        self.next_score = 0
        self.t0 = None
        self.removed = []      # 최근 학습 에피소드의 이탈 대수

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        print(f"  베이스라인 {self.baseline:.1f} (규칙 기반, 낮을수록 좋음)")
        print(f"  {'진행':>5} {'스텝':>9} {'경과':>7} {'잔여':>7} "
              f"{'평균보상':>10} {'점수추정':>9} {'대비':>8}")

    def _fmt(self, sec):
        return f"{int(sec) // 60}:{int(sec) % 60:02d}"

    def _on_step(self):
        # 끝난 에피소드에서 이탈 대수를 모은다 (추가 비용 없음)
        for info in self.locals.get('infos', ()):
            if 'episode' in info and 'removed' in info:
                self.removed.append(info['removed'])
        del self.removed[:-100]

        ep = self.model.ep_info_buffer
        if ep and self.removed:
            length = statistics.mean(e['l'] for e in ep)
            rm = statistics.mean(self.removed)
            self.logger.record('rollout/removed', rm)
            self.logger.record('rollout/score', length + 50 * rm)

        # 결정적 정책으로 실제 지표를 엿본다
        if self.num_timesteps >= self.next_score:
            self.next_score = self.num_timesteps + self.score_every
            score = quick_score(self.env_cls, self.model)
            self.logger.record('eval/score', score)
            self.logger.record('eval/gain_pct',
                               100.0 * (self.baseline - score) / self.baseline)
            self._last_score = score

        if self.num_timesteps < self.next_print:
            return True
        self.next_print += self.print_mark

        elapsed = time.time() - (self.t0 or time.time())
        frac = self.num_timesteps / self.total
        eta = elapsed / frac - elapsed if frac > 0 else 0

        rew = [e['r'] for e in ep] if ep else []
        rew_s = f"{statistics.mean(rew):10.1f}" if rew else f"{'-':>10}"

        score = getattr(self, '_last_score', None)
        if score is None:
            score = quick_score(self.env_cls, self.model)
        gain = 100.0 * (self.baseline - score) / self.baseline

        print(f"  {frac * 100:4.0f}% {self.num_timesteps:9,} "
              f"{self._fmt(elapsed):>7} {self._fmt(eta):>7} {rew_s} "
              f"{score:9.1f} {gain:+7.1f}%", flush=True)
        return True


def baseline_score(env_cls, episodes=20):
    scores = []
    for i in range(episodes):
        env = env_cls(debug=False, seed=PEEK_SEED_BASE + i, need_obs=False)
        env.reset(seed=PEEK_SEED_BASE + i)
        step = 0
        while True:
            _, _, terminated, truncated, info = env.step(baseline_action(env))
            step += 1
            if terminated or truncated:
                break
        scores.append(step + 50 * info['removed'])
    return statistics.mean(scores)


def load_config(path="./config/ppo_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_env(level, seed=None, debug=False):
    def _init():
        return Monitor(ENVS[level](debug=debug, seed=seed))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--config', default="./config/ppo_config.json")
    args = parser.parse_args()

    if args.level not in ENVS:
        sys.exit(f"문제 {args.level} 의 환경 파일이 없습니다 "
                 f"(env/garage_{args.level}.py).")

    cfg = load_config(args.config)
    level = args.level

    save_path = os.path.join(cfg["model_dir"], f"level{level}", "ppo")
    tb_path = os.path.join(cfg["tensorboard_dir"], f"level{level}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(tb_path, exist_ok=True)

    env = VecMonitor(DummyVecEnv([make_env(level, seed=cfg.get("seed"))]))

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tb_path,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        seed=cfg.get("seed"),
        verbose=cfg["verbose"],
    )

    print(f"\n[level {level}] 관측 차원 {env.observation_space.shape[0]}, "
          f"total_timesteps {cfg['total_timesteps']:,}, "
          f"n_steps {cfg['n_steps']}, ent_coef {cfg['ent_coef']}")

    base = baseline_score(ENVS[level])
    progress = Progress(ENVS[level], cfg["total_timesteps"], base)

    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=progress)
    elapsed = time.time() - t0

    model.save(save_path)
    print(f"\n  학습 시간 {int(elapsed) // 60}분 {int(elapsed) % 60}초 "
          f"({cfg['total_timesteps'] / max(elapsed, 1e-9):.0f} steps/s)")
    print(f"  저장: {save_path}.zip")
    print(f"  평가: python test.py --level {level} --baseline")
    print(f"  곡선: tensorboard --logdir={cfg['tensorboard_dir']} --port=6006")


if __name__ == "__main__":
    main()
