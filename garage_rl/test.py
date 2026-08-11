"""
평가 스크립트.

    python test.py --level 1
    python test.py --level 2 --model model/level2/ppo.zip

고정된 평가 시드 100개로 돌려 평균을 냅니다. 같은 모델이면 항상 같은 결과가
나오므로, 여러분이 바꾼 것이 실제로 도움이 되었는지 비교할 수 있습니다.

점수 = 소요 틱 수 + 50 * 이탈 차량 수   (낮을수록 좋음)

--baseline 을 주면 규칙 기반 베이스라인 점수를 같이 출력합니다.

이 파일은 수정하지 않아도 됩니다.
"""

import argparse
import os
import statistics
import sys

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

EVAL_SEED_BASE = 10_000     # 학습에 쓰이는 시드와 겹치지 않도록 고정


def rollout(env_cls, policy, seed, need_obs=True):
    env = env_cls(debug=False, seed=seed, need_obs=need_obs)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    while True:
        action = policy(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if terminated or truncated:
            break
    return {
        'score': step + 50 * info['removed'],
        'steps': step,
        'reward': total_reward,
        'serviced': info['serviced'],
        'removed': info['removed'],
        'truncated': truncated,
    }


def summarize(label, results):
    scores = [r['score'] for r in results]
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"  {label:<22} score {statistics.mean(scores):8.1f}  (sd {sd:5.1f})"
          f"   steps {statistics.mean(r['steps'] for r in results):7.1f}"
          f"   serviced {statistics.mean(r['serviced'] for r in results):6.2f}"
          f"   removed {statistics.mean(r['removed'] for r in results):5.2f}")
    return statistics.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--model', default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--baseline', action='store_true',
                        help="규칙 기반 베이스라인 점수도 함께 출력")
    args = parser.parse_args()

    if args.level not in ENVS:
        sys.exit(f"문제 {args.level} 의 환경 파일이 없습니다 "
                 f"(env/garage_{args.level}.py).")

    env_cls = ENVS[args.level]
    seeds = [EVAL_SEED_BASE + i for i in range(args.episodes)]

    print(f"\n=== level {args.level} · {args.episodes} episodes "
          f"(낮을수록 좋음) ===")

    base_score = None
    if args.baseline:
        base = [rollout(env_cls, lambda e, o: baseline_action(e), s, need_obs=False)
                for s in seeds]
        base_score = summarize("rule-based baseline", base)

    model_path = args.model or os.path.join("model", f"level{args.level}", "ppo.zip")
    if not os.path.exists(model_path):
        print(f"  (모델 파일이 없습니다: {model_path} — 먼저 train.py 를 실행하세요)")
        return

    from student import solution
    if int(solution.OBS_DIM) <= 0:
        print("  (student/solution.py 의 OBS_DIM 이 아직 0 입니다 — "
              "TODO 를 채운 뒤 다시 학습하세요)")
        return

    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    def ppo_policy(env, obs):
        return int(model.predict(obs, deterministic=True)[0])

    agent = [rollout(env_cls, ppo_policy, s) for s in seeds]
    agent_score = summarize("PPO agent", agent)

    n_trunc = sum(r['truncated'] for r in agent)
    if n_trunc:
        print(f"  경고: {n_trunc} 에피소드가 시간 상한에 걸려 강제 종료되었습니다.")

    if base_score is not None:
        gain = 100.0 * (base_score - agent_score) / base_score
        if gain > 0.5:
            verdict = "베이스라인보다 좋음"
        elif gain < -0.5:
            verdict = "베이스라인보다 나쁨"
        else:
            verdict = "베이스라인과 사실상 동일"
        print(f"\n  개선폭: {gain:+.1f}%  ({verdict})")


if __name__ == "__main__":
    main()
