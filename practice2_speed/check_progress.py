"""학습 진행 상황 확인 (AI 안 거치고 직접 실행용)

사용법: python check_progress.py
"""
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sorted(glob.glob("logs/PPO_*"), key=os.path.getmtime)[-1]
ea = EventAccumulator(run)
ea.Reload()
l = ea.Scalars("rollout/ep_len_mean")
r = ea.Scalars("rollout/ep_rew_mean")
print(f"run: {run}")
for i in range(max(0, len(l) - 8), len(l)):
    step = l[i].step / 1e6
    rps = r[i].value / max(l[i].value, 1)
    print(f"{step:6.2f}M  ep_len {l[i].value:5.0f}  ep_rew {r[i].value:6.0f}  rew/step {rps:.2f}")

ckpts = sorted(glob.glob("checkpoints/bump_challenge/*.zip"), key=os.path.getmtime)
print(f"\n최신 체크포인트: {ckpts[-1] if ckpts else '없음'}")
print("판독: rew/step ~1.0 = 정지 국소최적 / 2.5+ = 전진+보너스 수집 / ep_len 800+ = 승급 검토")
