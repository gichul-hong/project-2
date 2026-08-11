# garage_rl 실험 기록 (Level 3)

baseline (rule-based, level 3, test.py 100ep): **1490.8** (steps 710.8, removed 15.60)
판정: Pass ≤ 1300, Distinguished ≤ 1050. 참고 최고 950.

## Observation 진화

- **v1 (21d)**: slot_occupied[3] + station_busy[3] + station_progress[3] + per_car[size,year,damage,patience_norm]×3
- **v2 (27d)**: v1 + station_rem_ticks[3] + car_patience_abs[3] (같은 틱/50 단위)

## Reward 진화

- **R0**: SCALE=0.01, EXPIRE_MULT=50, INVALID=0.02, FINISH=0 (지표 1:50 정렬)
- **R1**: R0 + FINISH=0.05 (throughput 보너스 — 과잉 대기 억제)

## 전체 run 결과 (test.py 100ep)

| # | obs | rew | config (γ/ent/seed/steps) | score | steps | removed | 판정 |
|---|-----|-----|---|-----------|--------|---------|------|
| baseline | — | — | rule-based | 1490.8 | 710.8 | 15.60 | — |
| 200k | v1 | R0 | 0.995/0.01/0/200k | 1411.1 | 769.1 | 12.84 | +5.3% |
| l3v2_a | v2 | R0 | 0.995/0.01/0/500k | 1437.8 | 798.3 | 12.79 | +3.6% |
| l3v2_b | v2 | R0 | 0.999/0.01/0/500k | 1419.3 | 741.3 | 13.56 | +4.8% |
| l3v2_c | v2 | R0 | 0.995/0.05/0/500k | 1399.6 | 798.6 | 12.02 | +6.1% |
| **l3v3_a** | v2 | R1 | 0.995/0.05/0/500k | **1280.8** | 736.8 | 10.88 | **Pass +14.1%** |
| l3v3_b | v2 | R1 | 0.995/0.05/1/500k | 1443.1 | 808.6 | 12.69 | +3.2% |
| l3v3_c | v2 | R1 | 0.999/0.05/0/500k | 2582.4 | 771.9 | 36.21 | γ 발산 |

## 결론 (진행 중)

- R_FINISH=0.05 도입이 결정적 (steps 87→26 초과분 압축)
- γ=0.999는 R_FINISH와 결합 시 발산 (미래 보너스 과대평가)
- seed 분산 큼 (1280 vs 1443) → v4에서 seed {2,3,42} 추가 실행 중
- best 모델(`model/level3/ppo.zip`) = l3v3_a, 1280.8, Pass 확정
