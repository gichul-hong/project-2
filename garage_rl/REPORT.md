# 정비소 배치 문제 (Level 3) — 결과 리포트

**제출물**: `student/solution.py`, `model/level3/ppo.zip`, 본 리포트

## 1. 실행 방법

```bash
python train.py --level 3          # 학습 (config/ppo_config.json)
python test.py --level 3 --baseline  # 평가 (고정 시드 100 에피소드)
```

## 2. 관측 설계 (OBS_DIM = 27)

레벨별 하드코딩 없이 세 정비소의 조건(A: 손상도, B: 연식, C: 크기)을
정책이 스스로 학습할 수 있도록 차량 속성 전체를 노출하고,
정비소 가용 시점과 차량 이탈 시점을 같은 시간 단위로 비교할 수 있는
파생 피처를 추가했다.

| 인덱스 | 내용 | 정규화 |
|---|---|---|
| 0–2 | 대기 슬롯 점유 여부 | 0/1 |
| 3–5 | 정비소 A/B/C 가동 여부 | 0/1 |
| 6–8 | 정비소 진행률 | (할당틱−남은틱)/할당틱 |
| 9–11 | 정비소 절대 남은 틱 | min(1, 남은틱/50) |
| 12–23 | 슬롯별 차량 속성 ×3 | size (−3)/2, year (−10)/15, damage, patience/max_patience |
| 24–26 | 차량별 절대 남은 인내도 | min(1, 남은틱/50) |

절대 시간 피처(9–11, 24–26)는 "정비소가 비기 전에 차가 떠나는가"를
신경망이 직접 비교하게 하여 대기/배치 타이밍 학습을 돕는다.

## 3. 보상 설계 — 평가 지표와의 정렬

점수 = 틱 + 50×이탈 이므로, 보상을 점수의 음수 스케일과 정렬했다
(틱당 −0.01, 이탈당 −0.5 = **1:50 비율 유지**). 배치/종료 보너스는
지표에 없는 항목이라 보상 해킹(오배치 유도) 위험이 있어 배제했다.
invalid 행동에만 소량(−0.02) 페널티를 추가했다.

```python
reward = -0.01 - 0.5 * expired - (0.02 if invalid else 0)
```

## 4. 결과

TODO(최종값으로 교체): baseline 1490.8 (removed 15.6) 대비

| 모델 | test score | removed | 개선폭 |
|---|---|---|---|
| rule-based baseline | 1490.8 | 15.6 | — |
| PPO 200k | 1411.1 | 12.8 | +5.3% |
| PPO 500k (최종) | TODO | TODO | TODO |

## 5. 하이퍼파라미터 (최종)

TODO(최종값으로 교체): lr 3e-4, n_steps 2048, batch 64, n_epochs 10,
gamma 0.995, gae_lambda 0.95, clip 0.2, ent_coef 0.01, total_timesteps 500k.
sweep: gamma {0.995, 0.999}, ent_coef {0.01, 0.05} 비교 후 best 채택.
