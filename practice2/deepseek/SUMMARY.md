# Practice 2 최종 결과

> 2026-08-11 | Conda env: `rl`

---

## 최종 성능

| Task | Best Model | Checkpoint | avg_x | avg_steps | survived | bump1 | bump2 |
|------|-----------|------------|-------|-----------|----------|-------|-------|
| 1 (걷기) | walker_model | 2,040,000 | **50.1** | 994 | 0/20 | — | — |
| 2 (자연스러움) | walker_model | 2,040,000 | **50.1** | 994 | 0/20 | — | — |
| 3 (bump) | walker_model_v2 | 3,600,000 | **9.1** | 642 | 0/20 | 18/20 | **9/20** |

---

## Task 1 & 2: Flat Ground Walker

**결과**: avg_x=50.1, avg_steps=994. 20 ep 중 생존 0회지만 스텝 994로 거의 풀에피소드 소화.

**사용한 config**:
- reward: vanilla Walker2D-v5 (`healthy + forward - ctrl_cost`)
- HP: `ent_coef=0, gamma=0.995, lr=0.0001, net=[128,64,64]`
- N_ENVS=4, ~500 fps
- 2,040,000 steps

**학습 곡선**:

| Steps | avg_x | avg_steps |
|-------|-------|-----------|
| 1.2M | 24.8 | 760 |
| 1.6M | 33.7 | 819 |
| 2.0M | 49.5 | 999 |
| 2.04M | 50.1 | 994 |

---

## Task 3: Bump Traversal

**결과**: avg_x=9.1, bump1=18/20, bump2=9/20. **bump2 통과가 불안정 (45%)**.

**사용한 config**:
- observation: 22-dim (18 base + 4 bump info: dist to bump1/2, passed bump1/2)
- reward: vanilla + `bump_pass_bonus=+50` + 미세 안정성 페널티 (`torso_angle=-0.005`, `torso_vel=-0.002`, `sym=-0.003`)
- HP: `ent_coef=0, gamma=0.995, lr=0.0003, net=[128,64,64]`
- N_ENVS=4, ~550 fps
- v1: 2,000,000 steps, v2: 3,600,000 steps (continued)

**학습 곡선**:

| Steps | avg_x | bump1 | bump2 |
|-------|-------|-------|-------|
| 1.4M | 7.8 | 9/10 | 0/10 |
| 2.0M | 9.2 | 10/10 | 0/10 |
| 3.6M | 9.1 | 18/20 | 9/20 |

---

## 실패 원인 분석

### 1. 초기 보상 설계 오류 (LESSONS_LEARNED §3.1 위반)
- torso 안정성 페널티를 **0.05/0.02/0.03**으로 너무 크게 설정 → 720k에서 x=9.3 (기존 800k x=17 보다 훨씬 낮음)
- §3.3 "극단값 금지"에 따라 10분의 1로 축소했으나, 초기 학습 지연이 회복 불가했음
- 교훈: **보상은 지표 정렬이 우선, 보조 신호는 지표 달성 후 추가**

### 2. Bump 환경의 근본적 어려움
- Bump #2 (x=10.0, size 0.6x2.0x0.45)는 상당한 장애물
- Walker가 bump를 넘으려면 점프/등반 동작을 학습해야 하는데, forward-only reward로는 유도 부족
- 3.6M steps 학습 후에도 45%만 통과 → **reward 구조 자체의 한계**

### 3. SubprocVecEnv 불안정
- Windows 환경에서 SubprocVecEnv가 간헐적으로 hang 발생
- DummyVecEnv로 전환했으나 4 env에서 fps 저하

---

## 제출 모델

| Task | 모델 경로 |
|------|----------|
| 1, 2 | `checkpoints/walker_model/walker_model_2040000_steps.zip` |
| 3 | `checkpoints/bump_practice/walker_model_v2_3600000_steps.zip` |

---

## LESSONS_LEARNED.md 대비 이번 과제 특이점

- **§4.1 rich obs → ent=0**: Walker2D 18-dim obs에 ent_coef=0 적용이 유효했음
- **§4.2 gamma=0.995**: 기본 0.99 대비 long-horizon 신호 전파에 도움
- **§1.1 보상-지표 정렬**: Task 3에서 bump 보너스(+50)를 추가했으나, forward reward(ep당 ~1000) 대비 충분하지 않았을 가능성
- **§1.3 time-box**: bump 학습에 시간을 더 투자했어야 함 (2M → 3.6M으로 연장)
- **§8 반복 실패**: `ent_coef` 과다나 `gamma=0.999`+bonus 발산은 없었으나, **초기 penalty 과다**가 새로운 실패 패턴