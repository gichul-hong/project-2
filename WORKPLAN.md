# DS RL Project Work Plan

> **환경**: `pjt-2` conda env (`C:\Users\삼성\.conda\envs\pjt-2\python.exe`)
> **자료**: `docs/practice2_guide.md`, `docs/garage_rl_images/` (PDF는 image-only, 텍스트는 `garage_rl/README.md` 참고)
> **로깅**: tensorboard 사용 (`tensorboard --logdir=garage_rl/tensorboard --port=6006`)

---

## 프로젝트 구조

```
C:\hong\project-2\
├── garage_rl/                    # 실습1: Garage Scheduling PPO
│   ├── student/solution.py       # [EDIT] OBS_DIM + get_observation + compute_reward
│   ├── config/ppo_config.json    # [EDIT] PPO 하이퍼파라미터
│   ├── train.py                  # 학습 스크립트 (수정 불가)
│   ├── test.py                   # 평가 스크립트 (수정 불가)
│   └── env/                      # 환경 구현 (수정 불가, SHA256 검증)
├── practice2/                    # 실습2: Walker2D Controller
│   ├── custom_walker2d.py        # [EDIT] custom_observation/rewad/terminated/truncated
│   ├── learning.py               # [EDIT] policy_kwargs, learning_rate 등
│   ├── render.py                  # 시각화 (수정 불가)
│   └── asset/                     # MuJoCo XML 모델
└── .gitignore
```

> **수정 범위 주의**: `garage_rl/env/` 만 SHA256 검증 대상(`python verify_env.py`).
> `train.py`/`test.py`는 검증 대상이 아니지만 "수정하지 않아도 됨"이 원칙 —
> 결과 비교 가능성을 위해 수정하지 않는다.
> **`student/solution.py` 하나를 문제 1/2가 공유**하므로 레벨별 값 하드코딩 금지
> (예: 인내도는 반드시 `env.max_patience`로 정규화).

---

## Phase 0: 환경 세팅 [COMPLETED]

- [x] `pjt-2`에 패키지 설치: `stable_baselines3==2.7.0`, `gymnasium==1.1.1`, `torch==2.8.0+cpu`
- [x] PPTX → `docs/practice2_guide.md` (27 slides, images in `docs/images/`)
- [x] PDF 이미지 추출 → `docs/garage_rl_images/` (35 pages, image-only)
- [x] `.gitignore` 설정

---

## Phase 1: garage_rl — Garage Scheduling

> **[변경] 최종 제출 문제가 문제 3으로 변경됨** (`env/garage_3.py`, `garage_3.png`).
> 인내도 30틱, 매 틱 도착, A=손상도/B=연식≥20/C=크기≤4.0 조건. baseline 1490.8,
> Pass ≤1300, Distinguished ≤1050. 제출물: `student/solution.py` +
> `model/level3/ppo.zip` + A4 1장 리포트 PDF, **마감 17:20**.
> 상세 time-box 계획: `garage_rl/IMPLEMENTATION_GUIDE.md` §6 (최우선 실행).
> 기존 관측/보상 설계(21차원, 1:50 정렬)는 레벨 무관하게 그대로 적용 가능.

### 평가 지표: `점수 = 틱 수 + 50 × 이탈 차량 수` (낮을수록 좋음)

### 현재 상태 (`student/solution.py`)

| 항목 | 상태 | 내용 |
|------|------|------|
| `OBS_DIM` | `6` | 대기 슬롯 3 + 정비소 busy 3 |
| `get_observation` | 구현됨 | slot occupied boolean, station busy boolean |
| `compute_reward` | 기본 | assigned +1.0, invalid -0.5 |
| `ppo_config.json` | default | `total_timesteps=100` (너무 적음), `ent_coef=0.0` |

### Iteration Plan

#### Iter 1.0 — Baseline 확인

규칙 기반 베이스라인은 학습 없이 직접 확인한다 (README 문서값: Level 1 = 915.4, Level 2 = 630.4):

```bash
cd garage_rl
python env/garage_1.py      # rule-based baseline (100 episodes)
python env/garage_2.py
python verify_env.py        # env/ 원본 검증
```

**기록**: Level 1/2 rule-based baseline score (실측 재확인)

> Level 1은 세 정비소가 동일해 **점수를 낮출 방법이 없음** (README 명시).
> Level 1의 목표는 "베이스라인 근처 도달 = 학습 파이프라인 검증"이고,
> 실제 최적화 대상은 Level 2다.

#### Iter 1.1 — Observation 확장 + 파이프라인 검증

수정 파일:
- `student/solution.py`: `OBS_DIM` 증가, `get_observation()`에 per-car feature + station progress 추가
- `config/ppo_config.json`: `total_timesteps=200000` (약 1분), `ent_coef=0.01`

구체적 변경:
```python
# OBS_DIM = 6 + 3 + 3*4 = 21
#   slot_occupied[3] + station_busy[3]
#   + station_progress[3]           ← 추가: (할당틱-남은틱)/할당틱, idle=0 (busy flag가 idle/방금시작 구분)
#   + per_car[size, year, damage, patience_norm] × 3 slots  ← 추가
#     size=(car.size-3.0)/2.0, year=(car.year-10.0)/15.0, damage 그대로,
#     patience=env.car_patience[car.id]/env.max_patience  (하드코딩 금지)
```

- per-car `size` feature가 Level 2 핵심: size≤4 차량 → C(평균 ~9틱), 큰 차 → A/B를 **정책이 스스로 학습**하게 하는 유일한 통로.
- 관측은 매 스텝 [0,1] 범위 검증됨 — 정규화 누락 시 즉시 오류.

**검증 절차**:
1. Level 1 학습 → `test.py --level 1 --baseline`으로 베이스라인 근처(±5%) 확인 → 파이프라인 통과
2. Level 2 학습 → per-car feature 유/무 비교 (README (c) 전/후 비교 요구사항)

**기록**: Level 2 baseline 대비 gain%, per-car feature 유/무 비교, training curve

#### Iter 1.2 — Reward Shaping (점수-보상 정렬)

수정 파일: `student/solution.py`의 `compute_reward()`

**설계 원칙**: 점수 = 틱 + 50×이탈 이므로, 보상의 `틱당 페널티 : 이탈 페널티 = 1 : 50` 비율을 유지해 지표와 정렬한다. 비율을 깨는 sweep 조합은 지표와 어긋나므로 배제.

```python
def compute_reward(env, event):
    SCALE = 0.01                     # 보상 스케일 (sweep 축 1)
    EXPIRE_MULT = 50.0               # 이탈 가중 배율, 기본 = 지표와 동일 (sweep 축 2)
    reward = -SCALE                                  # 매 틱 시간 압박 (점수의 +1틱에 대응)
    reward -= event['expired'] * SCALE * EXPIRE_MULT # 이탈 (점수의 +50에 대응)
    if event['invalid']: reward -= 0.2 * SCALE * 10  # 헛발질 억제 (소량)
    # 선택 실험: 진행 신호 (지표에 없는 항목 → 보상 해킹 주의)
    # reward += event['finished'] * bonus            # 정비 완료 보너스
    # 주의: assigned 보너스는 "큰 차를 C에 빨리 넣기" 같은 악수 유발 가능 → 기본 제거
    # 주의: done 보너스는 모든 에피소드가 100대 처리 후 종료라 상수 신호 → 무의미, 사용 안 함
    return reward
```

**Sweep 포인트**:
- `SCALE` ∈ [0.005, 0.01, 0.02] (보상 크기만 조절, 비율 유지)
- `EXPIRE_MULT` ∈ [25, 50, 100] (이탈 회피 강조 정도)
- `finished` 보너스 유/무 (진행 신호가 학습 초기 도움 되는지)

**판단 기준**: `평균보상`이 아니라 train 로그의 `점수추정`(eval/score)과 `test.py` 점수. 평균보상↑인데 점수추정↓ 안 하면 보상 함수가 목표와 어긋난 것.

#### Iter 1.3 — Hyperparameter Tuning

`config/ppo_config.json` — 전수 grid(수백 조합)는 과다하므로 **1-factor 순차 sweep**: 기본값에서 축 하나씩 바꿔 best를 채택 후 다음 축으로.

탐색 순서 및 후보:
1. `gamma`: [0.99, 0.995, 0.999] ← 에피소드 600~2000틱, 이탈 신호(인내도 100틱)가 0.99 할인(유효 horizon ~100틱)으로 소실될 수 있어 최우선
2. `learning_rate`: [1e-4, 3e-4, 1e-3]
3. `ent_coef`: [0.0, 0.01, 0.05]
4. `n_steps`: [1024, 2048, 4096]
5. `batch_size` / `n_epochs`: [32, 64, 128] / [5, 10, 20] (여유 시)

각 후보는 `total_timesteps=200000`으로 비교, 최종 후보만 500k로 재확인.

#### Iter 1.4 — Level 2 검증

Level 2 특징:
- Station C: `size ≤ 4.0` 차량은 수리시간 50% 감소 (10~27틱 → 5~13틱)
- `ARRIVAL_MAX=0` → 대기열이 빌 때마다 즉시 채워짐
- `PATIENCE=100` → 긴 인내도

**주의**: `solution.py`는 두 문제가 공유 → `station=='C' and car.size<=4.0` 같은
Level 2 규칙 하드코딩 금지. C 활용 전략은 Iter 1.1의 per-car size 관측을 통해
정책이 학습해야 한다.

검증 항목:
- 학습된 정책이 실제로 작은 차를 C에 우선 배치하는지 debug 모드로 관찰 (`GarageEnv_2(debug=True)` 롤아웃)
- 안 되면: 관측에 정보 부족(1.1 재검토) vs 보상 신호 부족(1.2 재검토) vs 탐색 부족(`ent_coef`↑) 구분

#### Iter 1.5 — 최종 확인

- 최적 obs + reward + hyperparam 조합, `total_timesteps=500000`
- config의 `seed`를 [0, 1, 2]로 바꿔 3회 반복 학습 (seed 고정 시 반복이 동일 결과가 되므로 반드시 변경), 평균±표준편차 기록
- **최종 판정은 반드시 `test.py --level N --baseline`** (고정 시드 100 에피소드). 학습 중 `점수추정`은 5 에피소드 약식 지표일 뿐.
- best 모델의 `ppo.zip`을 백업 (`train.py` 재실행 시 덮어씀)

---

## Phase 2: practice2 — Walker2D Controller

### Goal: 20초 내에 (a) 평지 걷기 → (b) 자연스러운 보행 → (c) bump 통과

### 현재 상태 (`custom_walker2d.py`)

| 함수 | 상태 | 설명 |
|------|------|------|
| `custom_observation` | passthrough | obs 그대로 반환 |
| `custom_reward` | passthrough | original reward 그대로 |
| `custom_terminated` | passthrough | 기본 terminated |
| `custom_truncated` | passthrough | 기본 truncated |

### Observation Space (18-dim after excluding `exclude_current_positions_from_observation=False`)

```
[0]:  torso x       [3-8]: joint angles (thigh/leg/foot × right+left)
[1]:  torso z       [9-17]: joint velocities
[2]:  torso angle
```

### Iteration Plan

#### Iter 2.0 — Baseline

```bash
cd practice2
python learning.py
```

기본 reward + 기본 observation으로 flat ground 학습. 목표: 20초 내 종료.

#### Iter 2.1 — Flat Ground 걷기

`custom_reward()`:
```python
def custom_reward(self, obs, original_reward):
    # forward_velocity 강화
    forward_vel = obs[9]  # torso x-velocity
    # 너무 높이 뜨지 않게 torso z 제한
    height_penalty = max(0, obs[1] - 1.5)  # z > 1.5 페널티
    return original_reward + forward_vel * 2.0 - height_penalty * 0.5
```

#### Iter 2.2 — 자연스러운 보행

3가지 approach (PPTX slide 20-22 참고):
1. **상체 안정성**: torso angle(obs[2])의 절댓값 페널티, torso z-velocity(obs[10]) 페널티
2. **좌우 대칭**: right leg joints (obs[3:6])와 left leg joints (obs[6:9]) 간 L2 loss
3. **주기적 동작**: N-step 버퍼에 right leg 각도 저장 → N-step 후 left leg과 비교

```python
def custom_reward(self, obs, original_reward):
    reward = original_reward
    # 1. torso stability
    reward -= abs(obs[2]) * 0.5           # torso angle
    reward -= abs(obs[10]) * 0.1          # torso z-velocity
    # 2. symmetry
    sym_diff = np.sum((obs[3:6] - obs[6:9]) ** 2)
    reward -= sym_diff * 0.1
    # 3. forward velocity bonus
    reward += obs[9] * 1.0
    return reward
```

#### Iter 2.3 — Bump Practice

```bash
python learning.py --bump_practice
```

Reward 추가:
- bump 위를 넘을 때 torso 높이 변화에 덜 민감하게 (healthy_z_range 이미 (0.5, 10.0))
- bump 진입 시 slowdown penalty 방지 → `forward_vel * 0.5`로 가중치 축소

`custom_terminated()`:
- bump 통과 중 넘어짐 감지 → torso angle > 1.5 rad 시 early termination

#### Iter 2.4 — 최적화 Sweep

- `policy_kwargs`: net_arch variations
- `learning_rate`: [5e-5, 1e-4, 3e-4]
- `N_ENVS`: [4, 8]
- flat/natural/bump 각 모드 최적 조합 기록

---

## Phase 3: 결과 기록

wandb는 사용하지 않는다. 실험 추적은 다음으로 대체:

- **tensorboard**: `tensorboard --logdir=garage_rl/tensorboard --port=6006`
  - `eval/score`, `eval/gain_pct` (실제 지표), `rollout/score`, `rollout/removed`
  - config별로 `tensorboard_dir`을 분리하면 run 비교 가능
- **실험 로그 표**: 각 run의 config 요약 + `test.py` 최종 점수를 `garage_rl/EXPERIMENTS.md`에 표로 축적
  (config 파일명, obs 구성, reward 파라미터, seed, test score, baseline 대비 %)

---

## Agent Manager 병렬 실행 전략

### garage_rl 병렬화

한 번에 여러 config로 동시 학습. **주의**: `train.py`는 `model_dir/level{N}/ppo.zip`,
`tensorboard_dir/level{N}/`에 저장하므로 병렬 실행 시 **각 config 파일에 서로 다른
`model_dir`/`tensorboard_dir`을 반드시 지정**해야 덮어쓰기가 없다.

```json
// config/ppo_v1.json 예시 (경로 분리 필수)
{ ..., "model_dir": "model_v1", "tensorboard_dir": "tensorboard/v1" }
```

| Session | Command |
|---------|---------|
| agent-1 | `cd garage_rl && python train.py --level 2 --config config/ppo_v1.json` |
| agent-2 | `cd garage_rl && python train.py --level 2 --config config/ppo_v2.json` |
| agent-3 | `cd garage_rl && python train.py --level 2 --config config/ppo_v3.json` |

- 병렬 학습 결과는 `python test.py --level 2 --model model_v1/level2/ppo.zip` 처럼
  `--model` 인자로 개별 평가. best 모델만 기본 경로(`model/level2/ppo.zip`)로 복사해 보관.
- 관측/보상(`solution.py`)이 다른 실험은 파일이 공유되므로 **동시 실행 불가** — 하이퍼파라미터 sweep만 병렬화.

### practice2 병렬화

| Session | Command |
|---------|---------|
| agent-4 | `cd practice2 && python learning.py` (flat ground) |
| agent-5 | `cd practice2 && python learning.py --bump_practice` (bump) |

---

## 최종 산출물 체크리스트

- [ ] `garage_rl/student/solution.py` — 최종 observation + reward 구현
- [ ] `garage_rl/config/ppo_config.json` — 최종 하이퍼파라미터 (sweep용 버전은 별도 파일)
- [ ] `garage_rl/EXPERIMENTS.md` — iter별 config/점수 기록 표
- [ ] `practice2/custom_walker2d.py` — 최종 obs/reward/term 구현
- [ ] `practice2/learning.py` — 최적 config
- [ ] 각 iter의 best model 백업 및 `test.py` score 기록 (train 재실행 시 ppo.zip 덮어씀 주의)