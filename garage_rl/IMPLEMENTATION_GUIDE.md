# garage_rl 구현 지시서 (Implementation Guide)

> 이 문서는 구현 담당 에이전트/모델을 위한 상세 작업 지시서다.
> 전체 계획의 배경은 `../WORKPLAN.md` Phase 1 참고.
> 실행 환경: `pjt-2` conda env (`C:\Users\삼성\.conda\envs\pjt-2\python.exe`), Windows.

---

## 0. 절대 규칙 (위반 금지)

1. **수정 가능한 파일은 단 두 개**:
   - `student/solution.py` (OBS_DIM, get_observation, compute_reward)
   - `config/ppo_config.json` (+ sweep용 `config/ppo_*.json` 신규 생성 가능)
2. **`env/` 아래 파일은 절대 수정 금지.** `python verify_env.py`로 SHA256 검증됨.
   `train.py`, `test.py`도 수정하지 않는다 (결과 비교 가능성 유지).
3. **`solution.py` 하나를 문제 1/2가 공유한다.** 레벨별 상수 하드코딩 금지:
   - 인내도: 문제 1은 30, 문제 2는 100 → 반드시 `env.max_patience`로 나눠 정규화
   - `station=='C'`, `size<=4.0` 같은 문제 2 규칙을 관측/보상에 하드코딩 금지
     (C 활용 전략은 per-car size 관측을 통해 정책이 스스로 학습해야 함)
4. **관측 벡터의 모든 원소는 [0, 1] 범위.** 매 스텝 검증되며 벗어나면 즉시 오류.
   `OBS_DIM`과 `get_observation()` 반환 길이가 다르면 즉시 오류.
5. `student/solution-origin.py`는 원본 백업 — 건드리지 않는다.

---

## 1. 문제 정의 요약

- 정비소 3개(A, B, C), 대기 슬롯 최대 3개. 매 틱 행동 1개:
  - `0` = 대기, `1..9` = `waiting_area[(a-1)//3]`을 정비소 `('A','B','C')[(a-1)%3]`에 배치
  - 점유된 정비소에 배치 시도 → 아무 일도 없음 + `event['invalid']=True`
- 총 100대를 처리하면 에피소드 종료 (`MAX_TICKS=50000` 초과 시 truncated)
- **평가 지표(낮을수록 좋음)**: `점수 = 소요 틱 수 + 50 × 이탈 차량 수`
- 차량 속성: `size` 3.0~5.0, `year` 10~25, `damage` 0.0~1.0, 인내도(틱)

| | 문제 1 | 문제 2 |
|---|---|---|
| 도착 간격 | 0~30틱 | 없음 (빈 슬롯 즉시 채워짐) |
| 인내도 | 30 | 100 |
| A | 20~25틱 | 22~23틱 |
| B | 20~25틱 | 21~24틱 |
| C | 20~25틱 | 10~27틱, **size≤4.0이면 50% 감소 (5~13틱)** |
| baseline | 915.4 | 630.4 |

**핵심 인사이트**:
- 문제 1은 세 정비소가 동일 → 점수를 낮출 방법이 없음. **베이스라인 근처(±5%)면 통과** = 파이프라인 검증용.
- 문제 2가 실제 최적화 대상. 이론상 전략: 작은 차(size≤4)를 C로, 큰 차를 A/B로.
  100대 중 절반이 작은 차라 가정하면 C의 처리량이 약 2배가 되어 총 소요 시간 단축 가능.

**env에서 읽을 수 있는 정보** (`get_observation(env)`, `compute_reward(env, event)` 안에서):

```python
env.max_waiting          # 3
env.max_patience         # 레벨의 인내도 최댓값 (30 또는 100) — 정규화에 사용
env.current_time         # 현재 틱
env.waiting_area         # List[Car], 앞이 먼저 온 차. car.size/.year/.damage/.id
env.car_patience[car.id] # 남은 인내도 (틱)
env.repair_status[st]    # st ∈ 'A','B','C'. None=비어있음, (car, 남은틱, 할당틱)=정비중
```

**event 딕트** (`compute_reward`):

```python
event['assigned']  # bool: 이번 스텝에 차량 배치 성공
event['invalid']   # bool: 점유된 정비소에 배치 시도
event['expired']   # int:  이번 스텝에 이탈한 차량 수
event['finished']  # int:  이번 스텝에 정비 완료된 차량 수
event['done']      # bool: 에피소드 종료 (항상 100대 처리 후 발생 → 학습 신호로 무의미)
event['action']    # int:  선택한 행동
```

---

## 2. 실행 명령 레퍼런스

모든 명령은 `C:\hong\project-2\garage_rl`에서 실행. Windows PowerShell이므로 `&&` 대신 `;` 사용.

```powershell
# 환경 검증 (env/ 원본 확인)
python verify_env.py

# 규칙 기반 베이스라인 (solution.py 없이도 동작, 100 episodes)
python env/garage_1.py
python env/garage_2.py

# 학습 (config는 --config로 교체 가능)
python train.py --level 1
python train.py --level 2 --config config/ppo_config.json

# 평가 — 최종 판정 기준 (고정 시드 100 에피소드, 약 5초)
python test.py --level 2 --baseline
python test.py --level 2 --model model_v1/level2/ppo.zip   # 특정 모델 지정

# 학습 곡선
tensorboard --logdir=tensorboard --port=6006
```

- 학습 시간 (CPU 1스레드): 100k steps ≈ 28초, 200k ≈ 1분, 500k ≈ 2분 13초
- train.py 콘솔 출력의 `평균보상`(설계한 보상)과 `점수추정`(실제 지표, 5 에피소드 실측)을
  **반드시 같이 볼 것**. 평균보상↑인데 점수추정↓하지 않으면 보상 함수가 목표와 어긋난 것.
- tensorboard 지표: `eval/score`(결정적 5ep), `eval/gain_pct`, `rollout/score`, `rollout/removed`

---

## 3. 단계별 구현 지시

각 단계 완료 시 §4의 실험 기록 규칙에 따라 `EXPERIMENTS.md`에 기록한다.

### Step 1 — Baseline 확인 (코드 수정 없음)

```powershell
python verify_env.py
python env/garage_1.py
python env/garage_2.py
```

- 기대값: Level 1 ≈ 915.4, Level 2 ≈ 630.4. 실측치를 EXPERIMENTS.md에 기록.

### Step 2 — Observation 확장 (`student/solution.py`)

`OBS_DIM = 21`로 변경하고 `get_observation()`을 아래 스펙대로 구현:

```
인덱스   내용                          계산식
------  ---------------------------  ------------------------------------------
0-2     slot_occupied[i]             1.0 if i < len(env.waiting_area) else 0.0
3-5     station_busy[A,B,C]          1.0 if env.repair_status[st] is not None else 0.0
6-8     station_progress[A,B,C]      (할당틱-남은틱)/할당틱, idle이면 0.0
                                     status = env.repair_status[st]
                                     → (status[2]-status[1])/status[2]
9-20    per_car[i] × 3 slots (슬롯당 4개, 빈 슬롯은 전부 0.0):
        size_norm                    (car.size - 3.0) / 2.0
        year_norm                    (car.year - 10.0) / 15.0
        damage                       car.damage  (이미 [0,1])
        patience_norm                env.car_patience[car.id] / env.max_patience
```

구현 시 주의:
- `dtype=np.float32`로 반환
- progress 계산에서 할당틱은 항상 ≥1 (`max(1, base)` 보장됨)이라 0-division 없음
- patience_norm은 이론상 [0,1]이나 안전하게 `min(1.0, ...)` clip 권장
- `size_norm`이 문제 2의 핵심 feature (C 정비소 50% 감소 조건 size≤4.0 ↔ size_norm≤0.5를
  정책이 학습으로 발견)

**검증**: `python train.py --level 1` (기본 config, 100 steps)이 오류 없이 돌면 관측 형식 OK.

### Step 3 — Reward 재설계 (`student/solution.py`)

**설계 원칙**: 점수 = 틱 + 50×이탈. 보상을 점수의 음수 스케일과 정렬한다
(`틱당 페널티 : 이탈 페널티 = 1 : 50` 비율 유지). 기본 구현:

```python
# 파일 상단에 튜닝 상수로 분리 (sweep 시 이 값만 수정)
R_SCALE       = 0.01   # 전체 보상 스케일
R_EXPIRE_MULT = 50.0   # 이탈 가중 배율 (기본 = 지표와 동일 비율)
R_INVALID     = 0.02   # 헛발질 억제 (소량; = R_SCALE * 2)
R_FINISH      = 0.0    # 정비 완료 보너스 (기본 0, sweep에서 실험)

def compute_reward(env, event) -> float:
    reward = -R_SCALE                                   # 매 틱 시간 압박
    reward -= event['expired'] * R_SCALE * R_EXPIRE_MULT  # 이탈 = 점수 +50에 대응
    if event['invalid']:
        reward -= R_INVALID
    reward += event['finished'] * R_FINISH
    return reward
```

**하지 말 것**:
- `event['done']` 보너스 — 모든 에피소드가 100대 처리 후 종료라 상수 신호, 무의미
- `event['assigned']` 보너스 — "아무 데나 빨리 넣기"를 유도해 문제 2에서
  큰 차를 C에 넣는 악수를 강화할 수 있음 (보상 해킹). 사용하지 않는다.
- 원본의 `-100` 같은 극단값 — 보상 크기가 들쭉날쭉하면 학습 불안정

### Step 4 — 기본 config 갱신 (`config/ppo_config.json`)

```json
{
  "seed": 0,
  "learning_rate": 0.0003,
  "n_steps": 2048,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.995,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "ent_coef": 0.01,
  "total_timesteps": 200000,
  "verbose": 0,
  "model_dir": "model",
  "tensorboard_dir": "tensorboard"
}
```

변경점: `total_timesteps` 100→200000, `ent_coef` 0.0→0.01, `gamma` 0.99→0.995
(에피소드 600~2000틱 + 인내도 100틱 신호 특성상 유효 horizon 확장).

### Step 5 — 파이프라인 검증 (Level 1)

```powershell
python train.py --level 1
python test.py --level 1 --baseline
```

**통과 기준**: test 점수가 베이스라인 915.4의 ±5% 이내 (문제 1은 이길 수 없음이 정상).
- 크게 나쁘면(>+10%): 보상 신호 문제. `rollout/removed`가 0에 수렴하는지 확인.
  이탈이 많으면 `R_EXPIRE_MULT` 확인, 대기만 하면(에피소드 길이 폭증) 시간 페널티 확인.

### Step 6 — Level 2 학습 + per-car feature ablation

```powershell
python train.py --level 2
python test.py --level 2 --baseline
```

**Ablation (README 요구사항)**: per-car feature(인덱스 9-20)를 0으로 고정한 버전
(OBS_DIM은 21 유지, 값만 0)과 정식 버전을 각각 학습해 test 점수 비교.
per-car 있는 쪽이 유의하게 좋아야 정상 (없으면 C 활용 전략 학습 불가).
비교 후 정식 버전으로 되돌린다.

**목표**: baseline 630.4보다 낮은 점수. 학습된 정책 확인 방법:

```powershell
# 디버그 롤아웃으로 "작은 차 → C" 배치가 실제로 일어나는지 관찰
python - <<스크립트 내용>>
```

```python
# scripts/inspect_policy.py 로 저장해 사용 (신규 파일 생성 허용)
from stable_baselines3 import PPO
from env.garage_2 import GarageEnv_2
env = GarageEnv_2(debug=True, seed=123)
model = PPO.load("model/level2/ppo.zip")
obs, _ = env.reset(seed=123)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, info = env.step(int(action))
    if term or trunc:
        break
print(info)
```

디버그 출력의 `-> Assigned Car(...)` 라인에서 size와 station을 대조:
size≤4.0 차량이 주로 C에, size>4.0이 A/B에 가는지 확인.

**안 될 때 진단 순서**:
1. 관측 부족? → Step 2 구현이 스펙대로인지 재확인
2. 보상 신호 부족? → `R_FINISH=0.005` 추가 실험, `R_EXPIRE_MULT` 조정
3. 탐색 부족? → `ent_coef` 0.05로 상향, `total_timesteps` 500000으로 상향

### Step 7 — Reward sweep

`compute_reward`의 상수를 바꿔가며 Level 2 test 점수 비교. **한 번에 한 축만** 변경:

| 축 | 후보 | 고정 조건 |
|---|---|---|
| `R_SCALE` | 0.005, **0.01**, 0.02 | 비율 유지 (EXPIRE_MULT=50) |
| `R_EXPIRE_MULT` | 25, **50**, 100 | SCALE=best |
| `R_FINISH` | **0.0**, 0.005, 0.01 | 나머지 best |

(굵은 글씨 = 기본값. 각 run: `total_timesteps=200000`, seed=0, test.py로 판정)

### Step 8 — Hyperparameter sweep

전수 grid 금지. **1-factor 순차 sweep** — 축 하나씩 best 채택 후 다음 축:

| 순서 | 축 | 후보 |
|---|---|---|
| 1 | `gamma` | 0.99, **0.995**, 0.999 |
| 2 | `learning_rate` | 1e-4, **3e-4**, 1e-3 |
| 3 | `ent_coef` | 0.0, **0.01**, 0.05 |
| 4 | `n_steps` | 1024, **2048**, 4096 |
| 5 (여유 시) | `batch_size` / `n_epochs` | 32/64/128, 5/10/20 |

- 각 후보 `total_timesteps=200000`으로 비교, 각 축 best만 채택
- sweep용 config는 `config/ppo_g0.99.json` 식으로 별도 파일 생성
- **병렬 실행 시 config마다 `model_dir`/`tensorboard_dir`을 반드시 분리**
  (예: `"model_dir": "model_g0.99", "tensorboard_dir": "tensorboard/g0.99"`).
  분리하지 않으면 `model/level2/ppo.zip`이 서로 덮어씀.
- **주의**: `solution.py`(관측/보상)가 다른 실험은 파일 공유 때문에 동시 실행 불가.
  하이퍼파라미터 sweep만 병렬화 가능.

### Step 9 — 최종 확인

1. best 조합으로 `total_timesteps=500000`, `seed` ∈ {0, 1, 2} 3회 학습
   (seed는 config의 `"seed"` 필드 — 바꾸지 않으면 3회가 동일 결과)
2. 각 run을 `python test.py --level 2 --model <경로> --baseline`으로 평가,
   평균±표준편차 기록
3. best 모델을 기본 경로 `model/level2/ppo.zip`에 배치 (백업 필수 — train 재실행 시 덮어씀)
4. Level 1도 최종 solution.py로 재학습·재평가해 베이스라인 근처인지 확인
   (공유 파일이므로 Level 2 최적화가 Level 1을 망가뜨리지 않았는지 검증)

---

## 4. 실험 기록 규칙

`garage_rl/EXPERIMENTS.md`를 생성하고 모든 run을 다음 표 형식으로 축적:

```markdown
| # | date | level | obs | reward params (SCALE/EXP_MULT/INVALID/FINISH) | config 요약 (lr/gamma/ent/steps/total) | seed | test score | vs baseline |
|---|------|-------|-----|------|------|------|-----------|-------------|
| 1 | 08-10 | 2 | v1(21d) | 0.01/50/0.02/0 | 3e-4/0.995/0.01/2048/200k | 0 | 588.2 | +6.7% |
```

- **test score는 반드시 `test.py`(고정 시드 100 에피소드) 값.** train 로그의
  `점수추정`(5 에피소드)은 참고용일 뿐 기록 기준이 아니다.
- `vs baseline` = `(baseline - score) / baseline * 100` (+ 가 개선)
- 관측 구성을 바꾸면 `obs` 열에 버전 태그(v1, v2...)를 붙이고 하단에 구성 설명 추가

---

## 5. 완료 기준 (Definition of Done)

- [ ] `python verify_env.py` 통과 (env/ 무변경)
- [ ] Level 1 test 점수가 915.4 ± 5% 이내
- [ ] Level 2 test 점수가 **630.4 미만** (목표: 최대한 낮게)
- [ ] per-car feature ablation 결과가 EXPERIMENTS.md에 기록됨
- [ ] 관측/보상에 레벨별 하드코딩 없음 (max_patience 사용, 'C'/4.0 하드코딩 없음)
- [ ] 최종 seed 3회 평균±표준편차 기록
- [ ] best 모델이 `model/level1/ppo.zip`, `model/level2/ppo.zip`에 존재
- [ ] `EXPERIMENTS.md`에 전체 run 이력 존재

---

## 6. [최우선] Level 3 — 최종 제출 문제 (마감 당일 17:20)

> **문제 2 대신 문제 3이 최종 채점 대상으로 변경됨.** §3의 Step 순서 대신
> 아래 time-box 계획을 최우선으로 실행한다. §0 절대 규칙은 동일하게 적용.

### 6.1 문제 3 스펙 (`env/garage_3.py`)

| 항목 | 값 |
|---|---|
| 도착 | 간격 없음 (매 틱, 빈 슬롯 즉시 채워짐) |
| 인내도 | **30틱** (문제 2는 100이었음 — `env.max_patience` 정규화 필수) |
| A | `randint(15,19) × (1 + 1.5×damage)` — 손상도 0이면 15~19, 1이면 38~48 (연속) |
| B | `randint(26,36)`, **연식 ≥ 20이면 ×0.5** (13~18) |
| C | `randint(26,36)`, **크기 ≤ 4.0이면 ×0.35** (9~13) |
| baseline | **1490.8** (100대 중 16대 이탈 = 이탈 벌점만 800점) |

**판정** (`test.py --level 3`, 100 에피소드 평균):
- **≤ 1300: Pass** / **≤ 1050: Distinguished** / 참고 최고 기록: 950

**전략 인사이트**:
- 이탈 벌점(16×50=800)이 baseline 점수의 절반 이상 → **이탈을 줄이는 것이 최대 레버**.
- 조건 미충족 시 B/C는 26~36틱 > 인내도 30틱 → 잘못 배치하면 그 정비소가 다른 차의
  이탈을 유발. 매칭이 문제 2보다 훨씬 중요:
  - 크기≤4.0 → C (9~13틱), 연식≥20 → B (13~18틱), 저손상 → A
  - 어느 조건도 못 맞추는 차(크고·새것·고손상)는 A가 차악
- 기존 21차원 관측(size/year/damage/patience per-car)이 세 조건을 모두 커버.
  이후 **§6.4에서 27차원(v2)으로 확장됨** — 현재 solution.py는 v2.
- 인내도 30틱 → per_car `patience_norm`이 급박함 신호로 문제 2보다 중요.

### 6.2 Time-box 실행 계획 (총 ~2시간 + 버퍼)

**T+0:00 ~ 0:10 — 검증 및 baseline**
```powershell
python verify_env.py
python env/garage_3.py          # baseline ≈ 1490.8 실측 확인
```
- `train.py`는 `env/garage_3.py` 존재 시 `--level 3`을 자동 지원함 (try-import 확인됨).
- `student/solution.py`가 아직 §3 Step 2~3 스펙(21차원 obs + 정렬 보상)이 아니면 **지금 구현**.

**T+0:10 ~ 0:25 — 1차 학습 (Pass 확보)**
```powershell
python train.py --level 3      # 기본 config: 200k steps ≈ 1분
python test.py --level 3 --baseline
```
- 목표: **≤1300 (Pass) 확인**. 이 시점의 모델을 즉시 백업 → 제출 최저선 확보.
- train 로그의 `rollout/removed` 추이 관찰: 16 미만으로 내려가는지가 핵심.

**T+0:25 ~ 1:25 — 병렬 sweep (Distinguished 도전)**

config 파일을 만들어 Agent Manager로 병렬 실행 (**`model_dir`/`tensorboard_dir` 반드시 분리**):

| config | 변경점 | 가설 |
|---|---|---|
| `ppo_l3_a.json` | `total_timesteps=500000` (나머지 기본) | 학습량 증가만으로 개선 |
| `ppo_l3_b.json` | 500k + `gamma=0.999` | 이탈의 지연 신호 포착 |
| `ppo_l3_c.json` | 500k + `ent_coef=0.05` | "대기" 전략 탐색 (맞는 정비소가 빌 때까지 홀드) |
| `ppo_l3_d.json` | 500k + `R_EXPIRE_MULT=100` (solution.py 상수) | 이탈 회피 강조 |

- 주의: `R_EXPIRE_MULT` 변경(d)은 `solution.py` 수정이므로 **a~c와 동시 실행 불가**.
  a~c 병렬 → 결과 확인 → 필요 시 d를 별도로.
- 각 run 평가: `python test.py --level 3 --model model_l3_a/level3/ppo.zip --baseline`
- 여유가 있으면 best 축 위에 `learning_rate=1e-4`, `n_steps=4096` 1~2개 추가.

**T+1:25 ~ 1:45 — 최종 모델 확정**
- best config로 seed {0,1,2} 3회 학습(병렬 가능), test.py로 각각 평가
- **best 모델을 `model/level3/ppo.zip`에 배치** (제출 요구 경로 — 정확히 이 경로여야 함)
- 최종 확인: `python test.py --level 3 --baseline` (기본 경로로 재검)

**T+1:45 ~ 2:10 — 리포트 (A4 1장, PDF)**

포함 내용 (간략히):
1. 실행 방법: `python train.py --level 3` → `python test.py --level 3 --baseline`
2. 관측 설계: 21차원 구성 요약 (slot/busy/progress + per-car 4속성, 정규화 방식)
3. 보상 설계: 점수와 1:50 정렬 원칙 (`-SCALE`/틱, `-SCALE×50`/이탈), done/assigned 보너스 배제 이유
4. 결과: baseline 1490.8 vs 최종 점수, 이탈 수 변화, 학습 곡선 캡처 1장(tensorboard `eval/score`)
5. 하이퍼파라미터 최종값 표

**T+2:10 ~ 2:20 — 제출**
- `student/solution.py`, `model/level3/ppo.zip`, 리포트 PDF → geonholeem@imo.snu.ac.kr

### 6.4 관측 v2 (27차원) — 파생 피처 확장 [구현 완료]

`solution.py`에 반영된 현재 스펙. **v1(21d) 모델과 호환 불가** — v2 적용 후에는
v1으로 학습된 `ppo.zip`을 `test.py`로 평가할 수 없다 (OBS_DIM 불일치 오류).
이미 돌고 있는 v1 학습 프로세스는 시작 시점에 import 완료라 정상 종료되지만,
그 산출 모델을 평가하려면 solution.py를 v1로 되돌려야 한다.

```
인덱스    내용                 계산식
-------  ------------------  ------------------------------------------
0-2      slot_occupied       기존과 동일
3-5      station_busy        기존과 동일
6-8      station_progress    기존과 동일
9-11     station_rem_ticks   min(1, 남은틱/T_NORM), idle=0   [신규]
12-23    per_car × 3         기존과 동일 (size/year/damage/patience_norm)
24-26    car_patience_abs    min(1, 남은인내도틱/T_NORM)     [신규]
```

`T_NORM = 50.0` — 레벨 규칙이 아닌 시간 스케일링 상수 (초과분 1.0 clip).

**채택한 파생 피처와 근거**:
- `station_rem_ticks`: 진행률(6-8)은 비율이라 "몇 틱 뒤에 비는지"라는 절대 정보가
  없음. 대기(홀드) 전략 판단에 필수적인 **신규 정보**.
- `car_patience_abs`: 기존 `patience_norm`은 `max_patience` 상대값이라 레벨마다
  단위가 다르고 station_rem_ticks와 비교 불가. 같은 `틱/T_NORM` 단위로 맞춰
  "정비소가 비기 전에 차가 떠나는가"를 신경망이 직접 비교 가능.

**기각한 제안과 근거** (재제안 금지):
- `patience_urgency (1-p)`, `car_advantages (1-size_norm)`, `b/a_advantage`:
  기존 feature의 **선형 변환** — 2×64 MLP가 즉시 학습 가능, 정보량 0.
- `min_patience_norm`, `waiting_count_norm`: 기존 3개 값의 min/sum, 중복.
- `slot_station_fit_matrix` (9d): size↔C 같은 레벨별 매칭 규칙을 관측에
  주입하는 것 — §0 규칙 3(레벨 하드코딩 금지) 위반 소지.

**실행 지시 (다른 agent)**:
1. v1 sweep(l3_a/b/c) 종료 대기 → 각 로그(`train_l3_*.log`)의 최종 점수추정만 기록
   (test.py 평가는 OBS_DIM 불일치로 불가 — 점수추정으로 비교)
2. v2로 재학습: 기본 config 축 그대로 500k 3종 병렬
   (`model_dir`을 `model_l3v2_a` 식으로 분리한 config 신규 생성)
3. v1 로그의 best 축과 v2 결과를 비교해 최종 채택, 이후 §6.2의 T+1:25 단계로 복귀

### 6.5 시간 부족 시 우선순위 (컷라인)

1. **최우선**: T+0:25 시점의 Pass 모델 + 최소 리포트라도 제출 가능한 상태 유지
2. sweep은 a(500k)만이라도 — 200k→500k 만으로 개선되는 경우가 많음
3. seed 3회 반복은 생략 가능 (best 단일 run 제출)
4. 리포트는 절대 생략 불가 (제출물 3개 중 하나) — 늦어도 T+1:50에는 작성 시작
