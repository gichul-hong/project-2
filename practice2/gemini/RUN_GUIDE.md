# Practice 2 실행 가이드 (Gemini 버전)

> 3단계 순서로 실행: 자연스러운 걷기 → 장애물 통과 → 최적화

---

## 사전 준비

```bash
# conda 환경 활성화
conda activate rl

# gemini 코드를 practice2 루트로 복사 (실행은 practice2/ 에서)
cd /path/to/project-2/practice2
cp gemini/custom_walker2d.py ./custom_walker2d.py
cp gemini/learning.py ./learning.py
cp gemini/eval_gait.py ./eval_gait.py
```

> **⚠️ 중요**: `learning.py`와 `custom_walker2d.py`는 `practice2/` 루트에서 실행해야 합니다.
> `render.py`와 `asset/` 디렉토리가 같은 위치에 있어야 하기 때문입니다.

---

## Phase 1: 자연스러운 걷기 학습 (Task 1 + 2)

### 1-1. 학습 실행

```bash
cd /path/to/project-2/practice2
python learning.py
```

- 8개 병렬 환경 (`SubprocVecEnv`)
- TensorBoard 로그: `./logs/`
- 체크포인트: `./checkpoints/walker_model/` (10,000 step마다 저장)
- **VecNormalize** 통계도 체크포인트와 함께 자동 저장

### 1-2. 학습 모니터링

```bash
# 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=./logs/
# → http://localhost:6006 에서 확인
```

**주요 관찰 지표**:
- `ep_rew_mean`: 에피소드 평균 보상 (상승 추세 확인)
- `ep_len_mean`: 에피소드 평균 길이 (1000에 가까워야 = 넘어지지 않음)

### 1-3. 걷기 품질 평가 (hopping 진단)

```bash
# 예시: 1,000,000 스텝 체크포인트 평가
python eval_gait.py \
    --model ./checkpoints/walker_model/walker_model_1000000_steps.zip \
    --vecnormalize ./checkpoints/walker_model/walker_model_vecnormalize_1000000_steps.pkl \
    --episodes 10
```

**판단 기준**:
| 지표 | 값 | 판정 |
|------|-----|------|
| `balance_ratio` | > 0.7 | ✅ 양발 골고루 사용 |
| `balance_ratio` | 0.4 ~ 0.7 | ⚠️ 불균형 경향 |
| `balance_ratio` | < 0.4 | ❌ hopping |
| `mean_sep` (thigh separation) | > 0.3 | ✅ 가위질 동작 |
| `avg_x` | > 40.0 | ✅ 그리드 끝 도달 |

### 1-4. 시각적 확인

```bash
# 렌더링 확인
python render.py \
    --model ./checkpoints/walker_model/walker_model_BEST.zip

# 녹화 모드 (R키: 녹화 시작/정지, Q키: 종료)
python render.py \
    --model ./checkpoints/walker_model/walker_model_BEST.zip \
    --record
```

> **VecNormalize 주의**: `render.py`가 자동으로 `_vecnormalize_*.pkl`을 찾습니다.
> 같은 디렉토리에 `.pkl` 파일이 있어야 정상 작동합니다.

### 1-5. Phase 1 완료 기준

- [ ] `balance_ratio > 0.7` (양발 균형)
- [ ] `avg_x > 40.0` (그리드 끝 도달)
- [ ] 시각적으로 깡총거리지 않음

**목표 학습량**: 4M ~ 6M steps (약 1~2시간, 환경에 따라 다름)

---

## Phase 2: 장애물 통과 학습 (Task 3)

### 2-1. Curriculum Learning (전이 학습)

> **반드시 Phase 1의 best 모델에서 시작**합니다. (flat ground 걷기를 보존)

```bash
# Phase 1에서 best 모델/vecnormalize 경로 확인 후 실행
python learning.py \
    --bump_practice \
    --init_model ./checkpoints/walker_model/walker_model_XXXXXXX_steps.zip \
    --init_vecnormalize ./checkpoints/walker_model/walker_model_vecnormalize_XXXXXXX_steps.pkl
```

- 체크포인트: `./checkpoints/bump_practice/`

### 2-2. 장애물 통과 평가

```bash
python eval_gait.py \
    --model ./checkpoints/bump_practice/walker_model_XXXXXXX_steps.zip \
    --vecnormalize ./checkpoints/bump_practice/walker_model_vecnormalize_XXXXXXX_steps.pkl \
    --bump_practice \
    --episodes 20
```

**판단 기준**:
| 지표 | 값 | 판정 |
|------|-----|------|
| `bump1 pass` | > 90% (18/20) | ✅ |
| `bump2 pass` | > 80% (16/20) | ✅ |
| `avg_x` | > 12.0 | ✅ bump2 너머 도달 |

### 2-3. 시각적 확인

```bash
python render.py \
    --model ./checkpoints/bump_practice/walker_model_BEST.zip \
    --bump_practice
```

**목표 학습량**: 5M ~ 8M steps

---

## Phase 3: 최적화 및 HP Sweep

Phase 1, 2에서 만족스럽지 않은 경우에만 수행합니다.

### 3-1. ent_coef sweep

`learning.py`의 `ent_coef`를 변경하여 비교:

```
ent_coef=0.005  →  탐색 적음 (hopping에 빠질 수 있음)
ent_coef=0.01   →  기본값 (권장)
ent_coef=0.02   →  탐색 많음 (bump에 유리할 수 있음)
```

### 3-2. forward_vel 가중치 sweep

`custom_walker2d.py`의 `1.0 * forward_vel`을 변경하여 비교:

```
0.8  →  hopping 억제 극대화 (느리지만 안정적)
1.0  →  기본값 (권장)
1.2  →  속도 중시 (hopping 위험 증가)
```

> **주의**: 관측을 변경하면 이전 모델은 폐기됩니다! (LESSONS_LEARNED §8)

---

## 제출물 준비

### 최종 모델 선정

```bash
# Task 1+2: 가장 balance_ratio 높고 avg_x > 40인 flat ground 모델
# Task 3: bump2 통과율 가장 높은 bump_practice 모델

# 예시: 최종 평가 (20 에피소드)
python eval_gait.py \
    --model ./checkpoints/walker_model/walker_model_BEST.zip \
    --vecnormalize ./checkpoints/walker_model/walker_model_vecnormalize_BEST.pkl \
    --episodes 20

python eval_gait.py \
    --model ./checkpoints/bump_practice/walker_model_BEST.zip \
    --vecnormalize ./checkpoints/bump_practice/walker_model_vecnormalize_BEST.pkl \
    --bump_practice \
    --episodes 20
```

### 제출 파일

수정한 파일 2개:
1. `custom_walker2d.py` — 관측/보상/종료 조건
2. `learning.py` — 학습 하이퍼파라미터

---

## 파일 구조

```
practice2/
├── custom_walker2d.py    ← 제출용 (gemini/ 에서 복사)
├── learning.py           ← 제출용 (gemini/ 에서 복사)
├── render.py             ← 수정 불가
├── eval_gait.py          ← 평가 도구 (gemini/ 에서 복사)
├── asset/
│   └── custom_walker2d_bumps_practice.xml
├── gemini/               ← 원본 소스
│   ├── custom_walker2d.py
│   ├── learning.py
│   ├── eval_gait.py
│   ├── MASTER_PROMPT.md
│   └── RUN_GUIDE.md      ← 이 파일
├── checkpoints/
│   ├── walker_model/     ← Phase 1 체크포인트
│   └── bump_practice/    ← Phase 2 체크포인트
└── logs/                 ← TensorBoard 로그
```

---

## 트러블슈팅

### SubprocVecEnv가 hang 되는 경우 (Windows)
`learning.py`에서 `SubprocVecEnv` → `DummyVecEnv`로 변경:
```python
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([make_env(...) for _ in range(N_ENVS)])
```

### VecNormalize .pkl 파일을 찾을 수 없는 경우
체크포인트 디렉토리에서 확인:
```bash
ls ./checkpoints/walker_model/walker_model_vecnormalize_*.pkl
```
`save_vecnormalize=True`가 `CheckpointCallback`에 설정되어 있어야 합니다.

### 학습 속도 (steps/s) 확인
학습 시작 후 첫 10줄의 로그에서 `fps` 값을 확인하세요.
보통 8 env 기준 300~600 fps입니다.
