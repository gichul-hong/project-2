# Practice 2 Work Plan — DeepSeek Session

> 기준: LESSONS_LEARNED.md §7 체크리스트 + §8 실패 금지 항목
> 강의자료: `LECTURE2.md`

---

## 0. 관련 파일

| 파일 | 역할 | 수정 가능? |
|------|------|-----------|
| `practice2/custom_walker2d.py` | 관측/보상/종료조건 정의 | ✅ |
| `practice2/learning.py` | PPO 학습 스크립트 (HP, 병렬) | ✅ |
| `practice2/render.py` | 모델 시뮬레이션/녹화 | ❌ (사용만) |
| `practice2/asset/*.xml` | 환경 지형 (bump) | ❌ (검증 대상) |
| `LECTURE2.md` | 강의 내용 요약 | 참조 전용 |

**제약사항 확인 필요**:
- [ ] `custom_walker2d.py` 하나가 Task 1/2/3 공용인가? → 공용이면 관측/보상 하드코딩 금지
- [ ] 제출 경로/포맷 확인 (e.g., `checkpoints/{folder}/ppo_custom_walker2d_parallel.zip`)

---

## 1. 평가 지표 파악 (가장 먼저)

### Task 1: Walker2D 걷게 하기
- **목표**: 20초 이내에 그리드 끝에 도달
- **추정 지표**: x좌표 최종값 or 도달 시간 or forward 속도 적분값
- **확인 필요**: test.py가 있는지, 점수 계산 방식을 무엇인지

### Task 2: Walker2D 자연스럽게 걷기
- **목표**: 자연스러운 보행 동작으로 20초 이내 그리드 끝 도달
- **추정**: 속도 + 자연스러움(대칭성, 상체 안정성) 정량화 점수

### Task 3: 2개 Bump 통과
- **목표**: 20초 이내에 bump 2개 통과
- **추정**: 넘은 bump 수 + 시간 기반 점수
- Bump #1: (6.0, 0, 0), size (0.3, 2.0, 0.2)
- Bump #2: (10.0, 0, 0), size (0.6, 2.0, 0.45)

---

## 2. 시간 예산 (time-box)

| 단계 | 예상 시간 | 비고 |
|------|----------|------|
| 환경 설정 확인 + baseline 학습 | 20분 | conda env `rl` 사용, 초기 steps/s 측정 |
| Task 1: 기본 걷기 학습 + HP sweep | 60분 | Fallback 우선 확보 |
| Task 2: 자연스러움 보상 설계 + 학습 | 60분 | best of task1 기반 |
| Task 3: bump 학습 | 60분 | best of task1/2 transfer 가능성 확인 |
| 리포트 작성 | 30분 | 학습 중 병행 |
| **합계** | **~3.5시간** | |

**원칙**: 각 Task별 Fallback(Pass 기준) 먼저 확보 → 시간 남으면 최적화

---

## 3. 관측(Observation) 설계 방향

### 기본 17차원 관측 분석
- qpos(8): rootx, rootz, rooty, thigh, leg, foot, thigh_l, leg_l, foot_l
- qvel(9): velocities of above

### Task 1 추가 후보
- 현재 forward progress가 관측에 포함 (rootx) → 충분할 가능성
- 프레임스킵 10 → rootx 변화량이 관측만으로 visible

### Task 2 추가 후보 (§2.2 원칙: 선형조합 불가한 새 정보만)
- `torso_angular_vel` (이미 rooty vel로 존재)
- `torso_z_deviation = |obs[1] - 1.25|` → 선형변환 → **기각**
- `leg_symmetry = |obs[3] - obs[6]| + |obs[4] - obs[7]|` → 새 정보이나, reward로 주는 게 더 효과적

### Task 3 추가 후보
- bump 상대 위치: `dist_to_bump1`, `dist_to_bump2`
- `height_above_ground` → bump 넘을 때 높이 변화 감지용
- 충분한지 실측 후 판단

---

## 4. 보상(Reward) 설계 방향

### 기본 보상
```
reward = healthy_reward(1) + forward_reward - ctrl_cost
```

### Task 1
- forward_reward weight 조정으로 충분할 가능성 (기본이면 도달)
- 지표와의 정렬 확인 후 결정

### Task 2 (자연스러움)
PPT Solution 참고:
1. **Torso 안정성**: `-|torso_angular_vel|` 페널티 → 상체 뒤뚱임 억제
2. **주기적 대칭성**: L2 loss 꼴 `-|obs_right(t) - obs_left(t+period)|` → 왼/오른 다리 반주기 차이 보상
3. **대칭 action**: actor loss에 symmetry term 추가 (PPO 내부수정 필요, 복잡도↑)

→ Solution 1 + 2를 reward term으로 구현 (Solution 3은 복잡도 대비 효과 미지수)

### Task 3 (bump)
- bump 통과 보너스: rootx가 bump 위치 통과 시 양수 reward
- 넘어짐 방지: torso 각도 페널티 강화

**§8 주의**: 극단값(-100) 금지, ~O(1) 스케일 유지

---

## 5. 하이퍼파라미터 전략

### Baseline (learning.py 기본값)
```
net_arch: pi=[128,64,64], vf=[128,64,64]
log_std_init: -1.0
learning_rate: 0.0001
N_ENVS: 4
total_timesteps: 10^10 (사실상 무한 → 필요시 중단)
```

### Sweep 순서 (§4.3)
1. **gamma** (0.99 → 0.995 시도, §4.2)
2. **ent_coef** (0.0, 0.002, 0.01, 0.05 — 관측 rich 여부로 결정, §4.1)
3. **learning_rate** (1e-4 → 3e-4, 5e-5)
4. **n_steps / batch_size** (필요시)

### 초기 실측
- 짧은 학습(50k step)으로 steps/s 측정 → 전체 예산 보정
- 3 seed minimum → best-of-3 채택 (§4.4)

---

## 6. 실행 인프라

### Conda 환경
```bash
conda activate rl
cd C:\hong\project-2\practice2
```

### 학습 명령어
```bash
# Task 1: 기본 걷기
python learning.py

# Task 2: 자연스러운 걷기 (같은 스크립트, custom_walker2d.py 수정)
python learning.py

# Task 3: bump
python learning.py --bump_practice
```

### 병렬 실행 주의 (§5.1)
- `checkpoints/` 경로 분리 확인 (기본 코드에서 bump_practice / walker_model 분기)
- 실험별 model_dir 다르게 or `ppo_custom_walker2d_parallel.zip` 덮어쓰기 방지

---

## 7. 실험 추적

- `EXPERIMENTS.md`에 표로 축적 (§6.1)
- 각 run: obs 버전 / reward 파라미터 / config / seed / test score / x_distance / vs baseline
- **test.py** 확인 필요 — 있다면 그 값만 공식 기록

---

## 8. 작업 순서

1. [ ] `conda activate rl` + `pip list`로 패키지 확인 (sb3, gymnasium, mujoco)
2. [ ] `python learning.py` 10k step 짧게 돌려서 steps/s 실측
3. [ ] Task 1: **baseline 평가** — pretrained 모델 or 기본 학습으로 도달 거리 확인
4. [ ] Task 1: forward_reward weight sweep으로 Pass 확보 (Fallback)
5. [ ] Task 2: torso 안정성 + 주기적 대칭성 reward 구현
6. [ ] Task 2: ent_coef sweep (관측 rich 정도에 따라 결정)
7. [ ] Task 3: bump 환경으로 전환 + transfer from Task 1/2
8. [ ] Task 3: bump 위치 정보 관측 추가 검토
9. [ ] 전 Task 평가 + best model 선정
10. [ ] 리포트 작성 (실험 중 병행)