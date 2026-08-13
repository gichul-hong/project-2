# Problem 3: 병렬 학습 전략 — Speed Booster vs Stall Buster

## 현재 상태 (Baseline)

| 항목 | 값 |
|------|-----|
| 모델 | `checkpoints/bump_challenge/walker_model_2600000_steps.zip` |
| 점수 | **31.77 m** (evaluate.py) |
| 생존 | 1000/1000 스텝, 낙상 없음 |
| 평균 속도 | 1.59 m/s (평지 구간은 2.7~3.5 m/s 도달) |
| 골인 지점 | 101.9 m |

### diag_run.py 트레이스 — 진짜 병목

| 구간 | 관찰 |
|------|------|
| x=0~25 | 2.7~3.5 m/s로 순조롭게 주파 |
| **x≈26.4 (bump8, h=1.0)** | **step 500~930 (약 8.6초) 정체** — 0.5→1.0m 2단 계단 등반 실패 |
| x=28~31.8 | 돌파 후 3.3 m/s 재주행, 시간 종료 |

**핵심 진단:** 평지 속도(3.5 m/s)는 이미 충분하다. bump8 정체 8.6초만 해소하면 같은 속도로 **25m 이상 추가 주행 → 55~60m**. 병목은 속도가 아니라 **계단 등반 능력**이다.
- 그러나 평지 속도를 3.5→5.0 m/s로 올리면, 범프 사이 이동 시간이 줄어 더 먼 거리에 도달할 수 있고, 접근 속도 증가로 운동량으로 계단을 극복할 가능성도 있다.

두 가설을 **병렬로 동시에 검증**한다.

---

## 전략: 2-Run 병렬 학습

동일한 2.6M 체크포인트에서 서로 다른 보상 함수로 이어 학습한다. 두 Run의 체크포인트를 `score_monitor.py`로 동시에 평가하며, 더 나은 쪽을 승자로 선택한다.

| | Run A: Speed Booster | Run B: Stall Buster |
|---|---|---|
| **목표** | 평지 속도 상승으로 거리 증가 | bump8 계단 정체 해소로 거리 증가 |
| **가설** | 속도가 오르면 범프 사이 이동 시간 감소 + 운동량으로 계단 극복 | 정체 중 healthy 보상 착취를 막고 계단 보상을 강화하면 등반 학습 |
| **체크포인트 경로** | `checkpoints/bump_challenge_v10_speed/` | `checkpoints/bump_challenge_v10_stall/` |

---

## 공통 변경사항 (양 Run 모두 적용)

두 Run 모두 아래 변경을 공통으로 적용한다.

### 1. MIN_TORSO_Z 완화 (0.9 → 0.7)

```python
MIN_TORSO_Z = 0.7  # 0.9 → 0.7
```

**이유:** 학습 중 terminated 기준(z < 0.9)과 평가(evaluate.py)의 낙상 기준(z < 0.5) 사이에 OOD 갭이 있다. 학습 중 z∈[0.5, 0.9) 영역을 경험하지 못하면, 속도 증가로 상체 진동이 커질 때 OOD 액션으로 낙상 위험이 생긴다. 0.7로 낮춰 0.5~0.7 사이 0.2 마진을 확보한다.

### 2. height_factor 연동

```python
height_factor = np.clip((torso_z - (MIN_TORSO_Z + 0.1)) / 0.2, 0.0, 1.0)
# → np.clip((torso_z - 0.8) / 0.2, 0.0, 1.0)
```

**이유:** MIN_TORSO_Z 변경 시 height_factor도 자동 연동. 상수 하드코딩 방지.

### 3. 전진 보상 계수 상향 (60 → 80)

```python
reward += 80.0 * dx * height_factor  # 기존 60.0
```

**이유:** 기본 전진 인센티브를 33% 상향. 두 Run 모두 속도와 관계없이 전진 자체의 가치를 높인다.

### 4. speed_gate 상한 확장 (1.0 → 3.0)

```python
speed_gate = np.clip(obs[9] / 3.0, 0.0, 1.0)  # 기존: clip(obs[9], 0, 1)
```

**이유:** 위상 시프트 대칭 보상의 속도 게이트가 1.0 m/s에서 포화되어 있었다. 3.0 m/s까지 선형 보상을 제공하여 "빨리 걸으면서 대칭적인 보행"에 인센티브.

### 5. 안정화 페널티 동적 스케일링 (Strategy 2)

```python
speed = abs(obs[9])
penalty_scale = 1.0 - 0.8 * np.clip(speed / 3.0, 0.0, 1.0)  # 0→1.0, 3.0→0.2
reward -= 0.3 * (z_vel ** 2) * penalty_scale
reward -= 0.3 * (torso_angle ** 2) * penalty_scale
reward -= 0.05 * (torso_ang_vel ** 2) * penalty_scale
```

**이유:** 속도가 높을수록 상체 진동이 필연적으로 증가한다. 속도가 높을 때는 진동을 용인하여 큰 보폭을 허용하고, 느릴 때는 기존대로 안정성을 유지한다.

---

## Run A: Speed Booster (속도 인센티브 중심)

**가설:** 평지 구간에서 속도를 3.5→5.0 m/s로 끌어올리면 bump8 도달 시간이 빨라지고 운동량 증가로 계단을 극복할 수 있다.

### 추가 변경사항

#### 속도 목표 보상 (Strategy 1)

```python
x_vel = obs[9]
target_speed = 2.5
speed_bonus = 0.5 * max(0.0, x_vel - target_speed)  # 2.5 m/s 초과 시 선형 증가
speed_bonus = min(speed_bonus, 2.0)                   # 상한 +2.0/step

# 범프 근처에서는 속도 인센티브 약화 (범프에 전속력 돌진 방지)
big_unpassed_near = any(
    b["big"] and not self.passed_bumps[i]
    and b["front_x"] - 3.0 <= obs[0] <= b["back_x"]
    for i, b in enumerate(self.bumps))
if big_unpassed_near:
    speed_bonus *= 0.2  # 80% 감소

reward += speed_bonus
```

**설계 의도:**
- 2.5 m/s 이하: 기존 보상과 동일 → 안정적 보행 유지
- 2.5~6.5 m/s: 속도 비례 추가 보상 → 더 빨리 가도록 유도
- 6.5 m/s 이상: 상한 포화 → 비현실적 속도 추구 방지
- 범프 근처: 속도 인센티브 80% 감소 → "보고 천천히" 유지

#### stall limit 유지 (500)

기존과 동일하게 500스텝(10초) 정체 시 종료. 등반 시도를 길게 허용.

---

## Run B: Stall Buster (정체 해소 중심)

**가설:** bump8 정체 중에 `healthy = +1.0/step`이 "시도만 하고 실패해도 이득"인 잘못된 신호를 준다. 정체 시 healthy를 0으로 만들고 stall 종료를 250스텝으로 단축하면, 실패한 등반 시도가 빨리 종료되어 성공 샘플의 비중이 높아진다. 계단 등반 보상의 임계값을 완화하면 더 자주 보상을 받아 강화된다.

### 추가 변경사항

#### healthy 게이트 (PLAN_V10 Phase B-1)

```python
if self.stall_steps >= 50:
    reward = 0.0  # 정체 1초 이상이면 healthy 보상 0
else:
    reward = 1.0
```

**이유:** 50스텝(1초) 이상 전진이 없으면 "살아있기만 해도 +1.0"을 제거한다. 정체 중에도 보상을 받으면 계단 앞에서 무한정 시도만 하는 행동이 강화된다.

#### stall 종료 단축 (PLAN_V10 Phase B-3)

```python
stall_limit = 250  # 500 → 250 (5초)
```

**이유:** 학습 중 정체 에피소드를 빨리 끊어 샘플 효율을 높인다. 성공한 등반 시도의 비중이 높아지면 정책이 등반에 더 집중한다. evaluate.py는 terminated를 무시하므로 평가 점수에 영향 없음.

#### 계단 등반 보상 강화 (PLAN_V10 Phase B-4)

```python
# 점프 임계값 완화 + 보상 상향
jump_thresh = min(1.0, 2.0 * rel_height)  # 기존: min(1.5, 3.0 * rel_height)
jump_reward = 15.0                        # 기존: +10

# 높이 임계값 완화 + 보상 상향
height_thresh_add = 0.9                   # 기존: 1.05
height_reward = 20.0                      # 기존: +15
```

**이유:** 계단 2단(bump7→bump8)에서 두 번째 단의 실질 높이는 `base_height` 기준이다. 임계값을 낮추고 보상을 올리면 등반 시도가 더 빨리, 더 강하게 보상받아 학습이 가속된다.

#### 속도 목표 보상 (약하게)

```python
target_speed = 3.0
speed_bonus = 0.3 * max(0.0, x_vel - target_speed)  # Run A 대비 계수 0.5→0.3, 목표 2.5→3.0
speed_bonus = min(speed_bonus, 1.5)                   # 상한 +1.5 (더 낮게)

big_unpassed_near = any(...)
if big_unpassed_near:
    speed_bonus *= 0.1  # 범프 근처 90% 감소 (더 보수적)

reward += speed_bonus
```

**이유:** Run B는 속도보다 등반에 집중하지만, 평지에서 너무 느리면 bump8에 도달하기 전에 시간이 부족하다. 약한 속도 인센티브로 최소 속도만 확보한다. 범프 근처 감쇠를 0.1로 더 보수적으로 설정해 계단 접근 시 속도 인센티브를 거의 끈다.

---

## 코드 구조

모든 변경은 단일 `custom_walker2d.py`에 `run_mode` 파라미터로 통합되어 있다.

```python
# CustomEnvWrapper(run_mode="speed")  → Run A
# CustomEnvWrapper(run_mode="stall")  → Run B
```

`learning.py`의 `--run` 인자로 모드를 선택하면 체크포인트 경로도 자동 분리된다.

---

## 실행 명령어

```powershell
# Run A — Speed Booster
python -u learning.py --bump_challenge --resume checkpoints/bump_challenge/walker_model_2600000_steps.zip --run speed

# Run B — Stall Buster (별도 터미널)
python -u learning.py --bump_challenge --resume checkpoints/bump_challenge/walker_model_2600000_steps.zip --run stall

# 스코어 모니터 (별도 터미널)
python -u score_monitor.py --python python --dirs checkpoints/bump_challenge_v10_speed checkpoints/bump_challenge_v10_stall --watch --interval 180
```

`scores.csv`에 `timestamp, run, model_path, steps, score, status, avg_speed, sim_time`가 누적 기록된다.

---

## 모니터링

| 항목 | 방법 |
|------|------|
| 실제 점수 | `score_monitor.py --watch`로 `scores.csv` 자동 기록 |
| 보상 추이 | TensorBoard `ep_rew_mean` |
| 생존율 | TensorBoard `ep_len_mean` — 1000 유지 여부 |

### 붕괴 신호

- `ep_len_mean`이 1000→700 이하로 급락 → 안정성 희생 과다 → 해당 Run 중단
- `scores.csv`에서 기존 31.77m 밑으로 떨어지면 보행 붕괴 → 해당 Run 롤백

---

## 승자 판정 (2M 스텝 후)

1. 각 Run의 최고 점수를 `scores.csv`에서 확인
2. 더 높은 쪽을 승자로 선택
3. 승자 체크포인트에서 `--resume`하여 상대 Run의 유효 전략 일부 흡수:

| 승자 | 흡수할 전략 |
|------|------------|
| Run A 승리 | B: healthy 게이트, 계단 보너스 강화 |
| Run B 승리 | A: 속도 목표 보상 계수 상향 (0.3→0.5) |

---

## 성공 기준

| 단계 | 목표 |
|------|------|
| 최소 | evaluate.py 점수 **45 m 이상** (기존 31.77 대비 +42%) + 낙상 없음 |
| 권장 | evaluate.py 점수 **60 m 이상** (기존 대비 2배) + 낙상 없음 |
| 이상 | evaluate.py 점수 **80 m 이상** |

---

## 주의사항

1. **`OBS_NORM_*` 상수는 절대 변경하지 않는다.** 체크포인트와 한 쌍이며, 변경 시 기존 zip은 무효화된다.
2. **`evaluate.py`는 결정적이다.** 동일 체크포인트는 항상 동일 점수 → 한 번만 평가해도 확정.
3. **`make_eval_env()`는 `run_mode="base"`를 사용하므로 평가는 보상 변경의 영향을 받지 않는다.** evaluate.py Phase 1은 terminated/reward 반환값을 무시하고 물리 z만 검사하기 때문.
4. **SubprocVecEnv 자식 프로세스 정리:** 중단 시 `Get-Process python \| Stop-Process -Force`.
5. **Phase 0 건너뜀:** 보상 변경 없는 파인튜닝(PLAN_V10 Phase A)은 이 전략과 목표가 다르므로 생략하고 바로 Run A/B로 진행한다.
