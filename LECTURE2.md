# RL 프로젝트 실습2: 컨트롤러 학습 (2026.08.11)

> [2026 삼성 DS과정] PPT 내용 요약

---

## 실습 환경 설정

```bash
cd {프로젝트 폴더}
python -m venv venv
.\venv\Scripts\activate
pip install "stable_baselines3[extra]>=2.0.0a9"
pip install "gymnasium[mujoco]"
```

**VSCode cmd 터미널 3개:**
1. `.\venv\Scripts\activate` + `python learning.py`
2. `.\venv\Scripts\activate` + `python render.py --model checkpoints\pretrained_model\walker_model_1960000_steps.zip`
3. `.\venv\Scripts\activate` + `tensorboard --logdir logs/`

---

## 실습 내용

3가지 레벨:
1. **Walker2D 걷게 하기** — 20초 이내에 그리드 끝에 도달하기
2. **Walker2D 자연스럽게 걷게 하기** — 20초 이내에 자연스러운 보행 동작으로 그리드 끝에 도달하기
3. **2개의 bump 넘기** — 20초 이내에 2개의 bump를 통과하기

---

## Walker2D 환경 상세

### Observation Space
- `Box(-inf, inf, (17,), float64)`
- qpos (8): Position values of the robot's body parts
- qvel (9): Velocities of individual body parts

| Num | Observation | Joint | Type |
|-----|------------|-------|------|
| 0 | x-coordinate of the torso | rootx (slide) | position (m) |
| 1 | z-coordinate of the torso (height) | rootz (slide) | position (m) |
| 2 | angle of the torso | rooty (hinge) | angle (rad) |
| 3 | angle of the thigh joint | thigh_joint (hinge) | angle (rad) |
| 4 | angle of the leg joint | leg_joint (hinge) | angle (rad) |
| 5 | angle of the foot joint | foot_joint (hinge) | angle (rad) |
| 6 | angle of the left thigh joint | thigh_left_joint (hinge) | angle (rad) |
| 7 | angle of the left leg joint | leg_left_joint (hinge) | angle (rad) |
| 8 | angle of the left foot joint | foot_left_joint (hinge) | angle (rad) |
| 9 | velocity of x-coordinate of torso | rootx (slide) | velocity (m/s) |
| 10 | velocity of z-coordinate (height) of torso | rootz (slide) | velocity (m/s) |
| 11 | angular velocity of torso angle | rooty (hinge) | angular velocity (rad/s) |
| 12 | angular velocity of thigh hinge | thigh_joint (hinge) | angular velocity (rad/s) |
| 13 | angular velocity of leg hinge | leg_joint (hinge) | angular velocity (rad/s) |
| 14 | angular velocity of foot hinge | foot_joint (hinge) | angular velocity (rad/s) |
| 15 | angular velocity of left thigh hinge | thigh_left_joint (hinge) | angular velocity (rad/s) |
| 16 | angular velocity of left leg hinge | leg_left_joint (hinge) | angular velocity (rad/s) |
| 17 | angular velocity of left foot hinge | foot_left_joint (hinge) | angular velocity (rad/s) |

### Action Space
- `Box(-1.0, 1.0, (6,), float32)` — 각 hinge joint의 토크

| Num | Action | Joint |
|-----|--------|-------|
| 0 | Torque on thigh rotor | thigh_joint |
| 1 | Torque on leg rotor | leg_joint |
| 2 | Torque on foot rotor | foot_joint |
| 3 | Torque on left thigh rotor | thigh_left_joint |
| 4 | Torque on left leg rotor | leg_left_joint |
| 5 | Torque on left foot rotor | foot_left_joint |

### Rewards
```
reward = healthy_reward + forward_reward - ctrl_cost
```
- `healthy_reward`: 살아있으면 매 timestep마다 고정 1
- `forward_reward`: `weight * (torso x좌표 변화량) / (시간 간격)` = 앞으로 가는 속도
- `ctrl_cost`: action 크기에 따른 페널티 (불필요한 힘 억제)

### Starting State
- 초기 위치: `qpos = [0, 1.25, 0, 0, 0, 0, 0, 0, 0] + reset_noise`
- 초기 속도: `qvel = reset_noise`
- torso 높이 약 1.25, 소량의 random noise 적용

### Episode End
- **Termination**: Walker2d가 "건강하지 않은 상태" → 즉시 종료
  - state 값이 무한대 또는 NaN
  - torso 높이(z)가 `healthy_z_range` 밖 (default `[0.8, 2.0]`)
  - torso 각도(rooty) 절댓값이 `healthy_angle_range` 밖 (default `[-1, 1]`)
- **Truncation**: 최대 1,000 timesteps 도달 시 종료

---

## 실습 1: Walker2D 걷게 하기

**Goal**: 캐릭터가 앞으로 걸어가서 20초 안에 그리드 끝에 도달하기

```bash
# Training
python learning.py

# Rendering
python render.py --model checkpoints/walker_model/{model_name}.zip
```

---

## 실습 2: Walker2D 자연스럽게 걷게 하기

**Goal**: 자연스러운 보행 모션이 나오도록 하기
- 발목, 무릎, 엉덩이 관절을 고르게 사용
- 왼/오른 다리를 대칭적으로 사용

### 자연스럽지 않은 동작 예시
- 발목, 무릎, 엉덩이를 고르게 사용하지만 약간의 비대칭성

### Solution #1
보통 사람이 걸을 때, **상체의 움직임이 크지 않고 올바르게 서서** 걷는다는 직관을 사용

### Solution #2
일반적으로 사람이 걷거나 뛸 때 **두 다리가 같은 동작을 일정 간격으로 반복** → 주기적인 동작을 유도하는 reward 추가
- 매 step마다 오른쪽 다리 각 부분의 각도 및 각속도를 tuple로 저장
- 임의로 정한 주기 후 왼쪽 다리가 오른쪽 다리와 비슷한 각도/각속도를 가지도록 **L2 loss** 적용

### Solution #3
PPO의 actor가 **대칭 action**을 만들어 내도록 하는 loss를 추가

```bash
# Training
python learning.py

# Rendering
python render.py --model checkpoints/walker_model/{model_name}.zip
```

---

## 실습 3: 2개의 Bump 넘기

**Bump 정보** (`custom_walker2d_bumps_practice.xml`):

| Bump | Position | Size |
|------|----------|------|
| Bump #1 | (6.0, 0.0, 0.0) | (0.3, 2.0, 0.2) |
| Bump #2 | (10.0, 0.0, 0.0) | (0.6, 2.0, 0.45) |

```bash
# Training
python learning.py --bump_practice

# Rendering
python render.py --model checkpoints/bump_practice/{model_name}.zip --bump_practice

# Recording
python render.py --model checkpoints/bump_practice/{model_name}.zip --bump_practice --record
# R키 눌러 녹화 시작, 다시 R키 눌러 중지 및 mp4 저장. Q 키로 종료.
```