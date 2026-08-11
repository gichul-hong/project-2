# Practice 2 Claude Master Prompt (고득점용)

Below is the master prompt optimized for **Claude (Claude 3.5 Sonnet / Claude Code / Claude Chat)** to achieve top scores on **Practice 2 (Walker2D Controller Training)**.

---

```markdown
너는 MuJoCo 및 Gymnasium 기반 로봇 제어, PPO 강화학습 최고 전문가이다.
삼성 DS RL 실습 과제 `practice2` (Walker2D 제어)의 3가지 Task를 모두 완벽히 해결하는 Python 코드를 구현해라.

---

### [환경 및 과제 명세]
1. 수정 가능 파일: `custom_walker2d.py`, `learning.py`
2. 과제 목표:
   - Task 1 (기본 걷기): 20초(1,000 steps) 이내 그리드 끝 도달 (전진 속도 극대화).
   - Task 2 (자연스러운 걷기): 상체 직립 및 좌우 대칭성(Symmetry) 보상을 추가하여 자연스러운 폼으로 20초 이내 도달.
   - Task 3 (2개 Bump 통과): `custom_walker2d_bumps_practice.xml` 환경 (Bump1: x=6.0, h=0.2 / Bump2: x=10.0, h=0.45)에서 장애물을 넘어서 20초 이내 도달.

---

### [핵심 보상 및 관측 설계 요구사항]

1. **관측(Observation) 설계**:
   - 기본 17차원 obs 외에, Task 3(Bump) 대응을 위해 다음 신규 정보 추가:
     * `bump1_rel_x = (6.0 - torso_x) / 10.0`
     * `bump2_rel_x = (10.0 - torso_x) / 10.0`
     * `torso_height_dev = torso_z - 1.25`
   - 선형 조합에 불과한 단순 피처는 추가하지 마라.
   - 관측 공간 크기 변경 시 `self.observation_space` 차원 및 Box 범위가 자동 업데이트되도록 작성해라.

2. **보상(Reward) 설계 ($O(1)$ 스케일 유지)**:
   - **기본 보상**: `healthy_reward(1.0) + 1.5 * forward_vel - 0.001 * ctrl_cost`
   - **상체 안정성 (Task 2)**:
     * `- 0.5 * (torso_angle ** 2)`
     * `- 0.1 * (torso_ang_vel ** 2)`
   - **보행 대칭성 (Task 2 - Symmetry)**:
     * 오른다리 관절(thigh, leg, foot)과 왼다리 관절 간의 대칭 동작을 장려.
     * 순간 대칭 및 반주기 대칭성을 유도하기 위한 지표 사용: `- 0.2 * (|thigh_r + thigh_l| + |leg_r - leg_l_shifted|)` 형태 등 스무스한 L2/L1 페널티.
   - **장애물 통과 보상 (Task 3)**:
     * 장애물 근접 시(bump1, bump2 이전 1.0m 이내) 약간의 상향 속도(`rootz vel`) 및 다리 들기 장려.
     * 장애물 x좌표를 넘어섰을 때 일회성/구간 통과 보너스(`+2.0`) 부여.
   - **주의**: 극단적인 벌점(-100 등)은 수렴을 망치므로 절대 금지.

3. **하이퍼파라미터 설정 (`learning.py`)**:
   - `net_arch=[dict(pi=[256, 256], vf=[256, 256])]` (충분한 표현력 확보)
   - `learning_rate`: `3e-4`
   - `n_envs`: 8 (SubprocVecEnv)
   - `gamma`: `0.99`
   - `ent_coef`: `0.001` (Rich Observation이므로 탐색 노이즈를 낮춰 안정적 수렴 유도)
   - `n_steps`: 2048, `batch_size`: 64

---

### [제출물 요구사항]
1. `custom_walker2d.py` 코드 전체 (CustomEnvWrapper 내 custom_observation, custom_reward, custom_terminated 등 complete code)
2. `learning.py` 코드 전체
3. 각 Task 별로 코드를 전환하거나 공통으로 작동시키는 방식에 대한 짧은 안내.

지금 작성해라.
```
