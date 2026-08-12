# Walker2d bump_challenge 학습 이력

> `practice2/custom_walker2d.py` + `learning.py` (PPO, SB3) 보상 설계 수정 이력.
> 새로운 문제/수정이 생길 때마다 이 문서에 버전을 추가한다.

## ⇒ 지금 할 일은 `PLAN_V10.md`를 보라 (2026-08-12 14:00 기준)

공식 챌린지 코스(장애물 50개, md5 `338efa83…`)가 확정되어 작업이 v10으로 넘어갔다.
새 세션은 **`PLAN_V10.md`만 읽고 바로 학습을 시작할 수 있다.** 요약:

- 채점 = `evaluate.py`가 **20초(1000스텝) 동안 도달한 최대 x**를 재는 결정적 1회 주행. 완주(101.9m)는 불가 → 거리 경쟁
- 채점기는 `PPO.load(zip)`만 하고 **VecNormalize pkl을 읽지 않는다** → 정규화를 `custom_observation`에 상수로 내장 (v9.1)
  - `OBS_NORM_MEAN/VAR`는 체크포인트와 한 쌍. 학습 중 변경 금지
- 제출 베이스라인: `checkpoints/bump_challenge_v9_8bumps_final/walker_model_2600000_steps.zip` → **31.77 m**
- 병목: x≈26.4의 0.5→1.0m 계단 2단에서 **8.6초 정체**. 그 전까지는 2.7~3.5 m/s로 순조. 정체만 풀면 55m+ 기대
- 아래 v9 섹션은 그 직전 단계(8범프 맵)의 이력이다

## v9 상태 (2026-08-12 12:40, 구 8범프 맵 — 완료)

### 현재 상태 (v9 = 새 맵 8범프)
- `asset/custom_walker2d_bumps.xml`이 **8범프 새 맵**으로 교체됨 (잔범프 6개 + bump2 h=0.5 + bump7 h=0.5 / bump8 h=1.0 계단)
- 관측을 **맵 불변 25차원**으로 재설계하고, 구맵 정책(22차원, 32.2M)을 **가중치 수술**로 이식 → 새 맵 학습 진행 중
- 평가 결과 (원본 XML, `eval_ckpt.py`, deterministic):

| 체크포인트 | mean_len | bump2 | bump7 | bump8 | mean_max_x |
|---|---|---|---|---|---|
| surgery_0 (zero-shot) | 891 | 10/10 | 10/10 | 2~5/10 | 24 |
| 1.6M | 826 | 10/10 | 10/10 | 8/10 | 50 |
| **2.6M (최良)** | **940** | 18/20 | 18/20 | **18/20** | 57 |
| 3.6M | 886 | 18/20 | 18/20 | 18/20 | 54 |
| 4.6M | 845 | 19/20 | 19/20 | 17/20 | 49 |
- **목표(bump8 8/10+, mean_len 800+) 달성**. 남은 실패 유형이 bump8 등반이 아니라 **에피소드 초반(x<4) 낙상**으로 바뀜 (bump2 통과율 = bump8 통과율)
- 실행 환경: **conda `pjt-2`** (`C:\Users\삼성\.conda\envs\pjt-2\python.exe`, gymnasium 1.1.1 / SB3 2.7.0). `python`은 WindowsApps 스텁이라 동작하지 않음
- 학습 재시작:
  ```powershell
  cd practice2
  & C:\Users\삼성\.conda\envs\pjt-2\python.exe -u learning.py --bump_challenge --resume checkpoints/bump_challenge/walker_model_<최신>_steps.zip
  ```

### 다음 할 일
1. 초반 낙상(2/20) 원인 확인 — `render.py`로 x=0~4 구간 관찰. bump1(x=2, h=0.05) 발 걸림 여부
2. 개선 불필요 판단 시 학습 종료, 2.6M 체크포인트를 최종 제출본으로 사용 + 렌더링 녹화(`--record`)
3. (옵션) 공중제비 트랙은 `custom_walker2d.py`를 건드리지 않고 별도 파일로 분리해서 진행

### v9 변경 요약
- **Phase 1**: `BUMP_CONFIGS` 하드코딩 제거 → `_parse_bumps()`가 XML geom(`bump*`)에서 x/half_width/height 파싱. 계단 판정(`base_height`: 직전 범프와 간격<1.0m이고 더 낮으면 그 상단 기준)
- **Phase 2**: 관측 = base 18 + (다음 미도달 범프 K=2개 × [거리/10, 높이, 반폭]) + (torso_z-1.25) = **25차원**. 부족분은 `[2.0, 0, 0]` 패딩 → 맵/범프 개수 불변
- **Phase 3**: 통과 판정을 범프 **뒷면**(x+half_width)으로 변경. 점프/높이 보너스는 h≥0.2 범프만, 임계값을 높이 상대값으로(`z_vel > min(1.5, 3*rel_h)`, `torso_z > height+1.05` → bump8은 2.05). 잔범프 pass는 +50→+10. z_vel 벌점 해제에 "h≥0.4 통과 직후 2m 착지 구간" 추가. stall 종료 300→500스텝
- **Phase 4**: `surgery.py` — 구 첫 레이어(256×22)를 256×25로 열 매핑 이식(base 18열 그대로, 거리열 ×0.5로 /20→/10 스케일 보정, 높이·폭열 0, z offset열 이동), 나머지 레이어 전량 복사. **VecNormalize 통계도 같은 매핑으로 이식**(거리열 mean×2, var×4)한 것이 결정적 — 통계 미이식 시 zero-shot이 즉시 붕괴(len 24)했으나 이식 시 c1 5/5
- **Phase 5**: 신맵 커리큘럼 c1/c2 생성했으나 **zero-shot이 c1 5/5, c2 9/10로 졸업 기준을 이미 충족** → 커리큘럼 생략하고 원본에서 바로 학습
- **README 제약 준수**: `healthy_z_range`를 스켈레톤 원본 `(0.5,10.0)`으로 원복하고 z≥0.9 요구는 `custom_terminated`로 이전 (`custom_reward`가 `original_reward`를 쓰지 않아 동작 동등)

### 지표 확인
- TensorBoard: `logs/` 최신 PPO 런
- 핵심 판독법: `ep_rew_mean ≈ ep_len_mean`이면 정지류 국소최적

### 파일 구성
- `custom_walker2d.py` — 환경 wrapper + 보상 (v9)
- `learning.py` — 학습 (`--resume`, `--xml`)
- `surgery.py` — 관측 차원 변경 시 가중치/정규화 통계 이식 (`--graft_vecnorm`)
- `eval_ckpt.py` — 헤드리스 평가 (h≥0.2 범프만 집계, 범프 개수 자동)
- `render.py` — 렌더링 (`--xml`, VecNormalize 자동 로드)
- `asset/custom_walker2d_bumps{,_c1,_c2}.xml` — 신맵 (c1: bump2/7/8 = 0.3/0.3/0.5, c2: 0.4/0.4/0.75, 원본: 0.5/0.5/1.0)
- `asset/custom_walker2d_bumps_oldmap_c{1,2}.xml` — 구맵 커리큘럼 백업
- `checkpoints/bump_challenge_oldmap/` — 구맵 v8.3 정책 (수술 소스: `walker_model_32200000_steps.zip`)
- 구맵 원본 XML/코드 전체: git 태그 `v8.3-oldmap-final`

## (구) 새 세션 이어가기 가이드 (2026-08-12 01:45 기준, v8.3 구맵)

### 현재 상태
- **c1 졸업 (22.2M)**: c2 제로샷 bump2 10/10, bump3 9/10
- **c2 졸업 (29.2M)**: 27.4M 평가 — c2에서 10/10 전 에피소드 만주행(len 1000, x=72), **원본 XML 제로샷 bump3 8/10**
- **최종 단계(원본 0.6/0.5) 학습 진행 중**: 29.2M에서 `--xml` 없이 resume
- Kilo persistent 백그라운드 프로세스로 실행 중. 확인: `ps aux | grep learning.py`
- 다른 장소에서 이어갈 때: 학습이 중단된 상태면 아래로 재시작
  ```bash
  cd practice2
  python -u learning.py --bump_challenge --resume checkpoints/bump_challenge/walker_model_<최신>_steps.zip
  ```
- 체크포인트: 2M 간격 + 22.2M(c1 최종) + 최신만 보존. **이동 시 최소 필요: 최신 zip + 같은 스텝의 vecnormalize pkl + 코드/asset** (git에 push됨 — 떠나기 전 `git add practice2/checkpoints practice2/logs && git commit && git push`)

### 다음 할 일
1. **31~32M 체크포인트 원본 평가**: bump3 9/10+ 및 mean_len 900+ 확인
   ```bash
   python render.py --model checkpoints/bump_challenge/walker_model_<steps>_steps --bump_challenge
   ```
2. 충족 시 학습 종료 = 과제 완료. 최종 렌더링/녹화 (`--record`)
3. 한발 우세가 지속되면 → Tip #3 (actor 대칭 loss, SB3 PPO 서브클래싱) 구현

### 지표 확인
- TensorBoard: `logs/PPO_10` (이전 실패 런 로그는 삭제됨)
- 핵심 판독법: `ep_rew_mean ≈ ep_len_mean`이면 정지류 국소최적, 통과 보너스 반영 여부는 rew/step > 2.5 상승으로 판단

### 파일 구성
- `custom_walker2d.py` — 환경 wrapper + 보상 (v8.3, 아래 "현재 보상 구조" 참고)
- `learning.py` — 학습 스크립트 (`--resume`, `--xml` 지원)
- `render.py` — 렌더링 (`--xml` 지원, VecNormalize 자동 로드)
- `asset/custom_walker2d_bumps{_c1,_c2,}.xml` — 커리큘럼 XML (c1: 0.4/0.35, c2: 0.5/0.45, 원본: 0.6/0.5)
- `checkpoints/bump_challenge/` — 10M 미만은 2M 단위만 보존, 10M 이상(커리큘럼 구간) 전체 보존
- 삭제됨: 실패 정책 백업 3종(v2/v3/v4), 구 TB 런(PPO_1~9), `.bak`, `__pycache__`

## 요약 타임라인

| 버전 | TB 런 | 증상 (렌더링) | 핵심 원인 | 핵심 수정 |
|---|---|---|---|---|
| v2 | PPO_7 이전 | 제자리 호핑 | `max(0,dx)` 진동 착취 + 대칭항이 동시동작 보상 | — (분석만) |
| v3 | PPO_8 | 한 발짝 걷다 멈춤 (서있기) | 전진 보상이 healthy 대비 너무 약함 (10*dx=0.2/step) | 진동 착취 제거, 대칭항 삭제 |
| v4 | PPO_9 | 무릎으로 기어감 | `healthy_z_range=(0.5,·)`이 저자세 허용, 높이 무관 전진 보상 | 전진 보상 60*dx, 정지 종료 추가 |
| v5 | PPO_10 | 발목 점프, 범프2에서 넘어짐 | z_vel 벌점이 점프 억제, 정지 종료가 등반 시도 차단 | 높이 게이트, z_range 0.9 |
| v6 | PPO_10 (resume) | 범프2 앞 시도만 반복, 정체 | — | 범프 근처 z_vel 벌점 해제, 전면 마일스톤 +15, 정지 종료 6초, ent_coef 0.01 |
| v7 | PPO_10 (resume) | 범프2 일부 통과, 범프3 실패 | 범프2 = 1.4m 길이 0.6m 높이 플랫폼, 탐색만으론 등반 미발견 | 높이 커리큘럼 (0.4 → 0.5 → 0.6) |
| v8 | PPO_10 (resume) | (진행 중) | 한발 점프 gait의 착지 회복력 부족 → 범프2 하강 후 붕괴 | 위상 시프트 대칭 보상 (강의 Tip #2) |

체크포인트 백업:
- `checkpoints/bump_challenge_v2_hopping_backup/` — 제자리 호핑 정책
- `checkpoints/bump_challenge_v3_standing_backup/` — 서있기 정책
- `checkpoints/bump_challenge_v4_crawl_backup/` — 무릎 보행 정책
- `checkpoints/bump_challenge/` — 현재 (v5 + v6 resume, 스텝 번호 연속)

---

## v2 → v3: 제자리 호핑 (2026-08-11 밤)

### 증상
학습된 정책이 전진하지 않고 제자리에서 통통 뜀.

### 진단
1. **`forward_progress = max(0.0, dx)`** — 후진 무벌점. 앞뒤로 진동하면 전진 절반에서만 보상을 수확하는 착취 가능
2. **대칭 페널티가 호핑을 장려**: `-|‖vel_r‖-‖vel_l‖|`, `-|‖ang_r‖-‖ang_l‖|` 형태는 걷기(한 다리 스윙 + 한 다리 지지 = 필연적 불균형)를 벌주고, 양발 동시 호핑(완전 대칭 = 벌점 0)을 보상함. GEMINI.md에서 지적된 Deepseek 대칭항 결함과 동일 구조
3. **alternation 보상이 정적 자세로 충족**: `|thigh_r - thigh_l|`은 다리 벌린 채 서 있어도 만점
4. **범프 앞 z_vel 보상** (`+0.3*max(0,z_vel)` 매 스텝): 범프 앞 2m 구간에서 제자리 점프가 안정적 수익원
5. 스텝 평균 보상이 음수(-0.87/step) → "빨리 죽기"가 최적이 되는 부작용

### 수정
- `dx` 부호 유지 (후진 벌점 복원)
- 대칭 페널티 2종 삭제
- alternation을 `tanh(max(0, -thigh_vel_r * thigh_vel_l))`로: 반대 방향 스윙일 때만 보상
- 범프 근처 매 스텝 shaping 삭제, 1회성 +50만 유지
- 페널티 완화: z_vel² 1.0→0.3, torso_angle² 0.8→0.3, ang_vel² 0.2→0.05
- learning.py: `target_kl=0.03`, `batch_size` 64→256

### 결과
ep_len/ep_rew 초반 급상승했으나 → **서있기 국소최적으로 수렴** (v4에서 해결)

---

## v3 → v4: 서있기 (한 발짝 걷다 멈춤)

### 증상
렌더링 시 한 발짝 내딛다가 멈춤. 로그: `ep_len 972 / ep_rew 967` → 스텝당 보상 ≈ 1.0 = healthy 보너스만 수확하며 1000스텝 버티기.

### 진단
- **`env.dt = 0.02s`** 실측. `10*dx`는 1 m/s 보행 시 스텝당 +0.2에 불과
- 서있기(+1.0, 무위험) vs 걷기(+1.2, 넘어질 위험) → 서있기가 합리적 최적
- deepseek 세션 `v3_out.txt`도 동일 패턴(ep_rew ≈ ep_len)으로 종료됐음 — 같은 함정

### 수정
- 전진 보상 `10*dx` → `60*dx` (1 m/s ≈ +1.2/step > healthy 1.0)
- **정지 종료(stall termination)**: 150스텝(3초)간 `max_x`가 5cm 이상 안 늘면 `terminated=True`. truncated가 아닌 terminated로 끊어 가치 부트스트랩 차단 → "서서 버티기"의 기대가치 자체를 제거
- `ent_coef` 0.0 → 0.005

### 결과
전진 시작 (보상/스텝 > 1.0 확인) → **무릎 보행으로 수렴** (v5에서 해결)

---

## v4 → v5: 무릎 보행 (기어가기)

### 증상
2.4M~4M 체크포인트에서 첫발에 무릎 꿇고 기어감. 속도 낮음.

### 진단
- **`healthy_z_range=(0.5, 10.0)`** — 무릎 보행(torso z ≈ 0.75~0.85)이 healthy로 살아남음. 참고로 기본 Walker2d는 (0.8, 2.0)
- 전진 보상이 높이와 무관 → 낮고 안정적인 기는 자세가 유리

### 수정
- `healthy_z_range` (0.5, 10.0) → **(0.9, 10.0)** (범프 점프용 상한 10.0은 유지)
- **높이 게이트**: `height_factor = clip((torso_z - 0.9) / 0.2, 0, 1)`을 전진 보상에 곱함 → 몸을 낮추면 전진 보상 0
- **낮은 자세 페널티**: `-1.0 * max(0, 1.1 - torso_z)`

### 결과
직립 보행 시작. 범프1(x=6, h=0.3) 안정 통과. ep_len ~330에서 정체 = **범프2(x=10, h=0.6)에서 넘어짐**. 보행이 발목 위주 점프 (v6에서 해결 시도)

---

## v5 → v6: 발목 점프 / 범프2 실패 (진행 중)

### 증상
2.8M+ 체크포인트: 범프2에 걸려 넘어짐. 무릎을 거의 안 쓰고 발목 힘만으로 점프. ep_len ~330, ep_rew ~840에서 정체.

### 진단
1. **z_vel² 벌점이 점프를 억제**: 0.6m 범프를 넘으려면 큰 상승 속도가 필요한데 z_vel=2면 -1.2/step 벌점 → 벌점이 적은 최소 점프(발목 통통)로 수렴
2. **정지 종료 3초가 등반 시도를 차단**: 범프 앞에서 x 진전 없이 낑낑대면 3초 만에 에피소드 종료 → 오르는 동작을 실험할 시간이 없음
3. **보상 기울기 부재**: 범프2는 +50(통과) 아니면 0. "거의 넘음"과 "접근 못함"이 같은 보상

### 수정
- **z_vel² 벌점을 미통과 범프 근처(x ∈ [범프-2.5, 범프+1.0])에서 해제** — 벌점 제거이므로 착취 불가
- **범프 전면 마일스톤**: `x > 범프x - 0.4` 도달 시 1회성 +15 (x 단조 마일스톤이라 진동 착취 불가)
- 정지 종료 150 → **300스텝(6초)**
- learning.py에 **`--resume`** 추가: 체크포인트 + VecNormalize 통계 로드, `reset_num_timesteps=False`
- resume 시 `ent_coef` 0.01로 상향 (등반 동작 탐색)
- **6.8M 체크포인트에서 이어서 학습** (평지 보행 유지, 처음부터 재학습 회피)

### 결과
- resume(6.8M) 후 7.4M 시점: ep_len 330→**540**, ep_rew 840→**1030** — 일시 개선
- 그러나 9.6M까지 ep_len 420~550 진동, 돌파 실패 → v7 커리큘럼으로 전환

### 외부 리뷰 검토 (Gemini, 2026-08-12 00:13)
- "ep_len 478" 지적은 폐기된 PPO_9(v4 무릎 보행 런) 데이터를 본 것 — 현재 런과 무관
- `target_kl` 삭제 제안은 기각: SB3는 1.5×target_kl에서 중단하며, v2에서 관측된 업데이트 불안정을 막는 가드로 유지
- stall 300 제안은 v6에서 이미 적용됨
- 커리큘럼 제안은 타당하나 전제("걷기 불안정")가 틀렸고, flat(19)/practice(21)/challenge(22) 관측 차원이 달라 그대로는 전이 불가 — 적용 시 관측 통일 선행 필요

### 판단 기준 (다음 개입 시점)
- resume 후 +2~3M 스텝에도 ep_len 330 정체 → 범프2 통과 자체를 커리큘럼으로 (practice XML의 h=0.45 범프로 선학습) 또는 등반 shaping 재설계
- 한발 점프 고착 지속 → alternation 가중치(0.3) 상향 검토

---

## v6 → v7: 높이 커리큘럼 (진행 중, 2026-08-12 00:25)

### 증상
범프2 앞에서 기울이기/발 움직임 시도는 관측되나 3M 스텝(6.8M→9.6M) 동안 ep_len 420~550 진동 — 돌파 실패.

### 진단
- XML 확인 결과 box `size`는 half-extent → **범프2는 x∈[9.3,10.7], 길이 1.4m·높이 0.6m 플랫폼**. 점프로 넘는 게 아니라 "올라가서 → 위를 걷고 → 내려오는" 스텝업 과제
- 0.6m 스텝업은 고관절·무릎 대굴곡이 필수인데 현재 발목 위주 gait에서 무작위 탐색으로 발견될 확률 낮음

### 수정
- **높이 커리큘럼 XML** 생성 (범프 위치/개수 동일 → 관측 22차원 유지, 체크포인트 전이 가능):
  - `asset/custom_walker2d_bumps_c1.xml`: bump2 h=0.4, bump3 h=0.35
  - `asset/custom_walker2d_bumps_c2.xml`: bump2 h=0.5, bump3 h=0.45
  - 원본 `custom_walker2d_bumps.xml`: bump2 h=0.6, bump3 h=0.5 (최종)
- `CustomEnvWrapper`/`learning.py`/`render.py`에 `--xml` 오버라이드 추가
- 10M 체크포인트에서 c1으로 resume

### 단계 승급 기준
- 해당 단계에서 ep_len이 800+ 그리고 범프3까지 통과(ep_rew에 +130 보너스 반영) 안정화 → 다음 단계 XML로 resume
- c1 → c2 → 원본 순서. 각 단계 예상 2~5M 스텝

### 렌더링 (커리큘럼 단계 확인 시)
```bash
python render.py --model checkpoints/bump_challenge/walker_model_<steps>_steps --bump_challenge --xml asset/custom_walker2d_bumps_c1.xml
```

### 결과
- c1 resume(10M) 후 13.6M: 범프2(h=0.4) 어느 정도 통과, 범프3 거의 실패. len ~410 = 범프2 통과 직후 붕괴 패턴 → v8

---

## v7 → v8: 위상 시프트 대칭 보상 (진행 중, 2026-08-12 00:40)

### 증상
c1에서 범프2는 일부 통과하나 하강 착지 후 자세 붕괴로 범프3(x=15) 도달 실패. gait가 여전히 한발 점프 위주라 외란(착지) 회복력이 약함.

### 근거 (강의 자료 tip1.png / tip2.png)
- **Solution #2 (채택)**: 오른다리 각도·각속도를 버퍼에 저장, 반주기 후 왼다리가 재현하도록 L2 기반 보상 — 시간차 대칭이라 v2에서 삭제한 "순간 대칭"(동시 호핑 보상 결함)과 달리 교대 gait를 올바르게 유도
- **Solution #3 (보류)**: actor 대칭 loss는 SB3 PPO train() 서브클래싱 필요 + 비대칭 관측(범프 거리) mirror 설계 필요 → Tip1로 부족할 때 다음 카드

### 수정 (custom_walker2d.py)
- `GAIT_HALF_PERIOD = 15` (0.3s), 오른다리 상태 `[각도3, 0.1*각속도3]` deque 저장
- `+0.5 * exp(-2 * L2(왼다리_now, 오른다리_반주기전))` 보상 추가
- 미통과 범프 근처에서는 끔 (등반은 비대칭 동작이 필요)
- 호핑이 우연히 주기 일치로 만점 받는 걸 기존 "반대 방향 스윙" 보상(0.3)과 병행해 방어
- 13.8M 체크포인트에서 c1 XML로 resume

### 결과
- **성공 (22.2M, 2026-08-12 01:15)**: c1 헤드리스 평가(19.4M) bump3 8/10 → 22.2M을 c2에서 제로샷 bump2 10/10 / bump3 9/10 / mean_len 852. c1 졸업, 22.2M에서 c2로 승급 resume

### v8.1 핫픽스 (2026-08-12 00:45)
- **증상**: 범프3 통과 후 서서히 멈춤
- **원인 1**: 위상 대칭 보상의 정적 자세 exploit — 가만히 있으면 왼다리(현재)=오른다리(반주기 전)가 자명하게 성립해 +0.5 공짜 수확 (서있기가 1.5/step이 됨)
- **원인 2**: 보상 사다리가 범프3(x=15)에서 끝나 그 너머는 전진 유인 부재 + 미학습 영역
- **수정**: 대칭 보상에 전진 속도 게이트 `clip(x_vel, 0, 1)` 곱함 (정지 시 0) / 마지막 범프 +3m(+25), +5m(+50) 1회성 마일스톤 추가
- 검증: 정지 정책 스텝 보상 0.94 (대칭 보너스 미지급 확인). 14.4M에서 c1 resume

### v8.2 핫픽스 (2026-08-12 00:47)
- **증상**: 점프에서 왼발목 거의 미사용 (한발 점프 지속)
- **원인**: 대칭 계열 보상(교대 스윙, 위상 시프트)을 미통과 범프 근처에서 끄도록 설계 → 정작 점프 동작에는 대칭 유인이 전혀 없음
- **수정**: 범프 근처 한정 양발목 push-off 보상 `+0.2 * min(|action_foot_r|, |action_foot_l|)` — 두 발목 토크 중 작은 쪽 기준이라 한쪽만 쓰면 0, 양쪽을 같이 써야 보상
- 15.4M에서 c1 resume
- 참고: 평지 gait의 대칭화(v8 위상 보상)는 이제 막 시작이라 효과 판정에 3~5M 스텝 필요. 그래도 한발 우세가 지속되면 Tip #3(actor 대칭 loss, 커스텀 PPO)로 승격

### v8.3 명시적 점프 유도 (2026-08-12 00:52)
- **요청**: 기울이기/휘적거리기로는 범프가 높아 보임 — 점프를 명시적으로 가르칠 수 없나
- **설계**: v2에서 뺐던 매 스텝 z_vel 보상(제자리 점프 농사 exploit)을 **1회성 플래그**로 재도입 → 착취 불가
  - 접근 구간(범프 -2.0m ~ +0.7m)에서 처음 `z_vel > 1.5` 달성 시 +10 (도약 임펄스)
  - 같은 구간에서 처음 `torso_z > 1.55` 달성 시 +15 (체공/등반 높이 — 평지 보행 1.25~1.4로는 불가)
- 보상 사다리: 전면 도달(+15) → 도약(+10) → 몸 띄우기(+15) → 통과(+50)
- 16.8M에서 c1 resume

---

## 현재 보상 구조 (v9)

```
reward = 1.0                                  # healthy
       + 60 * dx * height_factor              # 전진 (높이 게이트, 후진 벌점)
       - 0.3 * z_vel²                         # 미통과 h>=0.2 범프 근처 / h>=0.4 통과 직후 2m는 해제
       - 0.3 * torso_angle²
       - 0.05 * torso_ang_vel²
       - 1.0 * max(0, 1.1 - torso_z)          # 낮은 자세 벌점
       + 0.2 * min(|a_ankle_R|, |a_ankle_L|)  # 큰 범프 근처 양발 push-off
       + 0.3 * tanh(max(0, -thigh_vel_r * thigh_vel_l))  # 교대 스윙
       + 0.5 * exp(-2*L2(왼다리_now, 오른다리_반주기전))   # 위상 대칭 (큰 범프 근처 해제)
       + 10 (1회) # 도약: 접근 구간에서 z_vel > min(1.5, 3*rel_height)   (h>=0.2만)
       + 15 (1회) # 몸 띄우기: torso_z > height + 1.05                   (h>=0.2만)
       + 15 (1회) # 범프 전면(x > front_x - 0.4) 도달                    (h>=0.2만)
       + 50 (1회) # 범프 통과 (x > back_x), 잔범프는 +10
       + 25/50 (1회) # 마지막 범프 +3m / +5m 지점
종료: 기본 termination + z<0.9(custom_terminated) + 500스텝 무진전
관측: base 18 + 다음 2범프 × [거리/10, 높이, 반폭] + (torso_z-1.25) = 25차원
```

## (구) 보상 구조 (v8, 구맵 3범프)

```
reward = 1.0                                  # healthy
       + 60 * dx * height_factor              # 전진 (높이 게이트, 후진 벌점)
       - 0.3 * z_vel²                         # 미통과 범프 근처에서는 해제
       - 0.3 * torso_angle²
       - 0.05 * torso_ang_vel²
       - 1.0 * max(0, 1.1 - torso_z)          # 낮은 자세 벌점
       + 0.3 * tanh(max(0, -thigh_vel_r * thigh_vel_l))  # 교대 스윙
       + 0.5 * exp(-2*L2(왼다리_now, 오른다리_반주기전))   # 위상 시프트 대칭 (범프 근처 해제)
       + 15 (1회) # 범프 전면(x > bump_x - 0.4) 도달
       + 50 (1회) # 범프 통과 (x > bump_x)
종료: 기본 termination + z<0.9 + 300스텝 무진전
```

## 학습 설정 (learning.py)

- PPO, MlpPolicy [256,256]/[256,256], `log_std_init=-1.0`
- `lr=3e-4, n_steps=2048, batch_size=256, gamma=0.995, target_kl=0.03`
- `ent_coef`: 신규 0.005 / resume 0.01
- N_ENVS=10, SubprocVecEnv + VecMonitor (**v9.1부터 VecNormalize 미사용** — 정규화는 환경에 내장)
- 체크포인트 200k 스텝마다 (`save_freq=20000 × 10 envs`), **zip만 저장 (pkl 없음)**

## 렌더링 / 평가

```powershell
cd practice2
$py = "C:\Users\삼성\.conda\envs\pjt-2\python.exe"
& $py -u evaluate.py --model checkpoints/bump_challenge/walker_model_<steps>_steps.zip   # 공식 점수
& $py -u diag_run.py --model checkpoints/bump_challenge/walker_model_<steps>_steps.zip   # 정체 지점 진단
& $py render.py --model checkpoints/bump_challenge/walker_model_<steps>_steps --bump_challenge
```
- 학습과 같은 플래그 필수 (`--bump_practice`로 열면 관측 차원이 달라짐)
- VecNormalize pkl은 더 이상 사용하지 않는다 (있으면 이중 정규화가 되므로 로드 금지)

## 교훈

1. 보상 항은 "착취 시나리오"를 먼저 생각할 것 (진동, 정적 자세, 저자세, 최소 동작)
2. `ep_rew_mean ≈ ep_len_mean`이면 healthy 보너스만 먹는 중 = 정지류 국소최적
3. 보상 스케일은 dt를 실측해서 계산할 것 (frame_skip 때문에 직관과 다름)
4. 벌점(z_vel 등)이 과제에 필요한 동작(점프)과 충돌하지 않는지 확인
5. 1회성 x-단조 마일스톤은 안전한 shaping, 매 스텝 상태 기반 shaping은 착취 위험
6. 관측 차원을 바꿔도 **첫 레이어 열 매핑 수술 + 정규화 통계 이식**으로 기존 정책을 거의 그대로 이식할 수 있다.
   통계 이식을 빠뜨리면 정책이 즉시 붕괴하므로(len 24) 가중치만 옮기는 것은 무의미하다 (v9 Phase 4)
7. 채점 파이프라인을 먼저 읽을 것. `evaluate.py`는 pkl을 로드하지 않아 VecNormalize에 의존한 정책은
   그대로 제출하면 0점이 된다. 학습 편의(러닝 정규화)와 제출 요건(자립형 zip)은 다를 수 있다
8. 평가 지표를 보상과 일치시킬 것. 점수가 "고정 시간 내 거리"라면 매 스텝 healthy 보너스는
   정체(버티기)를 보상하는 셈이 된다 (v10 Phase B의 출발점)
