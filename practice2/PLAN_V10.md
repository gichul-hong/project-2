# v10 작업 계획: 공식 챌린지 코스(장애물 50개) 거리 최대화

> 새 세션 인수인계 문서. 이 파일만 읽고 바로 학습을 시작할 수 있게 정리했다.
> 이전 이력은 `TRAINING_HISTORY.md`(v1~v9), `PLAN_V9.md`(관측 재설계·가중치 수술) 참고.

## 0. 실행 환경 (중요)

- 학습 미시작 상태로 인수됨. **Phase A 명령 한 줄로 바로 시작 가능** (§6)
- Python: **conda `pjt-2`** — `C:\Users\삼성\.conda\envs\pjt-2\python.exe` (gymnasium 1.1.1 / SB3 2.7.0)
  - `python`은 WindowsApps 스텁이라 **동작하지 않는다.** 항상 위 절대 경로를 쓸 것
  - 백그라운드로 돌린 뒤 중단할 때는 `SubprocVecEnv` 자식 프로세스(10개)가 남을 수 있다.
    `Get-Process python`으로 확인하고 남으면 `Stop-Process -Force`로 정리할 것
- 작업 디렉터리: `C:\hong\project-2\practice2`
- 공식 지형: `asset/custom_walker2d_bumps.xml`, **md5 `338efa83e0e7f70abf80b2ba514322cb`** (일치 확인됨)
  - 리포지토리 루트 `.gitattributes`에 `*.xml -text`가 있어야 체크아웃 시 줄바꿈 변환으로 md5가 깨지지 않는다
- 확인 명령:
  ```powershell
  cd C:\hong\project-2\practice2
  Get-FileHash -Algorithm MD5 asset/custom_walker2d_bumps.xml
  & C:\Users\삼성\.conda\envs\pjt-2\python.exe -u dump_map.py     # 장애물 50개 목록
  ```

## 1. 채점 방식 (`evaluate.py` = 서버와 동일)

- **점수 = 1000스텝(=20초) 동안 도달한 최대 x**. `GOAL_X=101.9` 완주는 평균 5.1 m/s가 필요해 사실상 불가 → **거리 경쟁**
- Phase 1: 우리 `CustomEnvWrapper`로 행동열만 수집 (관측/보상/종료는 우리 코드 그대로 동작)
- Phase 2: 학생 코드 없는 공식 env에 그 행동열을 재생해서 채점. `reset_noise_scale=0`, `seed=0`, greedy → **완전 결정적**
- 지형/`frame_skip`/`reset_noise`는 채점기가 강제. `exclude_current_positions_from_observation`은 우리 값이 통과됨
- 낙상 판정: torso z가 (0.5, 10.0) 밖. 중도 예외 발생 시 "실패"로 리더보드 제외
- 실행:
  ```powershell
  & C:\Users\삼성\.conda\envs\pjt-2\python.exe -u evaluate.py --model <ckpt.zip>
  & C:\Users\삼성\.conda\envs\pjt-2\python.exe -u evaluate.py --model <ckpt.zip> --video run.mp4
  ```

## 2. 제출물 계약 (pkl 없음)

- 채점기는 `PPO.load(zip)`만 한다. **VecNormalize pkl은 읽지 않는다.**
- 그래서 관측 정규화를 `custom_walker2d.py`의 `custom_observation` 안에서 직접 수행한다
  (`OBS_NORM_MEAN/VAR` 상수, `(x-mean)/std` 후 ±10 클립 — VecNormalize와 수치 동일)
- `learning.py`는 VecNormalize를 쓰지 않고 체크포인트도 **zip만** 저장한다
- `make_eval_env()`가 `custom_walker2d.py`에 정의돼 있어 채점기가 생성자 인자 탐색을 생략한다
- **주의**: `OBS_NORM_*` 상수는 체크포인트와 한 쌍이다. `bake_norm.py`로 다시 구우면 기존 zip은 전부 무효 →
  상수를 바꾸려면 재학습·재검증을 함께 해야 한다. 학습 중에는 절대 건드리지 말 것

## 3. README 제약 (반드시 유지)

- "Do not modify the XML file path or environment parameters in `__init__`"
- `healthy_z_range=(0.5, 10.0)`, `frame_skip=10`, `exclude_current_positions_from_observation=False` = 스켈레톤 원본값 유지
- 무릎 보행 차단(z≥0.9)은 `custom_terminated`에서 `MIN_TORSO_Z`로 처리 (환경 파라미터 아님)
- `xml_file` 인자는 기본값 `None`이며 그때 기본 경로를 사용 (커리큘럼 학습용, 채점에 영향 없음)
- 변경 허용 범위: Observation / Reward / Termination 및 그 보조 로직

## 4. 현재 성능 (베이스라인)

- 최종본: `checkpoints/bump_challenge_v9_8bumps_final/walker_model_2600000_steps.zip` (구 8범프 맵에서 2.6M 스텝)
  - 같은 폴더의 `obs_norm_source_2600000.pkl`은 `OBS_NORM_*` 상수를 굽는 데 쓴 원본 통계(참고용).
    이름을 일부러 `walker_model_vecnormalize_*` 규칙에서 벗어나게 두었다 — `render.py`가 자동으로 집어
    이중 정규화하는 것을 막기 위함. **다시 pkl을 로드하는 코드를 만들지 말 것**
- `checkpoints/bump_challenge/`는 비워둔 상태 (v10 학습 산출물이 여기 쌓인다). 구 정규화 방식으로 학습된
  중간 체크포인트는 상수와 짝이 맞지 않아 전부 정리했다 (git 이력에는 남아 있음)
- 공식 코스 점수: **31.77 m** (넘어짐 없음, 1000/1000 스텝, 평균 1.59 m/s)
- 진행 트레이스 (`diag_run.py --every 50`):

| 구간 | 관찰 |
|---|---|
| x=0~25 | **2.7~3.5 m/s로 순조롭게 주파** (잔범프 bump1·3·4·5·6, 0.5m 벽 bump2 모두 통과) |
| x≈26.4 (bump8, h=1.0) | **step 500~930 (약 8.6초) 정체** — 0.5→1.0m 계단 2단에서 막힘 |
| x=28~31.8 | 돌파 후 3.3 m/s로 재주행, 시간 종료 |

→ **정체 8.6초만 없애면 그 속도로 25m 이상 추가 주행 가능(≈55~60m).** 최우선 개선 대상은 명확하다.

## 5. 코스 구조 (요약, 전체는 `dump_map.py`)

- x=0~9: 평지 (도움닫기)
- 반복 패턴 A: 0.5m 단독 벽 (bump2@11.6, bump15@42, bump19@48, bump36@80.7, bump40@86.7)
- 반복 패턴 B: **0.5→1.0m 계단 2단** (bump7·8@25.6/26.6, bump19·20@48/49, bump40·41@86.7/87.7) ← 현재 병목
- 반복 패턴 C: 0.4→0.9m (bump17·18@45, bump38·39@83.7)
- 반복 패턴 D: 0.25→0.75→1.2m 3단 (bump24~26@57~59, bump45~47@95.7~97.7)
- 반복 패턴 E: **0.5→0.9→1.3m 3단, 간격 0.5m** (bump27~29@61~62, bump48~50@99.7~100.9) ← 최고 난도
- 잔범프(h≤0.15)는 x=9.6, 14.6, 17.6, 20.6, 22.6, 29, 31, 39, 67.7, 69.7, 77.7에 산재
- 우리 코드의 계단 판정(`base_height`: 직전 장애물과 간격<1.0m이고 더 낮으면 그 상단 기준)이 B/C/D/E 모두에 정상 적용됨

## 6. v10 작업 순서

### Phase A — 파인튜닝 먼저 (보상 변경 없이)
공식 코스에서 그냥 이어 학습해서 병목이 자연히 풀리는지 본다. 코스 자체가 커리큘럼(쉬운 구간 → 어려운 구간)이다.
```powershell
cd C:\hong\project-2\practice2
& C:\Users\삼성\.conda\envs\pjt-2\python.exe -u learning.py --bump_challenge --resume checkpoints/bump_challenge_v9_8bumps_final/walker_model_2600000_steps.zip
```
- 체크포인트: `checkpoints/bump_challenge/walker_model_<steps>_steps.zip` (200k 스텝마다, zip만)
- 1M 스텝마다 `evaluate.py`로 점수 확인. 판단 기준: **1~2M 안에 40m를 넘는지**
- resume 시 `ent_coef=0.01` 자동 적용 (learning.py 기존 로직)

### Phase B — 정체 해소용 보상 조정 (Phase A가 정체하면)
점수가 "20초 거리"이므로 현재 보상은 목표와 어긋난 부분이 있다. 아래 순서로 하나씩, 각 변경 후 1M 스텝 평가.
1. **healthy 상수 보상 축소**: `reward = 1.0`은 "버티기"에 이득을 준다. 정체 8.6초 동안에도 +1/step이 들어옴.
   → 전진 게이트를 걸거나(예: 최근 1초 진행 < 0.05m면 healthy 0), 값을 0.5로 낮춘다
2. **전진 가중치 상향**: `60.0 * dx` → 80~100. dx는 부호가 있어 후진 착취는 불가 (검증됨)
3. **stall 종료 단축**: 500 → 200~300스텝. 학습 시 정체 에피소드를 빨리 끊어 샘플 효율을 올린다
   (채점에는 종료가 영향 없음 — 채점기는 우리 terminated를 무시하고 1000스텝을 재생한다)
4. **계단 등반 보너스 강화**: 패턴 B/E에서 `rel_height` 기준 도약(+10)·높이(+15)를 상향하거나,
   "계단 상단 착지"(직전 상단보다 높은 위치에서 양발 접지) 1회성 보너스 추가
- 원칙: **1회성 플래그 또는 벌점 해제만.** 매 스텝 양수 shaping 추가 금지 (TRAINING_HISTORY 교훈 1·5)
- 변경 금지(검증된 항): 위상 대칭 0.5, 교대 스윙 0.3, 양발목 push-off 0.2, 높이 게이트, 낮은 자세 벌점

### Phase C — 제출
1. `evaluate.py`로 최고 점수 체크포인트 선정 (결정적이므로 1회 실행으로 확정)
2. `--video run.mp4`로 영상 확인 (ffmpeg 필요)
3. 최종본을 `checkpoints/bump_challenge_v10_final/`에 복사 + `TRAINING_HISTORY.md`에 v10 섹션 기록

## 7. 도구 목록

| 파일 | 용도 |
|---|---|
| `learning.py` | 학습 (`--bump_challenge --resume <zip>`, `--xml`로 지형 교체) |
| `evaluate.py` | **공식 채점기** (서버와 동일 점수) |
| `diag_run.py` | 채점기와 동일 조건으로 1회 주행하며 x/z/속도 트레이스 출력 (정체 지점 진단) |
| `eval_ckpt.py` | 헤드리스 다중 에피소드 평가 (장애물별 통과율, 참고용 — 채점과 달리 reset noise 있음) |
| `dump_map.py` | 지형 장애물 목록 덤프 |
| `bake_norm.py` | VecNormalize pkl → `OBS_NORM_*` 상수 블록 생성 (지금은 쓸 일 없음) |
| `surgery.py` | 관측 차원을 바꿀 때 구 정책 가중치 이식 (K 확대 등) |
| `render.py` | 실시간 렌더링 (`--xml`, VecNormalize 없이도 동작하도록 확인 필요) |
| `check_progress.py` | 체크포인트 진행 확인 |

## 8. 진행 체크리스트

- [ ] Phase A: 공식 코스 파인튜닝 시작, 1M/2M 스텝 평가 (점수: `___` / `___`)
- [ ] Phase B-1: healthy 보상 전진 게이트 (점수: `___`)
- [ ] Phase B-2: 전진 가중치 상향 (점수: `___`)
- [ ] Phase B-3: stall 종료 단축 (점수: `___`)
- [ ] Phase B-4: 계단 보너스 강화 (점수: `___`)
- [ ] Phase C: 최종 체크포인트 선정 + 영상 + 기록

## 9. 리스크 / 주의

| 리스크 | 대응 |
|---|---|
| `OBS_NORM_*` 상수를 바꿔 기존 zip 무효화 | 학습 중 변경 금지. 바꾸면 반드시 재학습 |
| 지형 md5 불일치 (줄바꿈 변환) | `.gitattributes`의 `*.xml -text` 유지, `Get-FileHash`로 확인 |
| 보상 개편 후 기존 보행 붕괴 | 변경은 한 번에 하나, 1M 스텝마다 `evaluate.py` 점수로 판정. 나빠지면 되돌림 |
| 정체 구간에서 healthy 보상 착취 | Phase B-1이 정확히 이 문제. 전진 게이트로 해결 |
| `python`으로 실행해 조용히 실패 | conda `pjt-2` 절대 경로 사용 |
