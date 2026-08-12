# v9 작업 계획: 새 맵(8범프) 학습 — 범용 관측 + 가중치 수술 (transfer)

> 실행 세션용 작업 지시서. 배경/이력은 `TRAINING_HISTORY.md` 참고 (특히 "교훈" 섹션과 v8.3 보상 구조).
> 원칙: 각 Phase 완료 후 커밋. 학습은 Phase 5에서만 시작.

## 배경 요약

- 구맵(범프 3개, x=6/10/15)에서 v8.3 보상으로 원본 zero-shot bump3 8/10 달성한 정책이 `checkpoints/bump_challenge/`에 있음 (관측 22차원)
- `asset/custom_walker2d_bumps.xml`이 **새 맵(8범프)으로 교체됨**:

| 범프 | pos x | half-width | 높이(top) | 성격 |
|---|---|---|---|---|
| bump1 | 2 | 0.5 | 0.05 | 잔범프 |
| bump2 | 4 | 0.3 | **0.5** | 얇은 벽, 도움닫기 짧음 |
| bump3 | 7 | 0.6 | 0.03 | 잔범프 |
| bump4 | 10 | 0.4 | 0.07 | 잔범프 |
| bump5 | 13 | 0.3 | 0.12 | 잔범프 |
| bump6 | 15 | 0.2 | 0.06 | 잔범프 |
| bump7 | 18 | 0.5 | **0.5** | 계단 1단 (x∈[17.5,18.5]) |
| bump8 | 19 | 0.4 | **1.0** | 계단 2단 (x∈[18.6,19.4]) — 직접 등반 불가, bump7 경유 필수 |

- `custom_walker2d.py`의 `BUMP_CONFIGS`는 구맵 하드코딩 상태 → **현재 코드로 새 맵 학습하면 관측/보상이 전부 어긋남. Phase 2 전까지 학습 금지**
- 전략 결정: 관측을 "다음 K=2개 범프의 (거리, 높이, 폭)"로 재설계(24차원, 맵 불변) + 구 정책 가중치 수술로 이식(옵션 A)

---

## Phase 0 — 백업

1. 현재 상태 커밋 + 태그:
   ```powershell
   cd C:\hong\project-2
   git add practice2
   git commit -m "backup: v8.3 old-map final state before v9 remap"
   git tag v8.3-oldmap-final
   git push; git push --tags
   ```
   주의: 구맵 XML은 이미 새 맵으로 덮어써짐 — git 이력에 구맵 버전이 있는지 `git log -p practice2/asset/custom_walker2d_bumps.xml`로 확인. 없으면 TRAINING_HISTORY.md의 수치(bump2 h=0.6@x=10, bump3 h=0.5@x=15)로 복원본을 `asset/custom_walker2d_bumps_oldmap.xml`에 남길 것.
2. 구맵 체크포인트 이동 (새 학습과 분리):
   ```powershell
   Move-Item practice2/checkpoints/bump_challenge practice2/checkpoints/bump_challenge_oldmap
   ```
3. 가중치 수술에 쓸 **소스 체크포인트 선정**: `checkpoints/bump_challenge_oldmap/`에서 최신 스텝의 zip + 같은 스텝의 `walker_model_vecnormalize_*.pkl` 존재 확인. 이 경로를 기록해둘 것 (Phase 4에서 사용).

## Phase 1 — 범프 자동 파싱 (custom_walker2d.py)

`BUMP_CONFIGS` 하드코딩 제거. `__init__`에서 `super().__init__(env)` 후:

```python
model = self.env.unwrapped.model
self.bumps = []
for i in range(model.ngeom):
    name = model.geom(i).name
    if name.startswith("bump"):
        pos = model.geom(i).pos      # [x, y, z]
        size = model.geom(i).size    # [half_w, half_d, half_h]
        self.bumps.append(dict(name=name, x=float(pos[0]),
                               half_width=float(size[0]),
                               height=float(pos[2] + size[2])))  # top = pos_z + half_h (현재 pos_z=0)
self.bumps.sort(key=lambda b: b["x"])
```

- 통과 판정 x는 `b["x"] + b["half_width"]` (범프 뒷면) 기준으로 변경 — 폭이 다양해져 중심 기준은 부정확
- `goal_milestones`는 기존 로직 유지 (last_x 기반 자동 계산)
- 순서 주의: `reset()`이 `self.bumps`를 참조하므로 파싱을 첫 reset 전에 완료

## Phase 2 — 범용 관측 (24차원 고정)

`custom_observation` 재작성:

```python
K = 2  # 다음 미도달 범프 수
# 미도달 = torso_x < bump.x + bump.half_width 인 것들 중 가까운 순 K개
# 각 범프당 3 feature: [(bump.x - torso_x)/10.0, height, half_width]
# K개 미만이면 [2.0, 0.0, 0.0]으로 패딩 (거리 max=20m 상당, 높이 0)
# 마지막에 torso_z - 1.25 (기존 유지)
```

- 최종 차원: 18(base) + 2*3 + 1 = **25** ← 코드 작성 후 실제 len(obs)를 print로 확인하고 이 문서에 기록
- 통과 순간 슬롯이 한 칸씩 밀리는 불연속은 허용 (VecNormalize가 완화). 학습이 이 때문에 불안정하면 K=3으로 확대

## Phase 3 — 보상 일반화 (v9)

v8.3 구조 유지하되 아래만 변경:

1. **점프/높이 보너스를 높이 상대값으로** (h ≥ 0.2인 범프에만 적용, 잔범프 헛점프 방지):
   - 접근 구간: `b.x - half_width - 2.0 <= x <= b.x + half_width`
   - 도약: 처음 `z_vel > min(1.5, 3.0 * b.height)` 시 +10
   - 몸 띄우기: 처음 `torso_z > b.height + 1.05` 시 +15
2. **계단 처리**: 이전 범프 상단에서 출발하는 경우(직전 범프와 간격 < 1.0m이고 직전이 더 낮음), 도약/높이 기준을 "직전 범프 상단" 기준 상대값으로. bump7(0.5)→bump8(1.0) 구간에서 bump8 보너스 조건이 `torso_z > 2.05`가 되도록
3. **face/pass 마일스톤**: h ≥ 0.2 범프에만 face(+15). pass(+50)는 전체 유지하되 잔범프는 +10으로 축소 (잔범프 6개 × 50이면 보상 인플레)
4. z_vel 벌점 해제 구간: 미통과 h≥0.2 범프 근처 + **h≥0.4 범프 통과 직후 2m(착지 구간)**
5. stall 종료 300 → **500스텝** (계단 등반 시도 시간)
6. 위상 대칭(0.5)·교대 스윙(0.3)·양발목 push-off(0.2)·높이 게이트·낮은자세 벌점: **변경 금지** (검증됨)
7. `near_unpassed_bump` 판정도 h ≥ 0.2 범프만 대상 (잔범프에서 대칭 보상 꺼지면 안 됨)

착취 점검 (TRAINING_HISTORY 교훈 1): 모든 신규 항은 1회성 플래그 or 벌점 해제만인지 확인. 매 스텝 양수 shaping 추가 금지.

## Phase 4 — 가중치 수술 스크립트 (`surgery.py` 신규)

목적: 구 정책(관측 22)의 hidden/output을 새 정책(관측 25)에 이식.

```
절차:
1. 새 관측 차원으로 더미 env 생성 → PPO("MlpPolicy", ...) 새 모델 생성 (learning.py와 동일 policy_kwargs/하이퍼)
2. 구 모델 PPO.load(구 체크포인트, device="cpu")
3. state_dict 비교: 첫 레이어(mlp_extractor.policy_net.0 / value_net.0 등 in_features가 obs_dim인 모든 Linear)만 shape 불일치
4. 첫 레이어 이식 규칙 (구 22 → 신 25):
   - 구 관측 배치: [0..17]=base 18, [18..20]=범프1~3 거리(/20), [21]=z offset
   - 신 관측 배치: [0..17]=base 18, [18..23]=다음2범프(거리/10,높이,폭)x2, [24]=z offset
   - base 18열: 그대로 복사 (신[i] ← 구[i], i=0..17)
   - 신[18](다음 범프 거리): 구[18](첫 범프 거리) 열 복사 후 **가중치 × 0.5** (스케일 /20→/10 보정)
   - 신[21](둘째 범프 거리): 구[19] 열 복사 × 0.5
   - 신[19,20,22,23](높이·폭): 0 초기화
   - 신[24](z offset): 구[21] 열 복사
   - bias: 그대로 복사
5. 나머지 레이어(hidden, action/value head, log_std): 전부 그대로 복사
6. 새 모델 save → checkpoints/bump_challenge/walker_model_surgery_0_steps.zip
7. VecNormalize: 구 pkl의 obs_rms.mean/var를 같은 열 매핑으로 이식한 새 VecNormalize pkl 생성
   (거리 열은 mean×0.5? → 아님: 정규화 통계는 관측값 자체 기준. 신 거리=구 거리×2(스케일 /10)이므로 mean×2, var×4.
    신규 열(높이·폭)은 mean=합리적 초기값(0.3, 0.4), var=1.0)
   ※ 여기가 실수하기 쉬움. 간단 대안: VecNormalize는 이식하지 않고 새로 시작 (norm 통계는 수 십만 스텝이면 수렴).
    대안 채택 시 학습 초기 1~2M은 성능이 낮게 보여도 정상.
8. 검증: 수술본을 render.py로 평지(--bump_challenge --xml c1)에서 열어 걷는지 육안 확인.
    완벽한 보행이 아니어도 "넘어지지 않고 전진 시도"면 성공. 완전 붕괴면 from scratch로 전환 (fallback, 손실 없음)
```

주의: SB3 policy 레이어 이름은 버전에 따라 다름 — `model.policy.state_dict().keys()`를 먼저 출력해서 확인하고 in_features==obs_dim인 레이어를 동적으로 찾을 것.

## Phase 5 — 커리큘럼 XML + 학습

1. 새 맵 기준 커리큘럼 XML 생성 (위치/개수/폭 동일, 높이만 변경):
   - `asset/custom_walker2d_bumps_c1.xml` **덮어쓰기 주의** — 구맵 커리큘럼 파일이 이 이름임. 먼저 `_oldmap_c1.xml`로 rename 백업 후 생성
   - c1: bump2=0.3, bump7=0.3, bump8=0.5 (잔범프 원본 유지)
   - c2: bump2=0.4, bump7=0.4, bump8=0.75
   - 원본: 0.5 / 0.5 / 1.0
2. `eval_ckpt.py` 갱신: bump 통과 집계를 h≥0.2 범프(2,7,8)만 출력하도록. `passed_bumps` 인덱스가 8개로 늘어남에 유의
3. 학습 시작 (persistent 백그라운드):
   ```powershell
   cd practice2
   python -u learning.py --bump_challenge --xml asset/custom_walker2d_bumps_c1.xml --resume checkpoints/bump_challenge/walker_model_surgery_0_steps.zip
   ```
   - `--resume`가 vecnormalize pkl을 못 찾으면 새 VecNormalize로 시작함 (learning.py 기존 로직, Phase 4의 간단 대안과 일치)
   - resume 시 ent_coef=0.01 적용됨 (기존 로직)
4. 모니터링: `python check_progress.py` (bump_challenge 폴더 glob은 그대로 동작). 판독 기준은 기존과 동일 (ep_rew≈ep_len이면 정지 국소최적)
5. 승급 기준: 해당 단계에서 mean_len 800+ & bump8 통과 8/10+ (`eval_ckpt.py --episodes 10`)
6. c1 → c2 → 원본. 각 단계 예상 3~6M 스텝 (bump2가 x=4로 이른 편이라 c1 초기에 ep_len이 짧아도 정상)

## Phase 6 — 기록

- `TRAINING_HISTORY.md`에 v9 섹션 추가: 관측 재설계, 수술 결과(성공/실패), 각 단계 승급 스텝
- 이 파일(PLAN_V9.md)의 체크박스를 진행하며 갱신

## 진행 체크리스트

- [ ] Phase 0: 백업 커밋 + 태그 + 구맵 체크포인트 이동 + 소스 체크포인트 선정: `______`
- [ ] Phase 1: 범프 자동 파싱
- [ ] Phase 2: 관측 25차원 (실측: `__`차원)
- [ ] Phase 3: 보상 v9
- [ ] Phase 4: surgery.py + 육안 검증 (성공/실패: `______`)
- [ ] Phase 5: 커리큘럼 XML + c1 학습 시작
- [ ] c1 졸업 (스텝: `__M`)
- [ ] c2 졸업 (스텝: `__M`)
- [ ] 원본 bump8 8/10+ 달성
- [ ] Phase 6: TRAINING_HISTORY.md v9 기록

## 리스크 / fallback

| 리스크 | 대응 |
|---|---|
| 수술본이 완전 붕괴 | from scratch (surgery zip 대신 신규 학습). 하방 손실 없음 |
| K=2 슬롯 불연속으로 학습 불안정 | K=3 확대 (수술 매핑도 거리열 하나 더 복사) |
| c1 bump2(0.3)에서도 정체 | bump2만 0.2로 낮춘 c0 추가 |
| 잔범프에서 발 걸려 넘어짐 반복 | 위상 대칭 가중치 0.5→0.8 상향 검토 (그 전에 렌더링으로 원인 확인) |
| VecNormalize 새로 시작에 따른 초기 저성능 | 정상. 2M 스텝까지는 판단 유보 |
