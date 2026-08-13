# B003 이후 Walker2D Controller 개선 내용

## 1. 개선 배경

B003에서는 실제 과제 맵의 50개 범프를 인식하도록 관측을 18차원에서 33차원으로 확장하고, **20초 동안 도달한 최대 x 거리**를 중심으로 보상을 설계하였다. 그 결과 3.48M checkpoint가 47.34m를 이동했지만, 일부 범프에서 양발을 붙인 채 오래 밀거나 고속으로 범프 끝만 넘은 뒤 쓰러지는 문제가 남았다. 또한 TensorBoard의 평균 reward가 증가해도 deterministic 평가 점수는 크게 흔들려 마지막 checkpoint가 항상 최고 모델은 아니었다. B003 이후 개선은 ① 범프 정체 탈출, ② 통과 이후 안정 착지, ③ PPO fine-tuning 안정화, ④ 서버 수치환경 재현성 확보의 네 방향으로 진행하였다.

## 2. 코드 수정사항

**B004—범프 병목 개선.** `custom_walker2d.py`에서 다음 미통과 범프의 `[front-0.3m, back+0.3m]` 구간만 정체를 측정하고, 새 최대 x가 0.05m 증가하면 카운터를 초기화하였다. 범프 높이에 비례한 준비시간 이후에는 `-0.03/step`, 장기 정체에는 초기 `-0.06/step`을 적용했으며 B005에서 `-0.10/step`으로 강화하였다. 인접 범프의 높이 차이를 `rel_height=max(0, current_height-previous_height)`로 계산하여 높은 범프 뒤의 내리막을 새 점프 대상으로 오인하던 문제도 수정했다. 진행 보상 상한은 0.1m/step에서 0.2m/step으로 높여 약 5m/s 이상의 속도 향상도 구분하도록 했다. 기존 33차원 관측의 shape와 순서는 유지해 B003 checkpoint를 그대로 이어 학습할 수 있게 하였다.

**B005—착지와 학습 안정화.** 단순 생존 `+0.03/step`을 제거하고, 범프 끝을 지난 뒤 최소 8 step 후 발 접촉, 지형 대비 torso 높이 `>0.7m`, `|angle|<1.3`, `vz>-1.5m/s`를 만족할 때만 `0.5+bump_height`의 안정 착지 보상을 한 번 지급하도록 했다. 착지 후보는 100 step 또는 3m가 지나면 폐기한다. 범프에 걸린 뒤에는 후진 비용 계수를 5에서 1로 낮춰 뒤로 물러난 후 재도약할 수 있게 했고, 지면 가까이에서 발생하는 급강하와 큰 torso 기울기만 추가로 벌점화했다. `learning.py`에는 checkpoint resume, critic-only warm-up 24,576 transition, 작은 PPO update(LR `1e-5`, entropy `0`, clip `0.05`, epoch 2, batch 1024), 주기적 평가와 `walker_model_best.zip` 자동 저장을 추가하였다.

**서버 호환 보강.** 로컬 60.79m best가 서버에서 약 20m로 하락한 원인은 단순 dtype 문제가 아니라 `obs[30:32]`의 binary foot-contact가 한 step 달라질 때 행동이 크게 바뀌는 수치 민감성이었다. 현재 코드는 contact list 대신 foot clearance가 0 이하인지로 접촉을 재구성하고, 최종 33차원 관측을 소수 둘째 자리로 양자화한다. best 선정도 단일 평가값이 아니라 nominal 평가와 reset noise `1e-8`의 3개 seed 중 최저 거리(`eval/robust_distance`)를 사용한다.

## 3. 최종 Reward 아이디어

핵심 원칙은 **생존 자체가 아니라 새 최대거리, 장애물 극복, 안정적인 다음 동작만 보상**하는 것이다. 현재 주 보상은 다음과 같이 요약된다.

```text
r_progress = 10 · clip(max(0, x - previous_max_x), 0, 0.2)
r_backward = (1 if bump_stuck else 5) · clip(Δx, -0.1, 0)
r_control  = -0.001 · Σ action²
r_posture  = -0.03·max(0, |angle|-0.8)²
r_fall     = -(10 + 0.5·remaining_seconds)
```

여기에 접근·도약·높이 확보·범프 통과 이벤트를 obstacle별 한 번만 지급하고, 실제 안정 착지에 별도 보상을 준다. 왕복 이동으로 진행 보상을 반복 획득할 수 없으며, 정지 생존은 양의 보상을 만들지 않는다. 반면 정상적인 점프 준비 시간은 허용하고, 오래 걸렸을 때는 후퇴와 재시도를 허용하여 정체 패널티가 무조건 전진만 강요하지 않도록 했다. 이 설계는 큰 낙상 보상 하나로 critic 분산을 키우는 대신, 위험한 착지 직전에는 작은 dense cost를 주고 실제 낙상에는 남은 시간에 비례한 제한된 패널티를 주는 방식이다.

## 4. 결과 및 결론

대표 deterministic 점수는 B003 47.34m에서 B004 최고 53.83m, B005 raw best 60.79m로 향상되었다. 다만 raw best는 플랫폼 차이에 매우 민감했으며, portable observation 적용 후에는 로컬 42.18m, 다른 라이브러리 스택 45.78m로 최고점은 낮아졌지만 서버 환경의 급격한 20m대 붕괴를 완화하였다. 따라서 이후 실험에서는 단일 로컬 최고점보다 `robust_distance`, 생존 여부, 안정 착지 수를 함께 사용해 모델을 선정해야 한다. 장기적으로는 binary contact를 완전히 제거하거나 연속 접촉 신호와 작은 관측 노이즈를 포함해 재학습하는 것이 가장 타당하다.
