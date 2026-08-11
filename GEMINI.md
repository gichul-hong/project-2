# Project 2 - GEMINI.md

## Update Log

### 2026-08-11 22:40
- **Gemini 코드 구현 및 실행 가이드 작성**: `practice2/gemini/` 하위에 전략 C 기반 코드 배치
  - `custom_walker2d.py`: z_vel² 페널티, thigh 각속도 부호 반대 보상, bump 접근 감속 유도
  - `learning.py`: VecNormalize + warm-start + [256,256] 네트워크 + ent_coef=0.01
  - `eval_gait.py`: hopping 정량 진단 (balance_ratio, thigh separation)
  - `RUN_GUIDE.md`: Phase 1→2→3 순서별 실행 명령어, 판단 기준, 트러블슈팅

### 2026-08-11 22:33
- **Practice2 전체 수행 전략 수립**: Claude/Deepseek 코드 전수 분석 후 전략 문서 작성
  - 깡총깡총 뛰는 문제의 근본 원인 분석 (hopping이 구조적 local optimum)
  - Deepseek 대칭 페널티의 치명적 결함 발견 (`-|joint_r - joint_l|`이 동시 동작도 보상)
  - 3가지 개선 전략 수립 (전략 C: 복합 개선 권장)
  - Task 3 장애물 통과 개선 방향: 접근 거리 확대, 감속 유도, curriculum learning
  - 실행 계획: Phase 1(걷기 해결) → Phase 2(장애물) → Phase 3(최적화)
