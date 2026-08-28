# archive/ 파일 색인

pointmass 계열의 과거 진단·실험 스크립트 이력(전부 `pointmass_core` 또는 그 파생인
`envs_enhanced`/`obstacle_env`/`multimodal_env` 등에 의존). 어떤 현재 실행 코드도 이걸
import하지 않는 죽은 코드이며, 코드 품질도 정리된 상태가 아니다 — **재사용 목적이
아니라, pointmass를 다시 들여다볼 때 "이런 걸 예전에 해봤다"를 찾아보는 색인**이다.
비슷한 걸 다시 만들 땐 여기서 아이디어만 참고하고 코드는 새로 짜는 걸 권장한다.

### 환경 구현
- `envs_enhanced.py` — 순간이동/외력/장애물/랜덤액션 개입 훅을 추가한 `pointmass_core.Point2D` 확장(개입 없으면 원본과 완전 동일 보장).
- `src/obstacle_env.py` — 관측 가능한 장애물 회피 환경(`ObstacleAvoidPoint2D`/`TwoObstacleAvoidPoint2D`/`PartialObsObstacleAvoidPoint2D`) — 웨이포인트가 만드는 인위적 다봉성을 없애고 장애물 우회만으로 자연스러운 비효율을 구성.
- `src/multimodal_env.py` — 장애물 하나를 사이에 두고 좌/우 우회가 50/50으로 갈리는, steps-to-go 라벨이 의도적으로 이봉인 도달 환경.

### 예측기/정책 학습
- `train_sft.py` — Stage 1 SFT(원본 노트북 재현) — TIMER 네트워크를 학습해 `checkpoints/sft_state.pkl` 생성.
- `src/train_predictor.py` — 멀티모달맵 예측기를 여러 학습 fraction 스냅샷으로 저장(예측기 품질 ablation용).
- `src/train_obstacle_predictor.py` — 장애물 회피 환경(5필드 관측) STG 예측기 + BC 정책 학습.
- `src/train_two_obstacle_predictor.py` — 위 스크립트를 2-장애물 환경(12필드 관측)으로 확장.
- `src/train_pusht_predictor.py` — PushT(사람 데모) STG 예측기 + BC 정책 학습 — 결정론적 전문가의 다봉성 한계 때문에 태스크를 옮긴 결과물.
- `src/skill_chaining_predict.py` — 갈림길 해소까지 남은 스텝을 사전에 예측하는 보조 헤드(사후 라벨링, HER 방식).

### 자기개선(REINFORCE) 실험
- `src/reinforce.py` — 멀티모달맵에서 pluggable reward(baseline vs 분산반영)로 REINFORCE 자기개선.
- `src/reinforce_obstacle.py` — 장애물 환경에 REINFORCE 자기개선을 이식(성공률 대신 에피소드 길이를 개선 지표로 사용).
- `src/run_e2.py` — E2: baseline vs 분산반영 reward의 다중시드 수렴곡선 비교 + alpha:beta/분산항 스윕(옛 `configs/e2_baseline.json`·`configs/e2_ours.json`의 값이 이 파일 57-60행에 하드코딩되어 있음).
- `src/run_e2_obstacle.py` — 장애물 환경판 E2(에피소드 길이 지표).
- `src/run_e3.py` — E3: 예측기 품질(MAE)별로 baseline 대비 개선폭이 유지되는지 비교.
- `src/run_sweeps.py` — decision-item 스윕(alpha:beta, 분산항 종류, eps 수치안정성) + 열화된 시작점에서의 E2.
- `src/run_experiment.py` — e0/e1/e1b/e2/sweeps/e3 실험들의 단일 진입점 디스패처.
- `src/experiments_observe.py` — E0(이봉 붕괴)/E1(상황별 분포)/E1b(Δμ·Δσ² 독립성) 관측 실험.

### 진단/분석
- `analyze_stg_gap.py` — 사람 데모 vs 정책 롤아웃의 STG 라벨을 최근접이웃 매칭으로 직접 비교(값싼 사전검증).
- `plot_gap_progression.py` — 학습 정도(수렴 전/후)에 따라 STG gap의 상태 종속 구조가 어떻게 바뀌는지 비교.
- `residual_sigma_2d.py` — 거리(μ) 효과를 구간 제거한 뒤에도 σ가 예측 오차를 추가로 설명하는지 상관검정.
- `src/analyze_mu_sigma_relationship.py` — μ-σ² 전역 관계(상관, 캘리브레이션, 진행단계별 조건부 분포) 정량화.
- `src/diagnose_multimodal.py` — STG 다봉성/분산급증 원인을 PD 진동 vs 웨이포인트 정체성 모호성 가설로 진단.
- `src/diagnose_obstacle_multimodal.py` — 웨이포인트 제거(신맵)로 인위적 다봉 사다리가 실제로 사라졌는지 구맵과 비교검증.
- `src/diagnose_switch_sign.py` — 웨이포인트 전환 순간 분산 증감의 부호를 결정하는 요인 회귀분석.
- `src/diagnose_variance_spikes.py` — 분산 급변(스파이크)을 봉우리간 질량이동/입력귀속/민감도지형/선형화 4갈래로 정확분해.
- `src/exp_reducible_uncertainty.py` — 앙상블 전분산분해로 epistemic(줄일 수 있는) vs aleatoric(타고난) 불확실성을 분리검증.
- `src/exp_two_obstacle_uncertainty.py` — 위 방법을 2-장애물 게이트 시나리오(12차원 관측)로 확장.
- `calibrate.py` — Phase 0: 신경망 없이 순수 동역학 시뮬로 GraspAngleTransport2D 파라미터 영역에서 기댓값-분위수 순위역전 조건 탐색.
- `src/skill_chaining_detect.py` — σ²가 평소보다 높아지는 구간을 갈림길로 보는 반응형 스킬 경계 탐지(재학습 없음).

### PushT 비교/시각화
- `src/dp_policy.py` — 사전학습 LeRobot Diffusion Policy를 gym-pusht에서 굴리는 헬퍼(lerobot 0.6.0 스키마 불일치를 수동 정규화로 우회, 별도 lerobot conda env 전용).
- `viz_pusht_compare.py` — 사람 데모 vs DP 롤아웃을 몽타주 PNG/나란히 mp4로 비교 렌더링.
- `src/viz_multimodal.py` — 멀티모달맵의 좌/우 데모 경로 + 시작 지점 이봉 STG 히스토그램 시각화.

### 뷰어/녹화 유틸
- `watch_obstacle.py` — 장애물 회피 환경 실시간 뷰어(에이전트+STG 분포 동시 관찰).
- `watch_stdmap.py` — 구맵 웨이포인트 환경에서 구/신 예측기 STG 분포를 겹쳐 실시간 비교.
- `drive_obstacle.py` — 마우스로 장애물 환경 에이전트를 직접 몰며 STG 분포 반응을 실시간 관찰.
- `interactive_control.py` — 키보드로 조종하며 개입(순간이동/외력/장애물/랜덤액션)을 트리거하는 GUI, 또는 JSON 이벤트 스크립트를 헤드리스로 재생해 mp4 저장(`--replay demo_session.json --out outputs/interactive/replay.mp4`).
- `demo_session.json` — `interactive_control.py --replay`용 예시 이벤트 스크립트(순간이동/장애물/외력 개입 시퀀스).
- `record_obstacle.py` — `watch_obstacle.py`와 동일 화면을 디스플레이 없이 mp4로 녹화.
- `record_partial_obs.py` — 부분관측 환경에서 장애물을 센싱하는 순간 STG 이봉이 붕괴하는 과정을 녹화.
- `record_skill_chaining.py` — 스킬 체이닝 경계 탐지(σ² 기반) 과정을 mp4로 녹화.
- `record_stdmap.py` — `watch_stdmap.py --probe both`와 동일 화면을 mp4로 녹화.
- `record_waypoint.py` — 웨이포인트 데모/학습 정책 주행을 mp4로 녹화(정리 문서용).
- `run_scenarios.py` — teleport/bias/obstacle/random 등 7종 개입 시나리오 배터리를 굴려 STG 분포 추이 플롯+CSV로 덤프.
- `src/probe_generic.py` — 3필드/5필드 등 관측 구성이 다른 체크포인트를 모두 다루는 범용 STG 프로브(`stg_probe.STGProbe`의 일반화판).
- `stg_probe.py` — STG 카테고리컬 분포 전체(기댓값뿐 아니라 분포 형태)를 기록/시각화하는 핵심 관찰 도구.

### 테스트
- `tests/test_core_equivalence.py` — `pointmass_core`가 원본 노트북 셀(exec로 로드)과 비트단위로 동일한 결과를 내는지 검증.
- `tests/test_env_enhanced.py` — `EnhancedPoint2D`가 개입이 없을 때 부모 `Point2D`와 수치적으로 완전히 동일한지 검증.
- `tests/test_multimodal_env.py` — 멀티모달 도달 맵의 리셋/장애물 배치 등 기본 동작 검증.
