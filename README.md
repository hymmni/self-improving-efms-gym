# Self-Improving EMFs Gym — Pointmass

Ghasemipour et al. 2025, *Self-Improving Embodied Foundation Models* (SI-EFM)의 공식 pointmass 데모를 재현한 클린 버전입니다. 2D 포인트매스가 목표 지점으로 이동하는 과제에서, 지도학습(Stage 1)만으로 학습한 정책이 자가개선(Stage 2, REINFORCE)을 거치며 성공률과 이동 효율이 향상되는 과정을 재현·관찰합니다.

## 1. 환경 설정

```bash
conda create -n emfs-gym python=3.11 -y
conda activate emfs-gym
pip install -r requirements.txt
```

- GPU(CUDA) 환경을 전제로 하며, `jax`는 `jax[cuda12]`로 설치됩니다. GPU가 없는 환경에서는 `jax[cpu]`로 대체해도 pointmass 규모(작은 MLP)는 CPU만으로 충분히 돌아갑니다.
- `requirements.txt`의 버전은 임의 고정이 아니라, 원본 노트북 코드가 최신 패키지들과 충돌하는 지점을 우회하기 위해 의도적으로 pin된 값입니다 (`jax==0.9.2`, `matplotlib==3.9.4`, `tfp-nightly` 등). 자세한 원인은 `experiments/2026-07-03_clean-repro.md` 참고.

## 2. 실행 방법

전체 파이프라인은 `pointmass_notebook.ipynb` 하나에 담겨 있습니다.

**대화형 실행 (권장 — 학습 곡선을 보며 진행)**
```bash
jupyter lab pointmass_notebook.ipynb
```
셀을 위에서부터 순서대로 실행합니다.

**Headless 일괄 실행 (검증/재현용)**
```bash
jupyter nbconvert --to notebook --execute --output pointmass_notebook_executed.ipynb pointmass_notebook.ipynb
```
모든 셀을 자동 실행하고 결과(로그, 그래프, 산출물)가 담긴 새 `.ipynb` 파일을 생성합니다.

## 3. 파이프라인 — 데이터셋 수집부터 학습까지

노트북은 다음 순서로 구성되어 있으며, 위에서부터 순서대로 실행해야 합니다.

### 3.1 환경 정의 (`Create the pointmass environment`)
`Point2D` (dm_env 인터페이스) — 2D 평면(`[-1, 1]²`)에서 포인트매스가 랜덤 목표 지점으로 이동. 관측값은 `{cur_pos, cur_vel, goal_pos}`, 액션은 2D 가속도, 목표 반경 0.15 이내 도달 시 성공.

### 3.2 시연 데이터 생성 (`Create the PD controller` → `Generate a Dataset`)
- PD 컨트롤러(`pd_controller`)가 목표까지 이동하는 시연 궤적을 생성합니다.
- `Generate a Dataset` 셀에서 여러 에피소드를 굴려 `(observation, action, time_to_success)` 튜플 데이터셋을 만듭니다.
- `save_datasets = True`(기본값)이면 `pointmass_dataset_trajs.pkl`(궤적 전체), `pointmass_dataset_tuples.pkl`(학습용 튜플)로 저장됩니다. 이 파일들은 재실행 때마다 재생성되므로 git에 커밋되지 않습니다(`.gitignore` 참고).

### 3.3 네트워크 정의 (`Create the networks`)
TIMER 네트워크(Haiku MLP) — 하나의 인코더에서 두 갈래로 분기:
- **액션 헤드**: 연속 행동 분포(대각 다변량 정규분포, `tfd.MultivariateNormalDiag`)
- **거리 헤드**: 목표까지 남은 스텝 수(steps-to-go)를 discrete bin에 대한 categorical 분포로 예측 (`Timestep prediction converters` 섹션에서 정의한 bin 변환기 사용)

### 3.4 Stage 1 — 지도학습(SFT) (`Implementations for Stage 1` → `Train Stage 1`)
- 손실 = BC loss(시연 액션에 대한 log-likelihood) + 거리 예측 loss(실제 time-to-success에 대한 log-likelihood)
- 기본 하이퍼파라미터: `global_minibatch_size=256`, `num_minibatches=128` (배치 크기 = 256×128), `learning_rate=3e-4`, `num_steps=32768` SGD step
- 학습 후 `Visualize Policies after the Stage 1` 섹션에서 정책을 굴려 성공률·궤적을 확인합니다.

### 3.5 Stage 2 — 자가개선(Self-Improvement) (`RL Utilities` → `Train Stage 2 Self-Improvement`)
- 외부 보상 없이, **정책 자신의 steps-to-go 예측값이 얼마나 줄었는지**를 REINFORCE reward로 사용합니다: `r_t = -(예측 d_{t+1} - 예측 d_t)`, 할인율 `gamma=0.9`.
- 기본 하이퍼파라미터: `reinforce_global_minibatch_size=64`, `reinforce_global_batch_size=2048`, `num_reinforce_sgd_steps=2048` (`Train Stage 2` 셀 상단에서 조절 가능하며, 이 셀은 여러 번 재실행해 계속 이어서 학습할 수 있습니다).
- 학습 중 4개 그래프(REINFORCE loss, 성공률, return, episode length)가 실시간으로 갱신됩니다.

### 3.6 결과 시각화 (`Visualize policies after the Stage 2` → `Generating Paper Figures and Website Videos`)
- Stage 1 대비 Stage 2 정책의 궤적/성공률을 나란히 비교하는 이미지·영상을 생성합니다 (`ten_tight_pointmass_*.png`, 임시 mp4 등 — 역시 재실행 시 재생성되는 산출물이라 git에는 포함되지 않습니다).

## 4. 검증된 결과 (2026-07-03 기준)

전체 파이프라인을 기본 하이퍼파라미터로 처음부터 끝까지 실행했을 때:

| 단계 | 성공률 | 평균 스텝 수 | 평균 Return |
|---|---|---|---|
| Stage 1 (SFT) 직후 | 92% | 68.9 | -58.6 |
| Stage 2 (Self-Improvement) 시작 시점 | 87% | 65.3 | -53.6 |
| Stage 2 학습 2048 step 후 | **100%** | **~13** | **~-8** |

Stage 2가 진행될수록 성공률이 100%로 수렴하고, 목표 도달에 필요한 스텝 수와 보상이 크게 개선됩니다 — 논문이 주장하는 자가개선 효과가 그대로 재현됩니다. 상세 로그와 의존성 이슈 해결 내역은 `experiments/2026-07-03_clean-repro.md`를 참고하세요.

## 5. 확장 도구 (Enhanced Simulator)

노트북(위 1~4절)이 재현 기준선이라면, 아래 모듈들은 그 위에서 **환경 교란(순간이동·외력·장애물·랜덤액션)을 가하며 steps-to-go 예측 분포를 관찰**하기 위한 도구입니다. 노트북 코드는 수정하지 않고 `pointmass_core.py`(노트북에서 추출한 모듈)를 재사용합니다.

모든 명령은 `conda activate emfs-gym` 상태에서 레포 루트에서 실행합니다. 산출물(`checkpoints/`, `outputs/`)은 재생성 가능하므로 git에 추적되지 않습니다.

### 5.0 코드 구성

| 파일 | 역할 |
|---|---|
| `pointmass_core.py` | 노트북 셀(환경/PD/데이터셋/TIMER 네트워크/학습기/정규화/RL 유틸)을 verbatim 추출한 모듈. 아래 스크립트들이 공통으로 import. |
| `train_sft.py` | Stage 1 SFT 학습 → **체크포인트 저장** (관찰 도구들의 전제) |
| `envs_enhanced.py` | `EnhancedPoint2D` — 순간이동/외력/장애물/랜덤액션 개입 API |
| `stg_probe.py` | steps-to-go **분포 전체**(확률/기댓값/분산/엔트로피) 기록·시각화 |
| `interactive_control.py` | 직접 조종(GUI) + 스크립트 재생(replay) |
| `run_scenarios.py` | 7개 개입 시나리오 배터리 → 경향 플롯·CSV |
| `tests/` | `pytest`로 원본 동일성·개입 정확성 검증 |

### 5.1 SFT 체크포인트 생성 (먼저 1회 실행)

관찰 도구(5.3~5.5)는 학습된 정책·steps-to-go 헤드가 담긴 체크포인트를 필요로 합니다.

```bash
python train_sft.py --num_steps 32768 --eval_episodes 50 --out checkpoints/sft_state.pkl
```
- 노트북과 동일한 하이퍼파라미터로 Stage 1을 학습하고 `checkpoints/sft_state.pkl`을 저장합니다 (params·정규화 통계·bin 설정·메타 포함).
- 주요 옵션: `--dataset_episodes`(기본 10000), `--seed`, `--dataset_steps`(빠른 저품질 실험용).
- GPU에서 수 분 소요. 롤아웃 성공률은 시드에 따라 ~0.84 내외입니다.

### 5.2 환경 개입 API (`EnhancedPoint2D`)

```python
from envs_enhanced import EnhancedPoint2D
import numpy as np

env = EnhancedPoint2D()
env.reset()
env.teleport(np.array([-0.5, 0.5]), zero_velocity=True)   # 순간이동
env.set_bias_force(np.array([2e-5, 0.0]))                 # 외력(매 substep 누적) — None으로 해제
oid = env.add_obstacle(np.array([0.0, 0.0]), radius=0.2)   # 원형 장애물(벽처럼 막힘) — remove_obstacle(oid)/clear_obstacles()
env.set_random_action(prob=0.1, scale=1.5e-3, seed=0)      # 확률적 랜덤 액션 — prob=0으로 해제
env.step(action)
print(env.intervention_log)                                # 에피소드 내 개입 이력
```
**개입이 하나도 활성화되지 않으면 원본 `Point2D`와 수치적으로 완전히 동일**하게 동작합니다 (기준선 보존). 외력은 물리 substep마다(스텝당 10회) 누적되므로 값이 매우 작아야 합니다 (예: `2e-5`).

### 5.3 steps-to-go 분포 관찰 (`stg_probe.py`)

```bash
python stg_probe.py --checkpoint checkpoints/sft_state.pkl --episodes 3 --out outputs/probe/
```
에피소드별로 (분포 evolution 히트맵 + 기댓값±표준편차 + 분산/엔트로피 곡선 + 궤적)을 한 장의 플롯으로 `outputs/probe/`에 저장합니다. 코드에서 직접 쓸 수도 있습니다:
```python
from stg_probe import STGProbe
probe = STGProbe('checkpoints/sft_state.pkl')
rec = probe.query(obs)                 # 환경 스텝 없이 단일 관측의 분포 조회
records = probe.rollout(env)           # 에피소드 롤아웃하며 매 스텝 분포 기록
```

### 5.4 직접 조종 / 재생 (`interactive_control.py`)

**GUI 모드** (모니터/DISPLAY 필요 — 사용자 터미널에서 실행):
```bash
python interactive_control.py --checkpoint checkpoints/sft_state.pkl
```
왼쪽에 환경, 오른쪽에 실시간 steps-to-go 분포가 표시됩니다. 조작: 방향키=이동, `Space`=무입력, `p`=정책 위임, `t`+클릭=순간이동, `o`+클릭=장애물, `b`=외력 토글, `x`=랜덤 토글, `g`=목표 재샘플, `n`=리셋, `s`=세션 저장, `q`=종료.

**Replay 모드** (headless — 스크립트를 재생해 동영상 저장):
```bash
python interactive_control.py --checkpoint checkpoints/sft_state.pkl \
    --replay scripts_replay/demo_session.json --out outputs/interactive/replay.mp4
```
`scripts_replay/demo_session.json`이 이벤트 스키마 예시입니다 (GUI의 `s` 저장 파일도 같은 스키마라 재생 가능).

### 5.5 시나리오 배터리 (`run_scenarios.py`)

```bash
python run_scenarios.py --checkpoint checkpoints/sft_state.pkl --episodes 20 --out outputs/scenarios/
# 일부만: --only teleport_away,baseline
```
7개 시나리오(`baseline`, `teleport_away`, `teleport_near`, `bias_small`, `bias_large`, `obstacle_path`, `random_act`)를 각 `--episodes`회 굴려, 시나리오별 플롯 + 종합 `summary.png` + 원시지표 `summary.csv`를 저장합니다. 개입 크기는 CLI로 오버라이드할 수 있습니다 (`--bias_small`, `--bias_large`, `--random_prob` 등). 관찰 결과 분석은 `experiments/2026-07-13_scenario-battery.md` 참고.

### 5.6 테스트

```bash
python -m pytest tests/ -v
```
`test_core_equivalence.py`(추출 모듈이 노트북과 bit-exact인지)와 `test_env_enhanced.py`(개입 정확성 + 개입 없을 때 원본 동일성)를 검증합니다.

## 6. 참고

- 원본 소스: Ghasemipour et al., *Self-Improving Embodied Foundation Models* (2025)
- 이 레포는 원본 노트북 코드를 그대로 사용합니다(수정 없음). 실행 환경(패키지 버전)만 현재 하드웨어/최신 라이브러리에 맞춰 조정되었습니다.
