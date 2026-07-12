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

## 5. 참고

- 원본 소스: Ghasemipour et al., *Self-Improving Embodied Foundation Models* (2025)
- 이 레포는 원본 노트북 코드를 그대로 사용합니다(수정 없음). 실행 환경(패키지 버전)만 현재 하드웨어/최신 라이브러리에 맞춰 조정되었습니다.
