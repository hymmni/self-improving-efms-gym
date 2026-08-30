# 명령어 총정리

이 레포는 스택이 다른 두 연구 트랙을 Docker 컨테이너 하나 + venv 2개로 나눠 돌린다
(상세는 `DOCKER.md` 참고). 모든 명령은 컨테이너 안, 레포 루트(`/workspace`)에서 실행한다.

```bash
docker compose run --rm dev bash
source /opt/venvs/jax/bin/activate     # grasp_carry
source /opt/venvs/torch/bin/activate   # square_assembly
```

GUI(온스크린 렌더링)가 필요한 명령은 호스트에서 `xhost +local:docker`를 먼저 실행해야
한다(`DOCKER.md` 4절). `pointmass`는 현재 휴면 트랙이라 이 문서에 없다 — 과거 실험
스크립트 색인은 `projects/pointmass/archive/README.md` 참고.

---

## 1. grasp_carry — 학습/데이터/시각화 (JAX/Haiku)

패키지 경로: `python -m grasp_carry.scripts.<train|collect|record|analyze>.<모듈명>`.
Docker `jax` venv가 `PYTHONPATH=/workspace/projects/grasp_carry/src`를 설정하므로
바로 실행된다. 로컬(비-Docker)에서는 `PYTHONPATH=projects/grasp_carry/src`를 직접
설정한다.

### train/

#### train_carry_actor.py
AI-E/AI-R 보상(qstg critic)으로 "잡은 직후 속도"를 고르는 소형 신경망(actor)을 학습한다. 실제 롤아웃 없이 예측기에만 역전파(DDPG류 actor 업데이트) — 학습 후에만 실제 시뮬레이터로 채점(`eval_carry_actor.py`).

```bash
python -m grasp_carry.scripts.train.train_carry_actor \
    --qstg-ckpt checkpoints/grasp_carry_qstg/predictor.pkl \
    --out checkpoints/grasp_carry_actor_risk/actor.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--qstg-ckpt` | (필수) | 얼려서 critic으로 쓸 액션-조건부 STG 예측기 체크포인트 |
| `--data` | `data/grasp_carry_demos_v3.pkl` | 상태(관측) 소스 — `is_held` 필터링 후 학습 |
| `--steps` | 3000 | 학습 스텝 수 |
| `--objective` | `mean` | `mean`/`cvar`/`renewal` 중 보상 통계 선택 |
| `--speed-penalty` | 0.1 | `P(성공) - 이 값*예측 스텝통계` (mean/cvar 전용) |

#### finetune_carry_diffusion.py
사전학습된 디퓨전 BC 정책을 qstg critic으로 직접 파인튜닝한다(역확산 전체가 미분 가능해서 critic 그래디언트를 바로 역전파). 원본 정책과의 L2 벌점(`--bc-reg`)으로 보상 해킹을 막는다.

```bash
python -m grasp_carry.scripts.train.finetune_carry_diffusion \
    --diff-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
    --qstg-ckpt checkpoints/grasp_carry_qstg/predictor.pkl \
    --out checkpoints/grasp_carry_diff100_finetuned/predictor.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--diff-ckpt` | `checkpoints/grasp_carry_diff100/predictor.pkl` | 파인튜닝할 사전학습 디퓨전 정책 |
| `--qstg-ckpt` | `checkpoints/grasp_carry_qstg/predictor.pkl` | 얼려서 critic으로 쓸 예측기 |
| `--lr` | 1e-5 | 사전학습(3e-4)보다 훨씬 작게 — 급격한 붕괴 방지 |
| `--bc-reg` | 1.0 | 원본 정책 대비 L2 벌점 가중치 |
| `--steps` | 1000 | 파인튜닝 스텝 수 |

#### train_carry_actor_reinforce.py
원논문(SI-EFM) Stage-2 Algorithm 1을 실제 on-policy REINFORCE로 그대로 구현한 버전 — `train_carry_actor.py`와 달리 실제 환경 롤아웃을 모아 몬테카를로 리턴으로 정책 그래디언트 업데이트를 한다. 대상은 확률적 스칼라 액터(속도 선택).

```bash
python -m grasp_carry.scripts.train.train_carry_actor_reinforce \
    --diff-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
    --out checkpoints/grasp_carry_actor_reinforce/actor.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--diff-ckpt` | `checkpoints/grasp_carry_diff100/predictor.pkl` | d(o,g) 계산 + 초기 정규화 통계용, 얼려지는 Stage-1 체크포인트 |
| `--iterations` | 30 | 수집→업데이트 반복 횟수 |
| `--episodes-per-iter` | 64 | 매 반복 실제로 굴릴 에피소드 수 |
| `--explore-std` | 1.0 | 탐색용 잠재값 z의 표준편차 |
| `--reinforce-scale` | 1.0 | REINFORCE 손실 크기(논문의 상수 c) |

#### train_carry_si.py
DDPO-SF(score-function REINFORCE, Black et al. 2023)로 디퓨전 파지·운반 정책을 자기개선하는 SI-EFM Stage-2(Algorithm 1) 메인 학습 루프. 보상은 얼려진 관측-only STG 예측기(`--d-ckpt`)에서 나오는 steps-to-go 감소분이며, 환경의 성공/실패 신호는 학습에 쓰이지 않는다(로그 전용). 옵션이 25개 이상으로 가장 복잡한 스크립트 — 핵심만 적는다, 전체는 `--help`.

```bash
python -m grasp_carry.scripts.train.train_carry_si \
    --policy-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
    --d-ckpt checkpoints/grasp_carry_dstg_succ/predictor.pkl \
    --statistic mean --iterations 100 --episodes-per-iter 32 \
    --gamma 0.9 --reinforce-scale 5e-2 --lr 1e-5 --termination learned \
    --out checkpoints/grasp_carry_si_mean_succ/predictor.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--d-ckpt` | (필수) | 얼려서 보상(d(o,g))으로 쓸 관측-only STG 예측기 |
| `--policy-ckpt` | `checkpoints/grasp_carry_diff100/predictor.pkl` | 자기개선시킬 디퓨전 정책 |
| `--statistic` | `mean` | `mean`/`cvar` — 보상을 예측 분포의 평균/CVaR 중 무엇으로 잴지 |
| `--iterations` / `--episodes-per-iter` | 100 / 32 | 수집→업데이트 반복 횟수 / 반복당 롤아웃 수 |
| `--reward-v2` | off | R-learning식 자기보정 대신 고정 스텝비용+PBRS 보상 사용(대안 실험) |

### collect/

#### collect_carry_demos.py
스크립트 정책(`ScriptedCarryPolicy`)으로 GraspCarry2D를 굴려 Stage-1 성공 데모를 수집한다(SI-EFM 논문과 동일하게 성공 에피소드만 저장, `time_to_success` 라벨). 기본은 정책의 안전 속도식 그대로라 실패가 거의 없는 게 정상 — 인위적 실패가 필요하면(v3 재현) `--explore-range`.

```bash
python -m grasp_carry.scripts.collect.collect_carry_demos \
    --episodes 500 --out data/grasp_carry_demos_v4.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--episodes` | 500 | 수집할 에피소드 수 |
| `--explore-range LOW HIGH` | (없음) | 레거시 v3 재현용 — 안전식 무시하고 무작위 속도로 강제 실패 유발 |
| `--keep-failures` | off | 실패 에피소드도 저장(기본은 성공만) |
| `--detour-prob` | 0.0 | 파지 직후 확률적으로 무작위 지점 경유(OOD 상태 수집용) |
| `--out` | `data/grasp_carry_demos.pkl` | 저장 경로 |

#### collect_carry_bc_rollouts.py
스크립트가 아니라 **학습된** 디퓨전 BC 정책을 실제로 굴려서 성공+실패가 섞인 Stage-2 스타일 롤아웃을 `collect_carry_demos.py`와 동일 포맷으로 저장한다 — fail-aware STG 예측기(`train_carry_dstg.py --include-failures`) 학습용 데이터.

```bash
python -m grasp_carry.scripts.collect.collect_carry_bc_rollouts \
    --diff-ckpt checkpoints/grasp_carry_diff100_v5/predictor.pkl \
    --episodes 600 --seed0 900000 \
    --out data/grasp_carry_bc_v5_rollouts.pkl
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--diff-ckpt` | `checkpoints/grasp_carry_diff100_v5/predictor.pkl` | 굴릴 학습된 디퓨전 정책 |
| `--episodes` | 600 | 수집할 에피소드 수 |
| `--seed0` | 900000 | 시작 시드 — 학습 데이터 시드 대역(<900000)과 안 겹치는 held-out |
| `--out` | `data/grasp_carry_bc_rollouts.pkl` | 저장 경로 |

#### collect_carry_teleop_detour.py
GUI 필요(디스플레이 있는 PC 전용). 정책이 파지 후 무작위로 방황하다 실제로 놓치면 자동으로, 또는 `h` 키로 언제든 수동으로 사람이 마우스로 개입해 복구하는 **비최적 데모** 수집 도구. `collect_carry_demos.py`와 동일 스키마 + `is_human` 필드.

```bash
python -m grasp_carry.scripts.collect.collect_carry_teleop_detour \
    --out data/grasp_carry_demos_teleop_detour.pkl
```

키 조작: 마우스 이동=그리퍼 목표, 좌클릭 유지=grip 닫힘, `h`=자동/사람 모드 토글, `r`=에피소드 버리고 재시작, `q`=저장 후 종료.

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--seed0` | 800000 | 시작 시드 |
| `--waypoint-min-steps`/`--waypoint-max-steps` | 10 / 25 | 자동 방황 중 경유지 유지 스텝 범위 |
| `--autosave-every` | 5 | 몇 에피소드마다 `--out`에 자동 저장할지 |

### record/

#### record_carry.py
같은 시드(같은 블록·은닉 물성)를 느린 속도 vs 빠른 속도로 나란히 굴려 "속도가 곧 위험"을 보여주는 mp4 녹화. `draw_env`가 다른 record 스크립트들의 렌더링 기반 함수.

```bash
python -m grasp_carry.scripts.record.record_carry \
    --seeds 3 7 11 --speeds 30 60 --out results/videos/grasp_carry.mp4
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--seeds` | `[3, 7, 11]` | 비교할 시드 목록 |
| `--speeds` | `[28.0, 58.0]` | 느린/빠른 속도 두 값 |
| `--explore-range LOW HIGH` | (없음) | 켜면 안전식 대신 무작위 속도 강제 |

#### record_carry_actor.py
`train_carry_actor.py`가 학습한 AI-E actor vs AI-R actor를 같은 시드로 나란히 녹화.

```bash
python -m grasp_carry.scripts.record.record_carry_actor \
    --seeds 3 7 11 --out results/videos/grasp_carry_actor_compare.mp4
```

| 옵션 | 기본 |
|---|---|
| `--exp-actor` | `checkpoints/grasp_carry_actor_exponly/actor.pkl` |
| `--risk-actor` | `checkpoints/grasp_carry_actor_risk/actor.pkl` |

#### record_carry_si.py
DDPO 적용 전(control) vs 후(SI arm) 디퓨전 정책을 같은 시드로 나란히 녹화. **주의**: 디퓨전 샘플링 노이즈 키가 `reset()`으로 안 되돌아가므로, `eval_carry_si.py`와 같은 결과를 재현하려면 `--seed0`부터 순서대로 다 호출해야 한다(그려서 저장하는 건 `--seeds`로 고른 것만).

```bash
python -m grasp_carry.scripts.record.record_carry_si \
    --seeds 900004 900006 900022 900039 \
    --out results/videos/grasp_carry_si_compare.mp4
```

| 옵션 | 기본 |
|---|---|
| `--control-ckpt` | `checkpoints/grasp_carry_diff100/predictor.pkl` |
| `--si-ckpt` | `checkpoints/grasp_carry_si_A_mean_succ/predictor.pkl` |
| `--seed0` | 900000 |

#### record_carry_si_video.py
DDPO-SF로 자기개선된(또는 순수 BC) 디퓨전 정책 하나를 실제로 굴려 STG 분포와 함께 녹화. `record_carry_stg_dist.py`(스크립트 정책 전용)의 학습된-정책 버전.

```bash
python -m grasp_carry.scripts.record.record_carry_si_video \
    --seed 3 --diff-ckpt checkpoints/grasp_carry_si_v5n50_successonly/predictor_best.pkl \
    --out results/videos/si_successonly_seed3.mp4
```

| 옵션 | 기본 |
|---|---|
| `--diff-ckpt` | (필수) |
| `--dstg-ckpt` | `checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl` |

#### record_carry_stg_dist.py
실패로 가는 롤아웃(스크립트 정책)에 STG 카테고리컬 분포를 나란히 붙여 녹화 — 실패로 흘러가는 동안 분포가 퍼지는지/쏠리는지 시간축으로 보여준다. 빨간 세로선 = 데드라인 B.

```bash
python -m grasp_carry.scripts.record.record_carry_stg_dist \
    --seed 52 --out results/videos/grasp_carry_stg_dist_seed52.mp4
```

| 옵션 | 기본 |
|---|---|
| `--dstg-ckpt` | `checkpoints/grasp_carry_dstg_succ_v4/predictor.pkl` |
| `--xmax` | 120 |

#### record_carry_bc_stg_dist.py
`record_carry_stg_dist.py`와 같은 화면 구성이지만, 액션이 스크립트 정책이 아니라 실제 학습된 디퓨전 BC 정책의 **실패** 롤아웃에서 나온다.

```bash
python -m grasp_carry.scripts.record.record_carry_bc_stg_dist \
    --seed 900003 --out results/videos/grasp_carry_bc_fail_seed900003.mp4
```

| 옵션 | 기본 |
|---|---|
| `--diff-ckpt` | `checkpoints/grasp_carry_diff100_v5/predictor.pkl` |
| `--dstg-ckpt` | `checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl` |

### analyze/

1회성 연구 진단 스크립트 모음. 옵션은 각 파일 `--help` 참고.

- `analyze_mu_jump_bimodal.py` — 파지 직후 STG 예측 μ 급등이 "쌍봉분포 붕괴 + 데드라인 B 이하에서의 상승"으로 설명되는지 검증.
- `analyze_mu_sigma_highrisk.py` — 고위험(μ 큰) 구간에서도 μ와 σ가 여전히 같이 움직이는지, 아니면 평균-기준 포기판정으로는 부족한지 검증.
- `calibrate_carry.py` — GraspCarry2D의 위험 민감 구조(얕은/깊은 파지 분리, 재파지 효용, 속도-위험 트레이드오프)가 실제 물리에서 성립하는지 검증하는 캘리브레이션 스윕.
- `compare_carry_selectors.py` — 기댓값 전용 STG 예측기(성공만 학습) vs 위험 인지 STG 예측기(실패 포함)로 고른 속도의 자동 데이터 수집 효율(스텝당 성공 데모 수)을 비교.
- `eval_carry_actor.py` — `train_carry_actor.py`로 학습한 AI-E/AI-R actor 정책을 실제 시뮬레이터로 채점.
- `eval_carry_si.py` — control(자기개선 없음) + DDPO-SF arm 3종을 같은 시드로 채점해 비교.
- `evaluate_stg_deadline.py` — fail-aware STG 예측기의 "μ(o) > 데드라인 B" 포기판정이 held-out 에피소드에서 실제 실패만 걸러내는지 평가.
- `evaluate_stg_deadline_cdf.py` — 포기판정에 μ만 쓰는 것보다 σ(분산)나 CDF 꼬리확률까지 같이 쓰면 재현율이 오르는지 비교.
- `probe_carry_qstg.py` — 학습된 액션-조건부 STG 예측기(qstg)에 가상의 느린/빠른 액션을 넣어 조건 A/B(속도-위험 구조)를 직접 재현.
- `rollout_carry_diff_stats.py` — 스크립트 정책이 아니라 학습된 디퓨전 정책으로 굴렸을 때 재파지율·에피소드 길이 등 통계가 어떻게 달라지는지 재측정.
- `run_bc_stg_guided.py` — 디퓨전 정책이 뽑은 K개 후보 액션을 STG 예측기로 채점해 하나를 고르는 best-of-K 가이던스 정책의 실행/평가 엔진(다른 analyze/record 스크립트들이 이걸 재사용).
- `verify_carry_qstg_condb.py` — `probe_carry_qstg.py`가 예측만으로 찾은 "조건 B" 상태들을 실제로 그 시드부터 물리 롤아웃해서 재검증.

---

## 2. square_assembly — 학습/데이터/시각화 (PyTorch/robomimic)

패키지 경로: `python -m square_assembly.scripts.<모듈명>`. Docker `torch` venv가
`PYTHONPATH=/workspace/projects/square_assembly/src`를 설정하므로 바로 실행된다.
`train*.py`/`eval.py`는 Hydra 기반(`key=value` 오버라이드), `collect_square_rollouts.py`·
`record_*.py`는 argparse 기반(`--flag value`)이다. 용어(task/env/dataset/benchmark
구분)와 이 서브셋에서 빠진 기능은 `projects/square_assembly/README.md` 참고.

#### train.py
통합 학습 진입점 — `policy_name`/`runner_name`(factory registry 키)으로 bc/diffusion/openvla × low_dim/image 조합을 전부 이 스크립트 하나로 다룬다. Hydra 기반.

```bash
python -m square_assembly.scripts.train use_wandb=true                                    # DP(image), 기본 300 epoch
python -m square_assembly.scripts.train task=square_stage                                  # DP(image) + stage conditioning
python -m square_assembly.scripts.train task=lift_low_dim policy=diffusion_unet_lowdim policy_name=diffusion_lowdim
python -m square_assembly.scripts.train resume=true                                        # 중단된 학습 이어받기
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `task` | `square` | 태스크 config (`configs/task/*.yaml`) |
| `policy_name` / `runner_name` | `diffusion` / `diffusion_trainer` | factory registry 키 — 실제 클래스 결정 |
| `num_epochs` | 300 | 학습 epoch 수 |
| `batch_size` | 64 | 배치 크기 |
| `lr` | 1e-4 | 학습률 (system.kind=apo 파인튜닝 시 1e-6~1e-7로 낮춰 override) |
| `use_wandb` | false | wandb 로깅 |

전체 옵션(weighting/system=apo 등 SIRIUS·APO 관련 축 포함)은 `configs/train.yaml` 참고.

#### train_dstg.py
`DstgPredictor`(d(o,g) := E[steps-to-go | o]) 학습 — square task, robomimic PH(성공만) 데모 대상. task/policy는 hydra defaults가 아니라 `policy_ckpt` 옆 `run_config.yaml`에서 복원한다(그 체크포인트가 실제로 학습된 설정의 유일한 출처).

```bash
python -m square_assembly.scripts.train_dstg \
    policy_ckpt=/path/to/policy_epoch1060.pt \
    hdf5_path=/path/to/square_image_v15.hdf5 \
    out=outputs/dstg/square_demo50_succ/predictor.pt
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `policy_ckpt` | (필수) | 대상 diffusion policy 체크포인트 |
| `hdf5_path` | (필수) | robomimic PH 데모 hdf5. 학습 당시 상대경로는 이 머신에서 안 열리므로 항상 명시 override |
| `out` | `outputs/dstg/predictor.pt` | 저장 경로 |
| `num_epochs` | 30 | — |
| `batch_size` / `lr` | 64 / 1e-4 | — |

PH 데모는 전부 성공 시연이라 succ 버전만 만든다 — 실패 라벨 포함 버전은 `train_dstg_failaware.py`.

#### train_dstg_failaware.py
`train_dstg.py`와 동일 목적이지만 성공+실패가 섞인 롤아웃 hdf5(`collect_square_rollouts.py` 산출물) 대상. 실패 transition은 전부 별도 클래스(`fail_bin`)로만 라벨링(몇 스텝 뒤 실패인지는 라벨링 안 함 — 근거는 grasp_carry의 `train_carry_dstg.py --fail-mode class`와 동일).

```bash
python -m square_assembly.scripts.train_dstg_failaware \
    policy_ckpt=/path/to/policy_epoch1060.pt \
    hdf5_path=data/square_rollouts_v1.hdf5 \
    out=outputs/dstg/square_failaware/predictor.pt
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `policy_ckpt` / `hdf5_path` | (필수) | 대상 policy / 성공+실패 혼합 hdf5 |
| `out` | `outputs/dstg/predictor_failaware.pt` | 저장 경로 |
| `num_epochs` | 30 | — |

나머지 옵션은 `train_dstg.py`와 동일(`configs/train_dstg_failaware.yaml` 참고).

#### train_si.py
DDPO-SF 자기개선(Self-Improvement) 학습 루프 — 학습된 `DstgPredictor`(d)의 감소분을 보상으로 삼아 diffusion policy를 REINFORCE로 파인튜닝(외부 reward 불필요, GraspCarry2D의 `train_carry_si.py`와 동일 원칙).

```bash
python -m square_assembly.scripts.train_si \
    policy_ckpt=/path/to/policy_epoch1060.pt \
    dstg_ckpt=outputs/dstg/square_demo50_succ/predictor.pt \
    iterations=50 episodes_per_iter=8 \
    out=outputs/si/square_si/predictor.pt
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `iterations` / `episodes_per_iter` | (필수, `???`) | REINFORCE 이터레이션 수 / 이터레이션당 롤아웃 에피소드 수 — 1 에피소드 실측 시간으로 정할 것 |
| `dstg_ckpt` | `outputs/dstg/square_demo50_succ/predictor.pt` | 보상 신호로 쓸 DstgPredictor |
| `statistic` | `mean` (`mean`\|`cvar`) | DstgReward 집계 방식 |
| `termination` | `learned` (`learned`\|`env`) | 에피소드 종료 판정 기준(보상은 항상 d로만 계산) |
| `lr` | 1e-6 | 파인튜닝이라 사전학습 lr(1e-4)보다 훨씬 작게 |
| `out` | (필수, `???`) | 저장 경로 |

매 iteration `env_succ_rate` 갱신 시 `<out>_best.pt`로 best-so-far 별도 저장, `train_stats.jsonl`에 iteration별 통계 로그.

#### eval.py
통합 rollout 평가(low_dim·image 정책 공용). `render=true`면 MuJoCo 온스크린 뷰어로 화면에 띄운다. OpenVLA는 인터페이스가 달라 `policy_name=="openvla"`일 때 별도 루프로 분기.

```bash
python -m square_assembly.scripts.eval checkpoint_path=outputs/train/square_diffusion_unet/policy_epoch300.pt
python -m square_assembly.scripts.eval task=square_stage checkpoint_path=... render=true
unset MUJOCO_GL  # low_dim 화면(mjviewer)일 때만 필요
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `checkpoint_path` | (필수) | 평가할 policy 체크포인트 |
| `use_run_config` | true | 체크포인트 옆 `run_config.yaml`에서 task/policy 자동 복원(학습·평가 설정 불일치 방지) |
| `num_episodes` | 50 | — |
| `max_steps` | 1000 | — |
| `render` | false | true면 화면 표시 |
| `save_gif` | null | 지정하면 첫 에피소드를 GIF로 저장 |

전체 옵션(async_infer, stage_shift 강건성 체크 등)은 `configs/eval.yaml` 참고.

#### collect_square_rollouts.py
학습된 policy로 성공+실패가 섞인 롤아웃을 수집해 hdf5로 저장 — `train_dstg_failaware.py`의 입력 데이터를 만드는 용도. argparse 기반(hydra 아님).

```bash
python -m square_assembly.scripts.collect_square_rollouts --episodes 120 --out data/square_rollouts_v1.hdf5
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--base-ckpt` | (경로는 학습 서버 기준 — 실제 체크포인트로 교체) | 롤아웃에 쓸 policy |
| `--episodes` | 120 | — |
| `--max-steps` | 500 | — |
| `--out` | `data/square_rollouts_v1.hdf5` | — |

#### record_si_video.py
SI(자기개선) 파인튜닝된 policy의 롤아웃을 mp4로 녹화 — 베이스 policy와 비교용. argparse 기반.

```bash
python -m square_assembly.scripts.record_si_video --si-ckpt outputs/si/square_si/predictor_best.pt --episodes 5
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--si-ckpt` | (필수) | SI 파인튜닝 체크포인트 |
| `--base-ckpt` | (학습 서버 기준 경로 — 실제 체크포인트로 교체) | task/policy 설정 원본(run_config.yaml 출처) |
| `--episodes` | 3 | — |
| `--hires` | 480 | 0이면 정책의 84x84 실관측 그대로 사용, N>0이면 NxN 별도 렌더 |
| `--out` | `outputs/si_video.mp4` | — |

#### record_si_stg_video.py
`record_si_video.py`와 동일하지만 화면 옆에 STG(steps-to-go) 분포 패널을 같이 그린다 — grasp_carry의 `record_carry_stg_dist.py`와 동일 구성. argparse 기반.

```bash
python -m square_assembly.scripts.record_si_stg_video --si-ckpt outputs/si/square_si/predictor_best.pt \
    --dstg-ckpt outputs/dstg/square_failaware/predictor.pt --episodes 2
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--si-ckpt` | (필수) | SI 파인튜닝 체크포인트 |
| `--dstg-ckpt` | `outputs/dstg/square_failaware/predictor.pt` | 분포 패널에 쓸 DstgPredictor |
| `--episodes` | 2 | — |
| `--fps` | 8 | — |
| `--out` | `outputs/si_stg_video.mp4` | — |

---

## 3. 테스트

```bash
# jax venv
pytest projects/grasp_carry/tests/

# torch venv
pytest projects/square_assembly/tests/
```

---

## 4. 하네스/워크플로우 유틸 (상세는 CLAUDE.md)

```bash
python scripts/execute.py <plan.md> [--model MODEL] [--checkpoint-every N] [--push]   # 플랜 task 자동 실행
python scripts/merge_to_main.py [feat-branch] [--push] [--yes]                        # main 병합 (직접 git merge 금지)
python scripts/scheduler.py {--time HH:MM | --in DURATION} [--resume ID | --cmd CMD] --prompt "..."  # 예약 실행
```
