# 레포 소스 정리: grasp_carry / pointmass 분리, mani_sim → square_assembly 리네임

## 배경

이 레포는 원래 SI-EFM 논문의 pointmass 재현이 출발점이었다. 태스크가 단순해 다른 태스크로
연구를 옮겼고(GraspCarry2D, robomimic square), pointmass는 archive/로 옮겨져 있었다. 하지만
pointmass는 "무가치해서 버린" 게 아니라 다시 들여다볼 수 있는 연구 트랙이라 archive보다는
독립 프로젝트 폴더로 두는 게 낫다는 결론에 도달했다.

동시에 루트에는 GraspCarry2D 관련 스크립트 25개와 `src/`의 라이브러리 코드가 뒤섞여 있고,
`pointmass_core.py`가 아직 안 지워진 채 남아 있다. `mani_sim/`은 이미 자기완결형이지만
폴더명이 스택 기반이 아니라 프로젝트명(원본 레포명 유래)이라 실제 수행 태스크(robomimic
square 조립)를 반영하지 못한다.

이 문서는 Docker 인프라(이미 별도로 구축 완료, 건드리지 않음)를 제외한 소스 코드 재배치
설계다.

## 목표

1. GraspCarry2D 관련 코드(루트 스크립트 25개 + `src/` 라이브러리)를 `mani_sim/`과 동일한
   자기완결형 구조의 `grasp_carry/`로 통합한다.
2. `pointmass_core.py`를 위한 최소 골격 `pointmass/`를 만들어 향후 부활에 대비한다(archive
   전체 이관은 이번 범위 밖).
3. `mani_sim/`을 태스크 이름 `square_assembly/`로 리네임하고 내부 패키지 네임스페이스·도커
   경로·문서 경로 참조를 맞춘다.
4. 조사 중 발견한 `mani_sim_external/`(사실은 vendoring 아닌 자체 작성 코드, git 미추적)을
   정상적인 내부 모듈로 흡수해 추적 대상으로 만든다.
5. 전체 import 경로 개수정 후 pytest로 검증한다.

## 비목표 (이번 세션에서 안 함)

- `COMMANDS.md` 재작성 (현재도 이미 archive된 pointmass 스크립트만 문서화한 stale 상태 —
  경로 치환이 아니라 내용 재작성이 필요해 후속 세션으로 미룸)
- `docs/ARCHITECTURE.md` / `docs/ADR.md`의 **내용**(1차 목표 서사, pointmass 재현 목표 등)
  갱신 — 단, 리네임으로 인해 깨지는 **경로 문자열**(`mani_sim/` 언급 등)은 이번에 고친다
  (아래 "문서" 절 참고). 이건 재작성이 아니라 기계적 치환이라 별개다.
- `archive/`의 진단/실험 스크립트(수십 개)를 pointmass-only vs grasp_carry-공유로 분류하고
  공용 폴더를 설계하는 작업 — 별도 세션
- `requirements.txt` 의존성 정리 — 애초에 "삭제되는 코드"가 없어졌으므로(pointmass_core.py도
  이동만 함) 뺄 의존성이 없어 해당 사항 없음

## A. `grasp_carry/` — GraspCarry2D 통합

### 목표 구조

```
grasp_carry/
  src/grasp_carry/
    __init__.py
    config.py, env.py, gripper.py, policy.py        # src/grasp_carry/*에서 이동
    ddpo.py, diffusion_act.py, conditional_unet1d.py,
    eval_carry.py, reward.py, carry_stg_reward.py,
    train_carry_dstg.py, train_carry_predictor.py,
    train_carry_qstg.py                              # 루트 src/*.py (flat)에서 이동
    networks.py                                       # 신규 — pointmass_core.py의
                                                        #   build_continuous_act_discrete_dist_v0
                                                        #   (TIMER 네트워크) 포팅
    scripts/                                           # 루트의 진입점 스크립트 25개, 기능별
                                                        #   하위 폴더로 분류해 이동 (mani_sim이
                                                        #   policies/losses/runners/datasets로
                                                        #   나누는 것과 같은 원리를 scripts/에도
                                                        #   적용 — 25개라 mani_sim/scripts/처럼
                                                        #   평평하게 두기엔 많음)
      train/            # train_carry_actor.py, train_carry_actor_reinforce.py,
                         #   train_carry_si.py, finetune_carry_diffusion.py
      collect/           # collect_carry_demos.py, collect_carry_bc_rollouts.py,
                         #   collect_carry_teleop_detour.py  (데이터셋 → pkl 생성)
      record/            # record_carry.py, record_carry_actor.py,
                         #   record_carry_bc_stg_dist.py, record_carry_si.py,
                         #   record_carry_si_video.py, record_carry_stg_dist.py  (mp4 영상 생성,
                         #   moviepy/mediapy 사용 확인됨)
      analyze/           # calibrate_carry.py, compare_carry_selectors.py, eval_carry_actor.py,
                         #   eval_carry_si.py, probe_carry_qstg.py, rollout_carry_diff_stats.py,
                         #   run_bc_stg_guided.py, verify_carry_qstg_condb.py,
                         #   analyze_mu_jump_bimodal.py, analyze_mu_sigma_highrisk.py,
                         #   evaluate_stg_deadline.py, evaluate_stg_deadline_cdf.py
                         #   (체크포인트를 읽어 평가·비교·진단·plot — 나머지 전부)
                         # mani_sim/scripts/처럼 __init__.py 없이 네임스페이스 패키지로 둔다
  tests/                # 루트 tests/*.py 전부 이동 (test_policy.py, test_gripper.py,
                         #   test_env.py, test_reward.py, test_render.py, test_config.py,
                         #   test_ddpo.py, test_dstg.py, test_si_loop.py, test_stg_reward.py)
  data/                 # 루트에서 이동
  checkpoints/           # 루트에서 이동
  results/               # 루트에서 이동
  outputs/                # 루트에서 이동
  pyproject.toml          # 신규 — mani_sim/pyproject.toml 패턴(packages.find where=["src"]),
                            #   의존성은 최소화(버전 고정은 루트 requirements.txt가 단일 출처)
  README.md               # 신규, 간단히 (mani_sim/README.md 정도 분량)
```

이동 후 루트의 `src/`, `tests/` 디렉토리는 삭제한다.

### Import 경로 규칙

| 기존 | 신규 |
|---|---|
| `from src.grasp_carry.config import CarryConfig` | `from grasp_carry.config import CarryConfig` |
| `from src.ddpo import build_ddpo` (flat 모듈) | `from grasp_carry.ddpo import build_ddpo` |
| `from src.train_carry_predictor import concat_obs` | `from grasp_carry.train_carry_predictor import concat_obs` |
| `from pointmass_core import build_continuous_act_discrete_dist_v0` | `from grasp_carry.networks import build_continuous_act_discrete_dist_v0` |
| 스크립트 간 상호참조 (예: `eval_carry_actor.py` → `train_carry_actor.py`) | `from grasp_carry.scripts.train.train_carry_actor import ...` |

실행은 `python -m grasp_carry.scripts.train.train_carry_actor ...` 처럼 하위 폴더까지 포함한
모듈 경로로 바뀐다. editable install 없이 `PYTHONPATH=grasp_carry/src`만으로 import되게 한다
(아래 Docker 절 참고 — mani_sim/torch.Dockerfile과 동일 패턴).

### Docker / 문서

- `docker/jax.Dockerfile`: `ENV PYTHONPATH=/workspace/grasp_carry/src` 한 줄 추가
- `docker-compose.yml`: 변경 불필요 (`working_dir: /workspace`가 이미 범용)
- `DOCKER.md`: jax 서비스 대상 코드 설명(`train_carry_*.py, record_carry_*.py, src/grasp_carry/`
  → `grasp_carry/`)과 예시 명령(`python train_carry_actor.py` →
  `python -m grasp_carry.scripts.train.train_carry_actor`) 갱신

## B. `pointmass/` — 최소 골격

```
pointmass/
  pointmass_core.py         # 루트에서 이동, 내용 그대로 (networks.py로 포팅된 함수도
                              #   원본에는 그대로 남겨둔다 — 원본 노트북 재현 코드는 손대지
                              #   않는다는 기존 원칙 유지)
  pointmass_notebook.ipynb   # archive/에서 이동, 원본 그대로
```

`archive/`의 나머지 pointmass 진단/실험 스크립트(수십 개, `archive/*.py`, `archive/src/*`)는
이번에 건드리지 않는다. pointmass-only vs grasp_carry-공유 분류와 공용 폴더 설계는 후속
세션 과제다.

## C. `mani_sim/` → `square_assembly/` 리네임

robomimic의 "square"(사각 너트를 펙에 끼우는 조립) 태스크명을 그대로 사용한다.

### 변경 대상

- 폴더 전체: `mani_sim/` → `square_assembly/`
- 내부 패키지: `mani_sim/src/mani_sim/` → `square_assembly/src/square_assembly/`. 그 안의
  모든 `from mani_sim.xxx import ...` / `import mani_sim...`를 `square_assembly.xxx`로
  개수정 (policies/, losses/, runners/, configs/, utils/, datasets/, networks/, envs/,
  scripts/ 등 하위 30여 개 파일 + `tests/*.py`)
- `pyproject.toml`: `name = "mani_sim"` → `name = "square_assembly"`
- `environment.yml`: `name: mani_sim` → `name: square_assembly` (로컬에 이 이름의 실제 conda
  env는 존재하지 않음 — torch 스택은 Docker로만 구동되는 것으로 확인됨. 파일 수정만으로
  충분, `conda rename` 등 별도 조치 불필요)
- Docker: `docker-compose.yml`의 `working_dir: /workspace/mani_sim` → `/workspace/square_assembly`;
  `docker/torch.Dockerfile`의 `COPY mani_sim/requirements.txt` 및
  `ENV PYTHONPATH=/workspace/mani_sim/src` → `square_assembly` 경로로; `.devcontainer/torch/devcontainer.json`의
  `workspaceFolder`(`/workspace/mani_sim` → `/workspace/square_assembly`)와
  `"name": "torch (mani_sim)"` → `"torch (square_assembly)"`. 도커 **서비스 이름**(`torch`)
  자체는 스택 기준 명명 원칙(ARCHITECTURE.md 기존 방침)에 따라 바꾸지 않는다.
- 문서 경로 문자열만 치환 (내용 재작성 아님):
  - `DOCKER.md`의 torch 서비스 대상 코드 설명 (`mani_sim/` → `square_assembly/`)
  - `docs/ARCHITECTURE.md` 디렉토리 트리 절의 `mani_sim/` 3줄
  - `docs/ADR.md` ADR-007 본문의 `mani_sim/`, `mani_sim/environment.yml`,
    `mani_sim/src/mani_sim/factory.py` 등 경로 언급
  - `grasp_carry/src/grasp_carry/train_carry_si.py`, `train_carry_predictor.py`의 주석에
    있는 `mani_sim/train_si.py`, `mani_sim/3D 태스크` 등 비교 언급도 `square_assembly`로

## D. `mani_sim_external/` 해체

조사 결과 `mani_sim_external/piper_capstone/replay_buffer.py`는 실제로 외부에서 가져온
vendoring 코드가 아니다 — 원본 레포(`Leejw221/manipulation_simulator`)가 참조하던 FLARE
프로젝트의 `ReplayBuffer`는 원본 레포 자신의 로컬 전용 gitignore 대상이라 애초에 복사해올
실체가 없었고, 이 프로젝트에 실제로 필요한 좁은 인터페이스(`.episode_ends`, `.data[key]`
읽기 전용)만 보고 새로 작성된 코드다. 게다가 이 폴더 전체가 git에 추적되지 않고 있어(로컬
전용) — `zarr_dataset.py`가 이 파일 없이는 동작하지 않는데도 커밋 이력이 없다는 실질적
위험이 있다.

"외부 코드"라는 전제가 틀렸으므로 `_external` 폴더 개념 자체를 없앤다:

- `mani_sim/mani_sim_external/piper_capstone/replay_buffer.py` →
  `square_assembly/src/square_assembly/datasets/replay_buffer.py` (일반 내부 모듈로 흡수)
- `zarr_dataset.py`, `normalization.py`, `task_utils.py` 3곳의
  `sys.path.insert(..., parents[3] / "mani_sim_external" / "piper_capstone")` 해킹을 제거하고
  `from square_assembly.datasets.replay_buffer import ReplayBuffer` 형태의 평범한 절대
  import로 교체
- `square_assembly/.gitignore`(구 `mani_sim/.gitignore`)의 `mani_sim_external/` 제외 규칙과
  관련 주석을 삭제 — 이 코드는 이제부터 다른 소스코드와 동일하게 git 추적 대상이다
- `mani_sim_external/`이라는 이름 자체가 사라지므로 별도 리네임 규칙은 불필요

## 검증

- import 개수정 후 `pytest grasp_carry/tests/`, `pytest square_assembly/tests/` 둘 다 통과
- `python -m grasp_carry.scripts.train.train_carry_actor --help` 등 대표 스크립트 몇 개 직접 실행해
  깨지지 않는지 확인
- `square_assembly/src/square_assembly/scripts/*.py` 쪽도 가능한 범위에서 동일하게 확인
  (torch 스택이라 로컬에 CUDA 환경이 없으면 import까지만 확인 가능할 수 있음 — 이 경우
  한계를 명시적으로 보고)
- `git status`로 gitignore 규칙 변경 후 `square_assembly/src/square_assembly/datasets/replay_buffer.py`가
  실제로 `git add` 대상에 잡히는지 확인
