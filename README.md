# Self-Improving Gym

Ghasemipour et al. 2025, *Self-Improving Embodied Foundation Models*(SI-EFM)의 pointmass 데모 재현에서 출발해, 같은 방법론(정책이 스스로 예측한 steps-to-go 감소분을 보상으로 삼는 self-improvement)을 서로 다른 환경·스택으로 확장해가는 연구 모노레포입니다. 방법론 요약과 설계 결정 이력은 `docs/ARCHITECTURE.md`·`docs/ADR.md`가 단일 진처입니다 — 아래는 이 레포를 처음 열었을 때 환경을 세팅하고 각 연구 트랙을 실행해보기 위한 안내입니다.

## 1. 디렉토리 구조

```
projects/
  pointmass/        # 1차 목표 — SI-EFM 원본 재현 (JAX/Haiku, 노트북 그대로 복사)
  grasp_carry/       # 은닉 물성 기반 파지·운반 (JAX/Haiku, pymunk 물리)
  square_assembly/   # robomimic Diffusion Policy + DDPO self-improvement (PyTorch, 별도 conda env)
docs/                # ARCHITECTURE.md·ADR.md(구조/결정 이력), superpowers/(플랜·스펙)
experiments/         # 의미 있는 변화가 있을 때마다 남기는 실험 기록
scripts/             # 하네스 실행기 (execute.py, merge_to_main.py 등) — 상세는 CLAUDE.md
docker/, docker-compose*.yml, DOCKER.md   # 두 스택(JAX/PyTorch)을 하나로 통합한 개발 컨테이너
references/          # 원본 참조 코드 (읽기 전용)
```

각 `projects/<name>/README.md`가 그 트랙의 구조·실행법을 담고 있고, 위 트리 전체의 최신 스냅샷은 `docs/ARCHITECTURE.md` 3절을 참고하세요.

## 2. 환경 설정

이 레포엔 서로 호환되지 않는 두 스택이 공존합니다: `pointmass`·`grasp_carry`는 JAX/Haiku, `square_assembly`는 PyTorch/robosuite/mujoco. 아래 두 방법 중 하나를 선택하세요.

### 방법 A — Docker (권장, 두 스택 한 번에)

```bash
cp .env.example .env        # WANDB_API_KEY 등 채우기
docker compose build
docker compose run --rm dev bash
```

컨테이너 안에서 스택별 venv를 activate 합니다.

```bash
source /opt/venvs/jax/bin/activate     # pointmass, grasp_carry
source /opt/venvs/torch/bin/activate   # square_assembly
```

GPU 서버에서 실행, VS Code Dev Containers 연동, X11 GUI(텔레옵/뷰어) 설정 등 상세는 `DOCKER.md`를 참고하세요.

### 방법 B — 로컬 conda (스택별 개별 설치)

**JAX 스택** (`pointmass`, `grasp_carry`) — 레포 루트에서:
```bash
conda create -n self-improving-gym python=3.11 -y
conda activate self-improving-gym
pip install -r requirements.txt
```
GPU(CUDA) 환경을 전제로 `jax[cuda12]`가 설치됩니다. GPU가 없어도 두 트랙 모두 규모가 작아 CPU만으로 충분히 돌아갑니다(`jax[cpu]`로 교체). 버전 pin의 근거는 `experiments/2026-07-03_clean-repro.md` 참고.

**PyTorch 스택** (`square_assembly`) — 별도 env:
```bash
cd projects/square_assembly
conda create -n square_assembly python=3.10
conda activate square_assembly
pip install -r requirements.txt && pip install -e .
```
버전 고정 근거는 `projects/square_assembly/docs/plan.md`의 "M0 확정 결과" 참고.

실제 하드웨어 구성(GPU 유무 등)은 `docs/private/ENVIRONMENT.md`(git 미추적, 로컬 전용)에서 확인하세요.

## 3. 서브프로젝트 실행

| 프로젝트 | 방법론 / 스택 | 문서 |
|---|---|---|
| `projects/pointmass/` | SI-EFM 원본 재현 + 환경 교란(순간이동·외력·장애물) 관찰 도구, JAX/Haiku | [`projects/pointmass/README.md`](projects/pointmass/README.md) |
| `projects/grasp_carry/` | 은닉 물성(마찰·질량) 기반 파지·운반, JAX/Haiku + pymunk | [`projects/grasp_carry/README.md`](projects/grasp_carry/README.md) |
| `projects/square_assembly/` | robomimic 벤치마크 위 Diffusion Policy + DDPO self-improvement, PyTorch | [`projects/square_assembly/README.md`](projects/square_assembly/README.md) |

각 README가 실행 예시(학습/평가/시각화)와 코드 구성을 담고 있습니다. 예:

```bash
# pointmass — 노트북 하나로 전체 파이프라인(Stage 1 SFT → Stage 2 self-improvement)
jupyter lab projects/pointmass/pointmass_notebook.ipynb

# grasp_carry — 정책 학습
python -m grasp_carry.scripts.train.train_carry_actor --help

# square_assembly — Diffusion Policy 학습
python -m square_assembly.scripts.train task=square
```

## 4. 하네스 / 개발 워크플로우

이 레포는 Claude Code 기반 자가 교정 하네스(`scripts/execute.py`)로도 작업을 진행합니다. 세션 시작 체크리스트, 플랜 실행 방법, main 병합 규칙 등 개발 프로세스 전체는 `CLAUDE.md`가 단일 진처입니다.

## 5. 문서 지도

- `docs/ARCHITECTURE.md` — 디렉토리 구조·기술 스택 스냅샷
- `docs/ADR.md` — 결정-이유-트레이드오프 로그
- `docs/private/ENVIRONMENT.md` — 로컬 하드웨어 구성 (git 미추적)
- `experiments/` — 실험 기록 (`LOG_TEMPLATE.md` 참고)
- `docs/superpowers/` — 브레인스토밍 스펙·플랜 (하네스용)
