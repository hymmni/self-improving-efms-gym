# Docker 개발 환경

이 리포지토리에는 서로 호환되지 않는 두 개의 pip 의존성 스택(JAX/Haiku/Optax 루트 코드, PyTorch/robosuite/mujoco `square_assembly/`)이 있습니다. Docker 이미지/컨테이너는 **하나**로 통합하고, 그 안에서 venv 2개로 스택을 나눕니다.

| venv | 대상 코드 | 스택 |
|---|---|---|
| `/opt/venvs/jax` | `grasp_carry/` (`src/grasp_carry/scripts/`, `src/grasp_carry/` 등) | JAX + Haiku + Optax |
| `/opt/venvs/torch` | `square_assembly/` | PyTorch + robosuite + mujoco |

컨테이너는 리포지토리 전체를 `/workspace`에 bind-mount 합니다. 즉 컨테이너 안에서 코드를 고치는 게 아니라, 평소처럼 호스트(VSCode 등)에서 코드를 고치면 컨테이너에 바로 반영됩니다. **코드를 고쳐도 이미지 재빌드는 필요 없습니다.** 의존성(`requirements.txt`)을 바꿨을 때만 재빌드하면 됩니다.

## 0-1. VS Code로 작업할 때 (Dev Containers)

VS Code에 [Dev Containers 확장](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)을 설치한 뒤:

1. `Cmd/Ctrl+Shift+P` → `Dev Containers: Reopen in Container` (컨테이너 하나만 뜨므로 선택 메뉴 없음)
2. 컨테이너 안에서 `Cmd/Ctrl+Shift+P` → `File: Open Workspace from File...` → `self-improving-gym.code-workspace` 선택 (최초 1회만 하면 이후 그대로 유지됨)
3. 이제 창 하나에 폴더 루트(jax) + `square_assembly`(torch)가 각각 별도 루트로 열리고, 폴더별로 다른 Python 인터프리터(`.vscode/settings.json`, `square_assembly/.vscode/settings.json`)가 자동 적용됩니다 — 어느 쪽 코드를 열어도 인텔리센스/디버거가 맞는 venv를 씀
4. 터미널은 각각 열어서 `source /opt/venvs/jax/bin/activate` / `source /opt/venvs/torch/bin/activate` 해두면 바로 실행 가능

컨테이너/창 재시작이 필요 없으니, 두 스택을 오가며 작업해도 VS Code를 하나만 켜두면 됩니다.

## 0. 최초 1회 준비

```bash
cp .env.example .env
# .env 파일 열어서 WANDB_API_KEY 채우기
```

## 1. 빌드

```bash
docker compose build
```

## 2. 실행 (CPU — 지금 이 개인 PC)

```bash
docker compose run --rm dev bash
```

들어간 뒤 스택에 맞는 venv를 activate 합니다.

```bash
source /opt/venvs/jax/bin/activate     # 2D 블록 이송 등 루트 코드
source /opt/venvs/torch/bin/activate   # square_assembly
```

jax venv 기준 실행 예시: `python -m grasp_carry.scripts.train.train_carry_actor ...`

## 3. GPU 있는 학습 서버에서 실행할 때

서버에 [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) + [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)이 설치되어 있어야 합니다 (드라이버는 서버에 이미 있을 가능성이 높고, 툴킷만 추가 설치하면 됨).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev bash
```

같은 이미지/코드가 그대로 GPU를 사용합니다. jax venv 쪽은 컨테이너 시작 시 GPU가 안 보이면 자동으로 `JAX_PLATFORMS=cpu`로 전환되고(entrypoint 스크립트), torch venv 쪽은 PyTorch가 `torch.cuda.is_available()`로 알아서 판단하므로 특별한 설정이 필요 없습니다.

## 4. GUI (텔레옵 수집, mjviewer) 쓰기

호스트(Linux)에서 컨테이너의 X11 접근을 한 번 허용해야 합니다:

```bash
xhost +local:docker
```

그 다음 평소처럼 `docker compose run` 하면 `DISPLAY`가 자동으로 전달됩니다 (`.env`의 `DISPLAY` 값 사용, 보통 호스트와 동일한 `:0` 등).

- jax venv: matplotlib TkAgg 백엔드 사용 (이미지에 `python3-tk` 설치됨)
- torch venv: mujoco `mjviewer` 온스크린 창 사용 시 컨테이너 환경변수 `MUJOCO_GL`을 비워야 함 (compose 기본값은 `MUJOCO_GL=egl`, 헤드리스 학습/평가용). 온스크린이 필요하면:
  ```bash
  docker compose run --rm -e MUJOCO_GL= dev bash
  ```

## 5. 알려진 사소한 경고

- GPU 없는 환경에서 mujoco EGL 오프스크린 렌더러를 쓰면 프로그램 종료 시 `OpenGL.raw.EGL._errors.EGLError`가 찍힐 수 있음 — 렌더링 자체는 정상 동작하고, 소프트웨어 EGL 컨텍스트를 정리(`__del__`)하는 과정에서만 나는 무해한 경고. 실제 GPU 서버에서는 보통 안 뜸.

## 6. 새 패키지가 필요할 때

컨테이너 안에서 바로 `pip install <패키지>`를 해도 **당장은** 잘 동작합니다. 하지만 그건 그 컨테이너가 살아있는 동안만 유효합니다 — `--rm`으로 뜬 컨테이너는 나가는 순간 사라지고, `--rm` 없이 재사용해도 언젠가 컨테이너를 지우거나(`down`), 이미지를 재빌드하거나, GPU 서버 등 다른 머신에서 새로 빌드하면 그 설치는 흔적도 없이 사라집니다.

그래서 규칙:
1. 계속 쓸 패키지라고 판단되면 → 해당 스택의 `requirements.txt`(jax는 루트 `requirements.txt`, torch는 `square_assembly/requirements.txt`)에 **버전을 명시해서** 추가
2. `docker compose build`로 재빌드해서 이미지에 영구 반영

이 순서를 안 지키면, 지금 `pymunk`/`cmake`가 빠져있던 것과 같은 "숨은 의존성" 문제가 또 생깁니다.

## 7. 자주 쓰는 팁

- 컨테이너가 이미 떠 있는 상태에서 셸을 하나 더 열고 싶으면: `docker compose exec dev bash`
- 의존성(`requirements.txt`) 수정 후에는: `docker compose build`
- 완전히 새로 빌드하고 싶으면: `docker compose build --no-cache`
