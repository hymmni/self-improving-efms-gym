# Docker 개발 환경

이 리포지토리에는 서로 호환되지 않는 두 개의 의존성 스택이 있어서, Docker 이미지도 두 개로 나뉩니다. 이름은 지금 올라가 있는 태스크(2D/3D)가 아니라 스택 기준입니다 — 태스크는 바뀔 수 있어도 이 스택 경계(루트=JAX, `mani_sim/`=PyTorch, ADR-003/ADR-007)는 잘 안 바뀌기 때문입니다.

| 서비스 | 대상 코드 | 스택 |
|---|---|---|
| `jax` | 리포지토리 루트 (`train_carry_*.py`, `record_carry_*.py`, `src/grasp_carry/` 등) | JAX + Haiku + Optax |
| `torch` | `mani_sim/` | PyTorch + robosuite + mujoco |

두 이미지 모두 리포지토리 전체를 `/workspace`에 bind-mount 합니다. 즉 컨테이너 안에서 코드를 고치는 게 아니라, 평소처럼 호스트(VSCode 등)에서 코드를 고치면 컨테이너에 바로 반영됩니다. **코드를 고쳐도 이미지 재빌드는 필요 없습니다.** 의존성(`requirements.txt`)을 바꿨을 때만 재빌드하면 됩니다.

## 0-1. VS Code로 작업할 때 (Dev Containers)

VS Code에 [Dev Containers 확장](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)을 설치한 뒤:

1. `Cmd/Ctrl+Shift+P` → `Dev Containers: Reopen in Container`
2. `jax`와 `torch` 중 어느 스택으로 열지 선택 (`.devcontainer/jax/`, `.devcontainer/torch/`)
3. VS Code가 그 서비스 컨테이너를 백그라운드로 띄우고 그 안에 붙음 — 터미널, 디버거, 파일 탐색기 전부 컨테이너 안 기준으로 동작하지만 파일은 bind-mount라 호스트와 동일

다른 스택으로 바꾸고 싶으면 같은 명령으로 다시 `Reopen in Container` 하고 반대쪽을 선택하면 됩니다.

## 0. 최초 1회 준비

```bash
cp .env.example .env
# .env 파일 열어서 WANDB_API_KEY 채우기
```

## 1. 빌드

```bash
docker compose build       # 둘 다 빌드
docker compose build jax   # 하나만
```

## 2. 실행 (CPU — 지금 이 개인 PC)

이미지 2개 = 컨테이너 2개, 완전히 독립적으로 따로 뜹니다. 동시에 켤 필요는 없고, 필요한 쪽만 그때그때 들어가면 됩니다.

```bash
# JAX 스택 (2D 블록 이송 등 루트 코드) 진입
docker compose run --rm jax bash

# PyTorch 스택 (mani_sim) 진입
docker compose run --rm torch bash
```

컨테이너 안 셸에 들어가면 평소처럼 `python train_carry_actor.py ...` 같은 명령을 그대로 실행하면 됩니다.

## 3. GPU 있는 학습 서버에서 실행할 때

서버에 [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) + [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)이 설치되어 있어야 합니다 (드라이버는 서버에 이미 있을 가능성이 높고, 툴킷만 추가 설치하면 됨).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm torch bash
```

같은 이미지/코드가 그대로 GPU를 사용합니다. `jax`는 컨테이너 시작 시 GPU가 안 보이면 자동으로 `JAX_PLATFORMS=cpu`로 전환되고(entrypoint 스크립트), `torch`는 PyTorch가 `torch.cuda.is_available()`로 알아서 판단하므로 특별한 설정이 필요 없습니다.

## 4. GUI (텔레옵 수집, mjviewer) 쓰기

호스트(Linux)에서 컨테이너의 X11 접근을 한 번 허용해야 합니다:

```bash
xhost +local:docker
```

그 다음 평소처럼 `docker compose run` 하면 `DISPLAY`가 자동으로 전달됩니다 (`.env`의 `DISPLAY` 값 사용, 보통 호스트와 동일한 `:0` 등).

- `jax`: matplotlib TkAgg 백엔드 사용 (컨테이너에 `python3-tk` 설치됨)
- `torch`: mujoco `mjviewer` 온스크린 창 사용 시 컨테이너 환경변수 `MUJOCO_GL`을 비워야 함 (compose 기본값은 `MUJOCO_GL=egl`, 헤드리스 학습/평가용). 온스크린이 필요하면:
  ```bash
  docker compose run --rm -e MUJOCO_GL= torch bash
  ```

## 5. 알려진 사소한 경고

- GPU 없는 환경에서 mujoco EGL 오프스크린 렌더러를 쓰면 프로그램 종료 시 `OpenGL.raw.EGL._errors.EGLError`가 찍힐 수 있음 — 렌더링 자체는 정상 동작하고, 소프트웨어 EGL 컨텍스트를 정리(`__del__`)하는 과정에서만 나는 무해한 경고. 실제 GPU 서버에서는 보통 안 뜸.

## 6. 새 패키지가 필요할 때

컨테이너 안에서 바로 `pip install <패키지>`를 해도 **당장은** 잘 동작합니다. 하지만 그건 그 컨테이너가 살아있는 동안만 유효합니다 — `--rm`으로 뜬 컨테이너는 나가는 순간 사라지고, `--rm` 없이 재사용해도 언젠가 컨테이너를 지우거나(`down`), 이미지를 재빌드하거나, GPU 서버 등 다른 머신에서 새로 빌드하면 그 설치는 흔적도 없이 사라집니다.

그래서 규칙:
1. 계속 쓸 패키지라고 판단되면 → 해당 스택의 `requirements.txt`(`jax`는 루트 `requirements.txt`, `torch`는 `mani_sim/requirements.txt`)에 **버전을 명시해서** 추가
2. `docker compose build <서비스명>`으로 재빌드해서 이미지에 영구 반영

이 순서를 안 지키면, 지금 `pymunk`/`cmake`가 빠져있던 것과 같은 "숨은 의존성" 문제가 또 생깁니다.

## 7. 자주 쓰는 팁

- 컨테이너가 이미 떠 있는 상태에서 셸을 하나 더 열고 싶으면: `docker compose exec jax bash`
- 의존성(`requirements.txt`) 수정 후에는: `docker compose build <서비스명>`
- 완전히 새로 빌드하고 싶으면: `docker compose build --no-cache <서비스명>`
