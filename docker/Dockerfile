# jax(2D grasp-carry) + torch(square_assembly) 스택 통합 이미지.
# 서로 다른 pip 스택(CUDA 버전 포함)이라 이미지 안에서는 venv 2개로 분리한다.
# From: /requirements.txt, projects/square_assembly/requirements.txt
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

# python3-tk는 apt로 설치해야 시스템 python3(3.11)와 ABI가 맞음 — venv도 이 인터프리터를
# 그대로 물려받으므로(python3 -m venv) tk가 두 venv 모두에서 문제없이 잡힌다.
# python3-dev: robosuite의 evdev(디바이스 입력) 등 C 확장을 소스에서 빌드할 때 Python.h가 필요.
# (python:3.10-slim 공식 이미지는 헤더를 이미 포함하지만, apt python3는 별도 설치해야 함)
# cmake: robomimic 의존성 egl_probe가 소스에서 빌드될 때 필요.
# libgl/libegl/libosmesa: mujoco 오프스크린(EGL) + onscreen(mjviewer/GLFW) 렌더링에 필요.
# libx11 등: mjviewer/matplotlib TkAgg 창을 DISPLAY로 띄울 때 필요 (X11 forwarding 시).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-tk python3-dev \
        git ffmpeg build-essential cmake \
        libgl1 libglx-mesa0 libegl1 libosmesa6 libglew2.2 libglfw3 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

# jax venv — 별도 COPY+RUN으로 둬서 projects/square_assembly/requirements.txt만 바뀔 때 이 레이어가 안 깨지게 함.
COPY requirements.txt ./requirements.txt
RUN python3 -m venv /opt/venvs/jax \
    && /opt/venvs/jax/bin/pip install --no-cache-dir -r requirements.txt

# torch venv
COPY projects/square_assembly/requirements.txt ./projects/square_assembly/requirements.txt
RUN python3 -m venv /opt/venvs/torch \
    && /opt/venvs/torch/bin/pip install --no-cache-dir -r projects/square_assembly/requirements.txt

# 컨테이너가 root가 아니라 host UID로 도므로(docker-compose.yml user:), 그 UID가 여기에
# 계속 pip install 할 수 있도록 world-writable로 열어둠 — 단일 개발자용 로컬 이미지라 허용.
RUN chmod -R o+w /opt/venvs

# grasp_carry/square_assembly 소스는 이미지에 굽지 않음 — 런타임에 bind-mount로 들어오고,
# PYTHONPATH가 그 경로들을 잡아주므로 jax venv에서 `import grasp_carry`가, torch venv에서
# `import square_assembly`이 바로 됨. venv가 서로 다른 site-packages를 쓰므로 이렇게 경로를 합쳐
# 놔도 서로의 의존성을 혼동하지 않는다.
# (무거운 설치 레이어 뒤에 둬서, 이 값만 바꿀 때 위 pip install 캐시가 안 깨지게 함)
ENV PYTHONPATH=/workspace/projects/grasp_carry/src:/workspace/projects/square_assembly/src

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
