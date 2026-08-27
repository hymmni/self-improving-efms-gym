# 3D robomimic square 스택 (PyTorch + robosuite + mujoco). From: mani_sim/requirements.txt
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

# libgl/libegl/libosmesa: mujoco 오프스크린(EGL) + onscreen(mjviewer/GLFW) 렌더링에 필요.
# libx11 등: mjviewer 창을 DISPLAY로 띄울 때 필요 (X11 forwarding 시).
# cmake: robomimic 의존성 egl_probe가 소스에서 빌드될 때 필요.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential cmake \
        libgl1 libglx-mesa0 libegl1 libosmesa6 libglew2.2 libglfw3 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY mani_sim/requirements.txt ./mani_sim/requirements.txt
RUN pip install --no-cache-dir -r mani_sim/requirements.txt

# mani_sim 소스는 이미지에 굽지 않음 — 런타임에 bind-mount로 들어오고,
# PYTHONPATH가 그 경로를 잡아주므로 `import mani_sim`이 바로 됨.
# (무거운 설치 레이어 뒤에 둬서, 이 값만 바꿀 때 위 pip install 캐시가 안 깨지게 함)
ENV PYTHONPATH=/workspace/mani_sim/src

CMD ["bash"]
