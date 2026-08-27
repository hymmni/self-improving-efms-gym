# 2D GraspCarry 스택 (JAX/Haiku/Optax). From: /requirements.txt
# 이 스택은 mani_sim(PyTorch/CUDA12.1)과 CUDA 런타임 충돌 위험이 있어 별도 이미지로 분리.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# python3-tk는 apt로 설치해야 시스템 python3(3.11)와 ABI가 맞음.
# (python:3.11-slim 공식 이미지의 자체 빌드 파이썬에 apt tk를 얹으면 인터프리터가 달라 안 잡힘)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-tk \
        libx11-6 libxext6 \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY docker/jax-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
