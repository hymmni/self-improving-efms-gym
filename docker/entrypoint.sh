#!/bin/sh
set -e

# nvidia-smi is only present when the container was started with --gpus / a GPU
# device reservation. Without it, force JAX onto CPU so jax-cuda12-pjrt doesn't
# try to initialize a GPU that isn't there and crash on import.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    export JAX_PLATFORMS=cpu
fi

exec "$@"
