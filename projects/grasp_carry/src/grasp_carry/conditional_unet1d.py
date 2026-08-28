"""ConditionalUnet1D — Diffusion Policy의 액션 시퀀스 노이즈 예측기 (JAX/haiku 이식).

# From: real-stanford/diffusion_policy
#   diffusion_policy/model/diffusion/conditional_unet1d.py
#   diffusion_policy/model/diffusion/conv1d_components.py
# (원본은 PyTorch. 로직만 옮기고 원본은 수정하지 않음.)

왜 필요한가: 액션 청크를 32차원 평탄 벡터로 뭉개 MLP에 넣으면 시간 구조를
전혀 못 쓴다. 공식 구현은 청크를 (H, act_dim) 시퀀스로 보고 시간축 1D 컨볼루션
UNet으로 처리하며, 관측·확산타임스텝은 FiLM(채널별 scale/bias)으로 각 블록에
주입한다. 실측에서 평탄 MLP 버전이 도달률 0.58에서 정체했고 공식
하이퍼파라미터를 붙이면 오히려 무너져(0.00), 아키텍처가 병목으로 지목됐다.

PyTorch는 (B, C, T) 채널우선, 여기서는 haiku 관례에 맞춰 (B, T, C)로 다룬다.
"""

from typing import Sequence, Optional

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk


def mish(x):
  return x * jnp.tanh(jax.nn.softplus(x))


def sinusoidal_pos_emb(t, dim):
  """From: diffusion_policy/model/diffusion/positional_embedding.py"""
  half = dim // 2
  emb = np.log(10000) / (half - 1)
  emb = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -emb)
  emb = t.astype(jnp.float32)[:, None] * emb[None, :]
  return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


def group_norm(x, n_groups=8, eps=1e-5, name=None):
  """채널 마지막(B,T,C) 기준 GroupNorm."""
  C = x.shape[-1]
  g = min(n_groups, C)
  while C % g != 0:
    g -= 1
  B, T = x.shape[0], x.shape[1]
  xr = x.reshape(B, T, g, C // g)
  mean = jnp.mean(xr, axis=(1, 3), keepdims=True)
  var = jnp.var(xr, axis=(1, 3), keepdims=True)
  xr = (xr - mean) * jax.lax.rsqrt(var + eps)
  x = xr.reshape(B, T, C)
  scale = hk.get_parameter(f'{name}_scale', [C], init=jnp.ones)
  offset = hk.get_parameter(f'{name}_offset', [C], init=jnp.zeros)
  return x * scale + offset


def conv1d_block(x, out_ch, kernel_size, n_groups, name):
  """Conv1d -> GroupNorm -> Mish (원본 Conv1dBlock)."""
  x = hk.Conv1D(out_ch, kernel_size, stride=1, padding='SAME',
                name=f'{name}_conv')(x)
  x = group_norm(x, n_groups, name=f'{name}_gn')
  return mish(x)


def cond_residual_block1d(x, cond, out_ch, kernel_size, n_groups,
                          cond_predict_scale, name):
  """원본 ConditionalResidualBlock1D. cond를 FiLM(scale/bias)으로 주입."""
  in_ch = x.shape[-1]
  out = conv1d_block(x, out_ch, kernel_size, n_groups, f'{name}_b0')
  cond_ch = out_ch * 2 if cond_predict_scale else out_ch
  embed = hk.Linear(cond_ch, name=f'{name}_cond')(mish(cond))   # (B, cond_ch)
  if cond_predict_scale:
    embed = embed.reshape(embed.shape[0], 2, out_ch)
    scale = embed[:, 0][:, None, :]      # (B,1,out_ch)
    bias = embed[:, 1][:, None, :]
    out = scale * out + bias
  else:
    out = out + embed[:, None, :]
  out = conv1d_block(out, out_ch, kernel_size, n_groups, f'{name}_b1')
  res = x if in_ch == out_ch else hk.Conv1D(out_ch, 1, name=f'{name}_res')(x)
  return out + res


def conditional_unet1d(sample, timestep, global_cond,
                       diffusion_step_embed_dim=256,
                       down_dims: Sequence[int] = (256, 512, 1024),
                       kernel_size=5, n_groups=8, cond_predict_scale=True):
  """sample: (B, T, act_dim), timestep: (B,), global_cond: (B, cond_dim)
  반환: (B, T, act_dim) 예측 노이즈."""
  input_dim = sample.shape[-1]
  dsed = diffusion_step_embed_dim

  # --- 확산 타임스텝 인코더
  te = sinusoidal_pos_emb(timestep, dsed)
  te = hk.Linear(dsed * 4, name='dse_l0')(te)
  te = mish(te)
  te = hk.Linear(dsed, name='dse_l1')(te)
  cond = te if global_cond is None else jnp.concatenate([te, global_cond], -1)

  all_dims = [input_dim] + list(down_dims)
  in_out = list(zip(all_dims[:-1], all_dims[1:]))

  # --- down
  x = sample
  skips = []
  for i, (_, d_out) in enumerate(in_out):
    is_last = i >= len(in_out) - 1
    x = cond_residual_block1d(x, cond, d_out, kernel_size, n_groups,
                              cond_predict_scale, f'down{i}_r0')
    x = cond_residual_block1d(x, cond, d_out, kernel_size, n_groups,
                              cond_predict_scale, f'down{i}_r1')
    skips.append(x)
    if not is_last:                       # Downsample1d: stride-2 conv
      x = hk.Conv1D(d_out, 3, stride=2, padding='SAME', name=f'down{i}_ds')(x)

  # --- mid
  mid = all_dims[-1]
  for j in range(2):
    x = cond_residual_block1d(x, cond, mid, kernel_size, n_groups,
                              cond_predict_scale, f'mid{j}')

  # --- up (skip 연결 후 채널 결합)
  for i, (d_in, d_out) in enumerate(reversed(in_out[1:])):
    is_last = i >= len(in_out) - 1
    x = jnp.concatenate([x, skips.pop()], axis=-1)
    x = cond_residual_block1d(x, cond, d_in, kernel_size, n_groups,
                              cond_predict_scale, f'up{i}_r0')
    x = cond_residual_block1d(x, cond, d_in, kernel_size, n_groups,
                              cond_predict_scale, f'up{i}_r1')
    if not is_last:                       # Upsample1d: ConvTranspose stride 2
      x = hk.Conv1DTranspose(d_in, 4, stride=2, padding='SAME',
                             name=f'up{i}_us')(x)

  # --- final
  x = conv1d_block(x, down_dims[0], kernel_size, n_groups, 'final_b')
  x = hk.Conv1D(input_dim, 1, name='final_conv')(x)
  return x
