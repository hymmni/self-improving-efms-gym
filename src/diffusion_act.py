"""디퓨전 액션 헤드 + 액션 청킹 (phase 4, PushT용).

왜 필요한가 (실측 근거):
  - PushT 데모에는 같은 관측에서 미는 방향이 갈리는 상태가 34% 있다(이웃 액션
    방향 집중도 R<0.7).
  - 그런데 K=3 가우시안 믹스처는 그걸 못 잡았다: 가중치는 나눠 쓰지만(2등>0.2가
    27.7%) 성분들이 전부 같은 방향을 가리킨다(방향차 중앙 5.0도, 진짜 이봉 0.2%).
  - 게다가 한 스텝 액션 우도로 조기 종료하면 롤아웃 성능이 오히려 떨어졌다
    (도달률 0.42 → 0.08~0.20). 한 스텝 예측이 여러 데모를 평균내기 때문.

그래서 (1) 액션을 H스텝 청크로 한 번에 예측해 시퀀스 수준에서 커밋하게 하고,
(2) 분포를 DDPM으로 모델링해 다봉을 표현한다. Diffusion Policy가 PushT에서
쓴 구성과 같은 계열이다.

구조: 액션 헤드와 STG(거리) 헤드는 각자 독립 MLP(기존 코드와 동일하게 백본 미공유).
  eps_head : [noisy_chunk(H*A), obs_emb, t_emb] -> 예측 노이즈 (H*A)
  dist_head: obs -> 카테고리컬 logits (기존과 동일)
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from src.conditional_unet1d import conditional_unet1d


class DDPMSchedule(NamedTuple):
  betas: jnp.ndarray
  alphas: jnp.ndarray
  alphas_cumprod: jnp.ndarray


def make_ddpm_schedule(n_steps: int, beta_start=1e-4, beta_end=0.02,
                       schedule='squaredcos'):
  """From: real-stanford/diffusion_policy (config: beta_schedule=squaredcos_cap_v2)
  공식 구현이 쓰는 코사인 스케줄. linear보다 저노이즈 구간에 스텝을 더 배분한다."""
  if schedule == 'linear':
    betas = jnp.linspace(beta_start, beta_end, n_steps)
  else:
    # squaredcos_cap_v2 (Nichol & Dhariwal): alphas_cumprod를 코사인으로 정의
    t = jnp.linspace(0, n_steps, n_steps + 1) / n_steps
    ac = jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2
    ac = ac / ac[0]
    betas = jnp.clip(1.0 - ac[1:] / ac[:-1], 0.0, 0.999)
  alphas = 1.0 - betas
  return DDPMSchedule(betas, alphas, jnp.cumprod(alphas))


def _time_embedding(t, dim=64):
  """정현파 타임스텝 임베딩. t: (B,) 정수."""
  half = dim // 2
  freqs = jnp.exp(-np.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
  ang = t.astype(jnp.float32)[:, None] * freqs[None, :]
  return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class DiffusionActNets(NamedTuple):
  init: object            # rng -> params
  apply: object           # (params, obs, noisy, t) -> (eps_pred, dist_logits)
  dist_logits: object     # (params, obs) -> logits
  sample_chunk: object    # (params, obs, key) -> 액션 청크 (B, H*A)
  schedule: DDPMSchedule
  n_steps: int


def build_diffusion_act_chunk(layer_sizes, chunk_flat_dim, num_dist_bins,
                              obs_dim, n_diffusion_steps=100, t_emb_dim=64,
                              clip_sample=True, backbone='unet',
                              horizon=16, act_dim=2,
                              down_dims=(256, 512, 1024), kernel_size=5,
                              diffusion_step_embed_dim=256):
  """chunk_flat_dim = 청크길이 H * 액션차원 A (평탄화).
  backbone='unet'이면 공식 ConditionalUnet1D(시간축 1D conv + FiLM),
  'mlp'이면 기존 평탄 MLP."""
  sched = make_ddpm_schedule(n_diffusion_steps)

  def _net(obs, noisy, t):
    # --- 액션(노이즈 예측) 헤드
    if backbone == 'unet':
      seq = noisy.reshape(noisy.shape[0], horizon, act_dim)
      eps = conditional_unet1d(seq, t, obs,
                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                               down_dims=down_dims, kernel_size=kernel_size,
                               cond_predict_scale=True)
      eps = eps.reshape(noisy.shape[0], chunk_flat_dim)
    else:
      obs_emb = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                            activate_final=True)(obs)
      te = hk.Linear(layer_sizes[-1])(_time_embedding(t, t_emb_dim))
      h = jnp.concatenate([noisy, obs_emb, te], axis=-1)
      h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu, activate_final=True)(h)
      eps = hk.Linear(chunk_flat_dim,
                      w_init=hk.initializers.VarianceScaling(1e-2))(h)
    # --- STG(거리) 헤드: 관측만 사용, 독립 MLP
    hd = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                     activate_final=True)(obs)
    logits = hk.Linear(num_dist_bins, with_bias=False)(hd)
    return eps, logits

  tn = hk.without_apply_rng(hk.transform(_net))

  def init(rng):
    return tn.init(rng,
                   jnp.zeros((2, obs_dim), jnp.float32),
                   jnp.zeros((2, chunk_flat_dim), jnp.float32),
                   jnp.zeros((2,), jnp.int32))

  def dist_logits(params, obs):
    _, logits = tn.apply(params, obs,
                         jnp.zeros((obs.shape[0], chunk_flat_dim), jnp.float32),
                         jnp.zeros((obs.shape[0],), jnp.int32))
    return logits

  def sample_chunk(params, obs, key):
    """역확산으로 액션 청크를 샘플. obs: (B, obs_dim) 정규화된 관측."""
    B = obs.shape[0]
    key, sub = jax.random.split(key)
    x = jax.random.normal(sub, (B, chunk_flat_dim))

    def body(carry, k):
      x, key = carry
      kk = n_diffusion_steps - 1 - k                   # T-1 -> 0
      t = jnp.full((B,), kk, jnp.int32)
      eps, _ = tn.apply(params, obs, x, t)
      a_t = sched.alphas[kk]
      ac_t = sched.alphas_cumprod[kk]
      if clip_sample:
        # From: diffusers DDPMScheduler(clip_sample=True) — x0를 [-1,1]로 자르고
        # 그로부터 posterior 평균을 다시 구한다(발산 방지)
        x0 = (x - jnp.sqrt(1.0 - ac_t) * eps) / jnp.sqrt(ac_t)
        x0 = jnp.clip(x0, -1.0, 1.0)
        ac_prev = jnp.where(kk > 0, sched.alphas_cumprod[jnp.maximum(kk - 1, 0)],
                            jnp.ones_like(ac_t))
        c0 = sched.betas[kk] * jnp.sqrt(ac_prev) / (1.0 - ac_t)
        cx = (1.0 - ac_prev) * jnp.sqrt(a_t) / (1.0 - ac_t)
        mean = c0 * x0 + cx * x
      else:
        coef = sched.betas[kk] / jnp.sqrt(1.0 - ac_t)
        mean = (x - coef * eps) / jnp.sqrt(a_t)
      key, sub = jax.random.split(key)
      noise = jax.random.normal(sub, x.shape)
      # 마지막 스텝(kk=0)에서는 노이즈를 더하지 않는다
      x = mean + jnp.where(kk > 0, jnp.sqrt(sched.betas[kk]), 0.0) * noise
      return (x, key), None

    (x, _), _ = jax.lax.scan(body, (x, key), jnp.arange(n_diffusion_steps))
    return x

  return DiffusionActNets(init=init, apply=tn.apply, dist_logits=dist_logits,
                          sample_chunk=sample_chunk, schedule=sched,
                          n_steps=n_diffusion_steps)


def diffusion_loss(nets: DiffusionActNets, params, obs, chunk, key):
  """DDPM 학습 손실: 무작위 노이즈 레벨에서 노이즈 예측 MSE."""
  B = chunk.shape[0]
  key, k1, k2 = jax.random.split(key, 3)
  t = jax.random.randint(k1, (B,), 0, nets.n_steps)
  eps = jax.random.normal(k2, chunk.shape)
  ac = nets.schedule.alphas_cumprod[t][:, None]
  noisy = jnp.sqrt(ac) * chunk + jnp.sqrt(1.0 - ac) * eps
  eps_pred, _ = nets.apply(params, obs, noisy, t)
  return jnp.mean((eps_pred - eps) ** 2)
