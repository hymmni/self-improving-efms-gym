"""DDPO(Black et al. 2023) 핵심 부품: 디퓨전 액션 정책의 단계별 로그확률.

왜 필요한가: 논문(SI-EFM) Stage-2는 REINFORCE로 정책을 업데이트하는데, 그러려면
`log p(a|o)`가 필요하다. 이산 토큰 정책은 소프트맥스 로그확률로 바로 나오지만,
`src/diffusion_act.py`의 연속 액션 디퓨전 정책은 역확산 100단계를 적분해야 닫힌
형태의 `log p(a|o)`가 나온다 — 계산이 불가능하다.

DDPO의 우회: 역확산의 각 단계(kk=1..n_steps-1)는 파라미터에 의존하는 평균
`mean(params, x, obs, kk)`과 파라미터 독립 상수 표준편차 `sigma(kk)=sqrt(betas[kk])`를
갖는 등방 가우시안 전이다. "체인 전체"가 아니라 "단계 하나하나"를 정책 그래디언트의
대상으로 삼으면 각 단계의 로그확률은 정확히 계산된다. 체인 로그확률은 단계 로그확률의
합이다. 마지막 단계(kk=0)는 노이즈를 더하지 않는 결정론적 변환이라 확률밀도가 디랙
델타이므로 합에서 제외한다.

이 모듈은 `nets`가 `clip_sample=True`로 빌드되었다고 가정한다(대상 체크포인트
`grasp_carry_diff100`이 그 경로를 쓰기 때문). `src/diffusion_act.py`는 수정하지
않는다 — PushT/파지·운반 학습 코드가 공유하는 모듈이라 재현성이 걸려 있다.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.diffusion_act import DiffusionActNets


class DDPOFns(NamedTuple):
  sample_with_trace: object   # (params, obs, key) -> (x_final, xs)
  step_logp: object           # (params, obs, x_in, x_out, kk) -> (B,)
  chain_logp: object          # (params, obs, xs) -> (B,)


def _posterior_mean(nets: DiffusionActNets, params, obs, x, kk_scalar_or_vec, eps):
  """`sample_chunk`의 body 내부와 동일한 clip_sample=True 경로.

  From: src/diffusion_act.py build_diffusion_act_chunk().sample_chunk.body
  (읽기 전용 참조 — 로직만 복제, 원본은 수정하지 않음)

  `kk_scalar_or_vec`이 0-d 스칼라(sample_with_trace, 배치 전체가 같은 kk)일 수도,
  (B,) 벡터(step_logp, env-step마다 다른 kk)일 수도 있다 — 벡터인 경우에만
  `x`(B,D)에 맞춰 뒤에 축을 하나 붙인다.
  """
  sched = nets.schedule
  a_t = sched.alphas[kk_scalar_or_vec]
  ac_t = sched.alphas_cumprod[kk_scalar_or_vec]
  ac_prev = jnp.where(kk_scalar_or_vec > 0,
                       sched.alphas_cumprod[jnp.maximum(kk_scalar_or_vec - 1, 0)],
                       jnp.ones_like(ac_t))
  betas_kk = sched.betas[kk_scalar_or_vec]
  if jnp.ndim(kk_scalar_or_vec) > 0:
    a_t, ac_t, ac_prev, betas_kk = (v[:, None] for v in (a_t, ac_t, ac_prev, betas_kk))
  x0 = (x - jnp.sqrt(1.0 - ac_t) * eps) / jnp.sqrt(ac_t)
  x0 = jnp.clip(x0, -1.0, 1.0)
  c0 = betas_kk * jnp.sqrt(ac_prev) / (1.0 - ac_t)
  cx = (1.0 - ac_prev) * jnp.sqrt(a_t) / (1.0 - ac_t)
  return c0 * x0 + cx * x


def build_ddpo(nets: DiffusionActNets, chunk_flat_dim: int) -> DDPOFns:
  """`build_diffusion_act_chunk()`가 돌려준 `nets`를 받아 DDPO용 함수 3개를 만든다."""
  sched = nets.schedule
  n_steps = nets.n_steps

  def sample_with_trace(params, obs, key):
    """`nets.sample_chunk`와 동일한 액션을 내되, 중간 latent를 함께 기록한다.

    난수 소비 순서를 `sample_chunk`와 정확히 맞춰야 동일한 샘플이 나온다.
    """
    B = obs.shape[0]
    key, sub = jax.random.split(key)
    x0 = jax.random.normal(sub, (B, chunk_flat_dim))

    def body(carry, k):
      x, key = carry
      kk = n_steps - 1 - k                              # T-1 -> 0
      t = jnp.full((B,), kk, jnp.int32)
      eps, _ = nets.apply(params, obs, x, t)
      mean = _posterior_mean(nets, params, obs, x, kk, eps)
      key, sub = jax.random.split(key)
      noise = jax.random.normal(sub, x.shape)
      # 마지막 스텝(kk=0)에서는 노이즈를 더하지 않는다 (sample_chunk와 동일)
      x_next = mean + jnp.where(kk > 0, jnp.sqrt(sched.betas[kk]), 0.0) * noise
      return (x_next, key), x_next

    (x_final, _), ys = jax.lax.scan(body, (x0, key), jnp.arange(n_steps))
    # ys[k] = x_next after iteration k = latent right before iteration k+1.
    # xs[i] := latent before applying kk = n_steps-1-i  =>  xs[0]=x0, xs[i]=ys[i-1] for i>=1.
    xs = jnp.concatenate([x0[None], ys], axis=0)
    return x_final, xs

  def step_logp(params, obs, x_in, x_out, kk):
    """(env-step, 역확산-단계) 쌍의 미니배치에 대한 한 단계 로그확률. kk: (B,) int32, kk>=1."""
    t = kk.astype(jnp.int32)
    eps, _ = nets.apply(params, obs, x_in, t)
    mean = _posterior_mean(nets, params, obs, x_in, kk, eps)
    sigma = jnp.sqrt(sched.betas[kk])                    # (B,) — 파라미터 독립 상수
    D = x_out.shape[-1]
    sq = jnp.sum(((x_out - mean) / sigma[:, None]) ** 2, axis=-1)
    return -0.5 * sq - D * jnp.log(sigma) - 0.5 * D * jnp.log(2.0 * jnp.pi)

  def chain_logp(params, obs, xs):
    """한 env-step의 체인 전체 로그확률 합 (kk = n_steps-1 ... 1). xs: (n_steps+1, B, D)."""
    B = obs.shape[0]

    def scan_body(acc, i):
      kk = n_steps - 1 - i
      x_in = xs[i]
      x_out = xs[i + 1]
      kk_b = jnp.full((B,), kk, jnp.int32)
      return acc + step_logp(params, obs, x_in, x_out, kk_b), None

    total, _ = jax.lax.scan(scan_body, jnp.zeros((B,)), jnp.arange(n_steps - 1))
    return total

  return DDPOFns(sample_with_trace=sample_with_trace, step_logp=step_logp,
                 chain_logp=chain_logp)
