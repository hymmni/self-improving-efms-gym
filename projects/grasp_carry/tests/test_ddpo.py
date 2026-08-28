"""Tests for `src/ddpo.py` (phase 4, step 0 — DDPO 로그확률 부품).

세 가지를 검증한다:
  1. `sample_with_trace`가 `nets.sample_chunk`와 동일한 액션을 내는가 (정책을
     바꾸지 않았다는 전제).
  2. `step_logp`가 실제 샘플링 분포(평균·표준편차)와 일치하는가.
  3. `chain_logp`의 그래디언트가 eps(액션) 헤드에는 흐르고 STG(거리) 헤드에는
     전혀 흐르지 않는가 (DDPO 업데이트가 보상 예측 헤드를 오염시키지 않는다는 보장).
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.diffusion_act import build_diffusion_act_chunk
from grasp_carry.ddpo import build_ddpo, _posterior_mean

OBS_DIM = 6
CHUNK_DIM = 4
NUM_BINS = 20
N_STEPS = 10
LAYER_SIZES = (32, 32)


def _build():
  nets = build_diffusion_act_chunk(LAYER_SIZES, CHUNK_DIM, NUM_BINS, OBS_DIM,
                                   n_diffusion_steps=N_STEPS, backbone='mlp')
  ddpo = build_ddpo(nets, CHUNK_DIM)
  return nets, ddpo


# ------------------------------------------------- 1. sample_chunk와의 동일성
def test_trace_matches_sample_chunk():
  nets, ddpo = _build()
  params = nets.init(jax.random.PRNGKey(0))
  obs = jax.random.normal(jax.random.PRNGKey(1), (3, OBS_DIM))

  for seed in range(5):
    key = jax.random.PRNGKey(100 + seed)
    x_direct = nets.sample_chunk(params, obs, key)
    x_trace, xs = ddpo.sample_with_trace(params, obs, key)

    np.testing.assert_allclose(np.array(x_direct), np.array(x_trace),
                               atol=1e-6, rtol=0)
    assert xs.shape == (N_STEPS + 1, 3, CHUNK_DIM)
    np.testing.assert_allclose(np.array(xs[-1]), np.array(x_trace),
                               atol=1e-6, rtol=0)


# ------------------------------------------------- 2. step_logp vs 실측 분포
def test_step_logp_matches_empirical_distribution():
  nets, ddpo = _build()
  params = nets.init(jax.random.PRNGKey(0))
  B = 2
  kk_val = 5  # 1 <= kk_val <= N_STEPS-1
  obs = jax.random.normal(jax.random.PRNGKey(1), (B, OBS_DIM))
  x_in = jax.random.normal(jax.random.PRNGKey(2), (B, CHUNK_DIM))
  kk_arr = jnp.full((B,), kk_val, jnp.int32)

  # 실제 샘플러(_posterior_mean, sample_with_trace가 쓰는 바로 그 함수)로 한 단계를
  # N번 굴려 진짜 샘플링 분포를 만든다.
  eps, _ = nets.apply(params, obs, x_in, kk_arr)
  mean_real = _posterior_mean(nets, params, obs, x_in, kk_arr, eps)
  sigma_real = jnp.sqrt(nets.schedule.betas[kk_val])

  N = 20000
  noise = jax.random.normal(jax.random.PRNGKey(3), (N, B, CHUNK_DIM))
  x_out_samples = mean_real[None] + sigma_real * noise
  emp_mean = np.asarray(x_out_samples).mean(axis=0)
  emp_std = np.asarray(x_out_samples).std(axis=0)

  # step_logp가 내부적으로 쓰는 mean을, step_logp 자체에 대한 그래디언트로
  # 역추출한다(블랙박스 검증 — _posterior_mean을 재사용하지 않음):
  #   logp(x_out) = -0.5*((x_out-mean)/sigma)^2 + const
  #   => d(logp)/d(x_out) = (mean - x_out) / sigma^2
  probe = jnp.zeros_like(x_in)
  grad_fn = jax.grad(lambda xo: ddpo.step_logp(params, obs, x_in, xo, kk_arr).sum())
  g = grad_fn(probe)
  sigma_expected = float(jnp.sqrt(nets.schedule.betas[kk_val]))
  mean_from_step_logp = probe + (sigma_expected ** 2) * g

  np.testing.assert_allclose(emp_mean, np.asarray(mean_from_step_logp),
                             rtol=0.05, atol=0.05)
  np.testing.assert_allclose(emp_std, sigma_expected, rtol=0.05, atol=0.05)


# ------------------------------------------------- 3. chain_logp == step_logp 합
def test_chain_logp_equals_sum_of_steps():
  nets, ddpo = _build()
  params = nets.init(jax.random.PRNGKey(0))
  B = 4
  obs = jax.random.normal(jax.random.PRNGKey(1), (B, OBS_DIM))
  _, xs = ddpo.sample_with_trace(params, obs, jax.random.PRNGKey(2))

  total = ddpo.chain_logp(params, obs, xs)

  manual = jnp.zeros((B,))
  for i in range(N_STEPS - 1):
    kk = N_STEPS - 1 - i
    kk_arr = jnp.full((B,), kk, jnp.int32)
    manual = manual + ddpo.step_logp(params, obs, xs[i], xs[i + 1], kk_arr)

  np.testing.assert_allclose(np.array(total), np.array(manual),
                             atol=1e-5, rtol=1e-5)


# ------------------------------------------------- 4. 그래디언트가 eps 헤드에만 흐름
def _is_dist_head_key(module_key: str) -> bool:
  """haiku 모듈 이름 판별 기준 (build_diffusion_act_chunk의 backbone='mlp' 경로 고정
  구조에서 유도): STG(dist) 헤드는 액션(eps) 헤드의 두 mlp/두 linear 모듈이 먼저
  생성된 뒤 마지막으로 생성되는 mlp 1개 + linear(bias 없음) 1개다. haiku는 생성
  순서대로 접미사를 붙이므로 STG 헤드는 항상 가장 큰 접미사(`mlp_2`, `linear_2`)를
  받는다."""
  return module_key.startswith('mlp_2') or module_key == 'linear_2'


def test_chain_logp_gradient_is_finite_and_nonzero():
  nets, ddpo = _build()
  params = nets.init(jax.random.PRNGKey(0))

  # 판별 기준이 실제 파라미터 키와 맞는지 먼저 확인
  keys = list(params.keys())
  dist_keys = [k for k in keys if _is_dist_head_key(k)]
  eps_keys = [k for k in keys if not _is_dist_head_key(k)]
  assert dist_keys == ['mlp_2/~/linear_0', 'mlp_2/~/linear_1', 'linear_2']
  assert set(eps_keys) == {'mlp/~/linear_0', 'mlp/~/linear_1', 'linear',
                           'mlp_1/~/linear_0', 'mlp_1/~/linear_1', 'linear_1'}

  B = 2
  obs = jax.random.normal(jax.random.PRNGKey(1), (B, OBS_DIM))
  _, xs = ddpo.sample_with_trace(params, obs, jax.random.PRNGKey(2))

  def loss(p):
    return ddpo.chain_logp(p, obs, xs).sum()

  grads = jax.grad(loss)(params)

  for module_key, leaf in grads.items():
    for param_name, arr in leaf.items():
      arr_np = np.asarray(arr)
      assert np.all(np.isfinite(arr_np)), f'non-finite grad at {module_key}/{param_name}'
      if _is_dist_head_key(module_key):
        assert np.all(arr_np == 0.0), f'STG head got nonzero grad at {module_key}/{param_name}'
      else:
        assert np.any(arr_np != 0.0), f'eps head got all-zero grad at {module_key}/{param_name}'
