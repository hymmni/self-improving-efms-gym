"""Tests for `src/carry_stg_reward.py` (phase 4, step 2 — d/보상/성공판정 래퍼).

실제 학습 체크포인트 없이 돌아가도록, 작은 랜덤 네트워크로 즉석 체크포인트를
만들어 쓴다(`_write_tiny_ckpt`). 통계량 계산(`_d_from_probs`)은 확률 배열을
직접 주입해 순수 함수로 검증한다.
"""

import os
import pickle
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_carry_dstg import build_dstg_net
from src.carry_stg_reward import StgReward, _d_from_probs, calibrate_threshold

OBS_DIM = 6
LAYER_SIZES = (8, 8)


def _write_tiny_ckpt(path, num_bins, fail_bin, seed=0):
  _, init_fn = build_dstg_net(LAYER_SIZES, OBS_DIM, num_bins)
  params = init_fn(jax.random.PRNGKey(seed))
  max_steps = fail_bin if fail_bin is not None else num_bins
  ckpt = {
      'params': jax.device_get(params),
      'norm_stats': {'frame_mean': np.zeros(OBS_DIM, dtype=np.float32),
                     'frame_std': np.ones(OBS_DIM, dtype=np.float32)},
      'obs_dim': OBS_DIM,
      'num_bins': num_bins,
      'fail_bin': fail_bin,
      'max_steps': max_steps,
      'layer_sizes': LAYER_SIZES,
      'meta': {'seed': seed},
  }
  with open(path, 'wb') as fp:
    pickle.dump(ckpt, fp)


# ---------------------------------------------------------- 1. shape/monotone
def test_d_shapes_and_monotone_bins():
  with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'predictor.pkl')
    _write_tiny_ckpt(ckpt_path, num_bins=5, fail_bin=None)
    reward = StgReward(ckpt_path, statistic='mean')

    obs = np.random.default_rng(0).normal(size=(7, OBS_DIM)).astype(np.float32)
    d = reward.d(obs)
    assert d.shape == (7,)

  bin_vals = jnp.arange(5, dtype=jnp.float32)
  probs_low = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]])   # 전부 bin 0 (가깝다)
  probs_high = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]])  # 전부 bin 4 (멀다)
  d_low = _d_from_probs(probs_low, bin_vals, 'mean', 0.8)
  d_high = _d_from_probs(probs_high, bin_vals, 'mean', 0.8)
  assert float(d_high[0]) > float(d_low[0])


# --------------------------------------------------------------- 2. cvar>=mean
def test_cvar_ge_mean():
  num_bins = 12
  key = jax.random.PRNGKey(3)
  logits = jax.random.normal(key, (9, num_bins))
  probs = jax.nn.softmax(logits, axis=-1)
  bin_vals = jnp.arange(num_bins, dtype=jnp.float32)

  for alpha in (0.5, 0.8, 0.95):
    mean = _d_from_probs(probs, bin_vals, 'mean', alpha)
    cvar = _d_from_probs(probs, bin_vals, 'cvar', alpha)
    assert bool(jnp.all(cvar >= mean - 1e-4)), (alpha, mean, cvar)


# ------------------------------------------------------------ 3. fail bin 영향
def test_fail_bin_raises_d():
  num_bins = 5
  fail_bin = 4
  fail_value = 100.0
  bin_vals = jnp.arange(num_bins, dtype=jnp.float32).at[fail_bin].set(fail_value)

  # 성공 bin(0..3) 쪽 상대 분포는 동일하게 두고, 실패 bin에 실리는 질량만 바꾼다.
  succ_shape = jnp.array([0.4, 0.3, 0.2, 0.1])
  probs_a = jnp.concatenate([succ_shape * (1 - 0.1), jnp.array([0.1])])[None]
  probs_b = jnp.concatenate([succ_shape * (1 - 0.5), jnp.array([0.5])])[None]

  d_a = _d_from_probs(probs_a, bin_vals, 'mean', 0.8)
  d_b = _d_from_probs(probs_b, bin_vals, 'mean', 0.8)
  assert float(d_b[0]) > float(d_a[0])


# ---------------------------------------------------------------- 4. 얼려짐
def test_frozen():
  with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'predictor.pkl')
    _write_tiny_ckpt(ckpt_path, num_bins=6, fail_bin=5)
    reward = StgReward(ckpt_path, statistic='mean')

    params_before = jax.tree.map(np.copy, reward.params)
    obs = np.random.default_rng(1).normal(size=(4, OBS_DIM)).astype(np.float32)

    d1 = reward.d(obs)
    d2 = reward.d(obs)
    np.testing.assert_allclose(np.asarray(d1), np.asarray(d2))

    params_after = reward.params
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b),
                 params_before, params_after)


# ------------------------------------------------------- 5. calibrate_threshold
def test_calibrate_threshold_runs_and_returns_metrics():
  with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, 'predictor.pkl')
    _write_tiny_ckpt(ckpt_path, num_bins=8, fail_bin=None)
    reward = StgReward(ckpt_path, statistic='mean')

  rng = np.random.default_rng(0)
  n_eps, ep_len = 4, 5
  ep_ids = np.repeat(np.arange(n_eps), ep_len)
  ttg = np.tile(np.arange(ep_len - 1, -1, -1), n_eps).astype(np.float32)
  data = {
      'observation': {'frame': rng.normal(size=(n_eps * ep_len, OBS_DIM))
                      .astype(np.float32)},
      'episode_id': ep_ids,
      'is_success': np.ones(n_eps * ep_len, dtype=bool),
      'time_to_success': ttg,
  }
  val_eps = {0, 1}
  best_s, metrics = calibrate_threshold(reward, data, val_eps)
  assert isinstance(best_s, float)
  for k in ('precision', 'recall', 'f1', 'best_s'):
    assert k in metrics
