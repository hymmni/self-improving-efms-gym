"""Tests for `src/train_carry_dstg.py` (phase 4, step 1 — 관측-only d(o,g))."""

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.train_carry_dstg import build_dstg_net

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'grasp_carry_demos_v3.pkl')


def _run_tiny_training(tmpdir, include_failures):
  out = os.path.join(tmpdir, 'succ' if not include_failures else 'fail',
                     'predictor.pkl')
  cmd = [sys.executable, '-m', 'src.train_carry_dstg', '--data', DATA_PATH,
         '--steps', '50', '--eval-every', '25', '--no-early-stop',
         '--out', out]
  if include_failures:
    cmd.append('--include-failures')
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  subprocess.run(cmd, cwd=repo_root, check=True,
                 capture_output=True, text=True)
  with open(out, 'rb') as fp:
    return pickle.load(fp)


def test_checkpoint_schema():
  with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_succ = _run_tiny_training(tmpdir, include_failures=False)
    ckpt_fail = _run_tiny_training(tmpdir, include_failures=True)

  required_keys = {'params', 'norm_stats', 'obs_dim', 'num_bins', 'fail_bin',
                   'max_steps', 'layer_sizes', 'meta'}
  for ckpt in (ckpt_succ, ckpt_fail):
    assert required_keys.issubset(ckpt.keys())
    assert set(ckpt['norm_stats'].keys()) == {'frame_mean', 'frame_std'}
    assert isinstance(ckpt['obs_dim'], int)
    assert isinstance(ckpt['num_bins'], int)
    assert isinstance(ckpt['max_steps'], int)
    assert tuple(ckpt['layer_sizes']) == (256, 256, 256)
    assert 'val_mae' in ckpt['meta'] and 'val_nll' in ckpt['meta']

  assert ckpt_succ['fail_bin'] is None
  assert ckpt_succ['num_bins'] == 200
  assert ckpt_succ['meta']['include_failures'] is False
  assert 'val_fail_accuracy' not in ckpt_succ['meta']

  assert ckpt_fail['fail_bin'] == 200
  assert ckpt_fail['num_bins'] == 201
  assert ckpt_fail['meta']['include_failures'] is True
  assert 'val_fail_accuracy' in ckpt_fail['meta']
  assert 'val_fail_base_rate' in ckpt_fail['meta']


def test_bin_labels():
  with open(DATA_PATH, 'rb') as fp:
    data = pickle.load(fp)
  is_succ = data['is_success']
  ttg = data['time_to_success']
  max_steps = int(data['meta']['max_steps'])

  # --include-failures 라벨: 실패 에피소드는 전부 max_steps(=200), 성공은 0..199
  assert np.all(ttg[~is_succ] == max_steps)
  assert np.all(ttg[is_succ] >= 0) and np.all(ttg[is_succ] < max_steps)

  # --include-failures 없음: 학습에 쓰이는 라벨(성공만)의 최댓값은 199 이하
  assert ttg[is_succ].max() <= max_steps - 1


def test_obs_only():
  obs_dim, num_bins = 60, 200
  apply_fn, init_fn = build_dstg_net((32, 32), obs_dim, num_bins)
  import jax
  params = init_fn(jax.random.PRNGKey(0))
  obs = jnp.zeros((5, obs_dim), jnp.float32)
  logits = apply_fn(params, obs)
  assert logits.shape == (5, num_bins)

  import inspect
  sig = inspect.signature(apply_fn)
  # apply_fn(params, obs) 만 받는다 (haiku apply 시그니처 — 액션 인자 없음)
  assert len(sig.parameters) == 2
