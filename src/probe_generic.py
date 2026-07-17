"""관측 필드 구성이 다른 체크포인트를 모두 다루는 범용 STG 프로브.

stg_probe.STGProbe는 3필드(cur_pos, cur_vel, goal_pos) concat이 하드코딩되어
있어 phase-3의 5필드 체크포인트를 못 읽는다. 이 프로브는 체크포인트의
'obs_fields' 항목(없으면 원본 3필드로 폴백)을 읽어 어떤 필드 구성이든 동일한
API(query / rollout / STGRecord)를 제공한다. 정규화기는 필드 구성에 따라
원본 make_normalizers 또는 phase-3 make_normalizers_obstacle을 선택한다.
"""

import pickle
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

from pointmass_core import (
    build_continuous_act_discrete_dist_v0,
    build_discrete_distance_converter,
    make_normalizers,
)
from stg_probe import STGRecord
from src.train_obstacle_predictor import make_normalizers_obstacle

_DEFAULT_FIELDS = ('cur_pos', 'cur_vel', 'goal_pos')
_FIELD_DIMS = {'cur_pos': 2, 'cur_vel': 2, 'goal_pos': 2,
               'obstacle_rel_pos': 2, 'obstacle_radius': 1}


class GenericSTGProbe:
  """체크포인트 자기술(obs_fields) 기반의 STG 분포 프로브."""

  def __init__(self, checkpoint_path: str):
    with open(checkpoint_path, 'rb') as fp:
      ck = pickle.load(fp)
    self.obs_fields = tuple(ck.get('obs_fields', _DEFAULT_FIELDS))
    obs_dim = sum(_FIELD_DIMS[f] for f in self.obs_fields)

    dc_cfg = ck['dc_config']
    self.dc = build_discrete_distance_converter(
        dc_cfg['min_distance'], dc_cfg['max_distance'], dc_cfg['num_bins'])
    self.bin_vals = np.linspace(
        dc_cfg['min_distance'], dc_cfg['max_distance'], dc_cfg['num_bins'] + 1,
        endpoint=True, dtype=np.float32)[:-1]
    self.nets = build_continuous_act_discrete_dist_v0(
        (256, 256, 256), 2, dc_cfg['num_bins'],
        np.ones((4, obs_dim), dtype=np.float32))
    if 'obstacle_rel_pos' in self.obs_fields:
      self.normalize_obs, _, self.unnormalize_action = \
          make_normalizers_obstacle(ck['norm_stats'])
    else:
      self.normalize_obs, _, self.unnormalize_action = \
          make_normalizers(ck['norm_stats'])
    self.params = ck['params']
    self.meta = ck.get('meta', {})

    def _infer(params, concat, rng):
      preds = self.nets.network.apply(params, concat)
      act = self.nets.sample_act(preds.act_dist_params, rng)
      return act, preds.dist_to_succ_dist_params.logits

    self._infer = jax.jit(_infer, backend='cpu')

  # ---- STGProbe와 동일 API --------------------------------------------------
  def _logits_and_act(self, obs, rng):
    norm = self.normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
    concat = jnp.concatenate(
        [jnp.asarray(norm[f]) for f in self.obs_fields], axis=-1)
    act, logits = self._infer(self.params, concat, rng)
    return np.asarray(act), np.asarray(logits)[0]

  # From: stg_probe.STGProbe._record — 동일 수식 (기존 결과와 직접 비교 위함)
  def _record(self, step_idx, obs, logits):
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    exp = float(np.sum(self.bin_vals * probs))
    var = float(np.sum(self.bin_vals ** 2 * probs) - exp ** 2)
    ent = float(-np.sum(probs * np.log(probs + 1e-12)))
    return STGRecord(step_idx, {k: np.asarray(v).copy() for k, v in obs.items()},
                     probs, exp, var, ent)

  def query(self, obs: dict, rng=None) -> STGRecord:
    if rng is None:
      rng = jax.random.PRNGKey(0)
    _, logits = self._logits_and_act(obs, rng)
    return self._record(-1, obs, logits)

  def rollout(self, env, max_steps: int = 500, policy: str = 'learned',
              seed: int = 0, action_source: Optional[Callable] = None) -> list:
    np.random.seed(seed)
    key = jax.random.PRNGKey(42)
    ts = env.reset()
    records = []
    step_idx = 0
    while (not env.success()) and step_idx < max_steps:
      obs = ts.observation
      key, sub = jax.random.split(key)
      act, logits = self._logits_and_act(obs, sub)
      records.append(self._record(step_idx, obs, logits))
      if policy == 'learned':
        action = self.unnormalize_action(act)[0]
      elif policy == 'external':
        action = np.asarray(action_source(obs), dtype=np.float32)
      else:
        raise ValueError(f'unknown policy {policy!r}')
      ts = env.step(action)
      step_idx += 1
    key, sub = jax.random.split(key)
    _, logits = self._logits_and_act(ts.observation, sub)
    records.append(self._record(step_idx, ts.observation, logits))
    return records
