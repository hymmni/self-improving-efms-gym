"""Tests for EnhancedPoint2D interventions.

The critical test is `test_no_intervention_matches_parent`: with no intervention
active, the subclass must be numerically identical to the parent Point2D.
"""

import numpy as np
import pytest

from pointmass_core import Point2D
from envs_enhanced import EnhancedPoint2D


def _fixed_actions(n, seed=0):
  rng = np.random.default_rng(seed)
  return (rng.normal(0, 0.002, size=(n, 2))).astype(np.float32)


def _rollout(env, actions, seed=0):
  np.random.seed(seed)
  env.reset()
  obss = []
  for a in actions:
    ts = env.step(a.copy())
    obss.append({k: v.copy() for k, v in ts.observation.items()})
  return obss


def test_no_intervention_matches_parent():
  actions = _fixed_actions(100)
  base = _rollout(Point2D(), actions, seed=123)
  enh = _rollout(EnhancedPoint2D(), actions, seed=123)
  assert len(base) == len(enh)
  for b, e in zip(base, enh):
    for k in ('cur_pos', 'cur_vel', 'goal_pos'):
      np.testing.assert_array_equal(b[k], e[k])


def test_random_action_prob_zero_matches_parent():
  actions = _fixed_actions(50)
  env = EnhancedPoint2D()
  env.set_random_action(prob=0.0, scale=0.01, seed=7)
  enh = _rollout(env, actions, seed=5)
  base = _rollout(Point2D(), actions, seed=5)
  for b, e in zip(base, enh):
    np.testing.assert_array_equal(b['cur_pos'], e['cur_pos'])


def test_teleport_applies_and_logs():
  env = EnhancedPoint2D()
  np.random.seed(0)
  env.reset()
  env.step(np.zeros(2, dtype=np.float32))
  target = np.array([-0.5, 0.7], dtype=np.float32)
  env.teleport(target, zero_velocity=True)
  ts = env.step(np.zeros(2, dtype=np.float32))
  np.testing.assert_allclose(ts.observation['cur_pos'], target, atol=1e-6)
  np.testing.assert_array_equal(ts.observation['cur_vel'], np.zeros(2))
  kinds = [k for _, k, _ in env.intervention_log]
  assert 'teleport' in kinds


def test_bias_force_shifts_trajectory():
  actions = _fixed_actions(30)
  base = _rollout(Point2D(), actions, seed=9)
  env = EnhancedPoint2D()
  env.set_bias_force(np.array([0.003, 0.0], dtype=np.float32))
  biased = _rollout(env, actions, seed=9)
  # trajectories must differ, and biased must end further in +x direction
  assert not np.allclose(base[-1]['cur_pos'], biased[-1]['cur_pos'])
  assert biased[-1]['cur_pos'][0] > base[-1]['cur_pos'][0]


def test_bias_force_clear_restores_parent():
  actions = _fixed_actions(20)
  env = EnhancedPoint2D()
  env.set_bias_force(np.array([0.003, 0.0], dtype=np.float32))
  env.set_bias_force(None)
  enh = _rollout(env, actions, seed=3)
  base = _rollout(Point2D(), actions, seed=3)
  for b, e in zip(base, enh):
    np.testing.assert_array_equal(b['cur_pos'], e['cur_pos'])


def test_obstacle_blocks_entry():
  env = EnhancedPoint2D()
  np.random.seed(0)
  env.reset()
  # place the mass and an obstacle, then push straight into it
  center = np.array([0.0, 0.0], dtype=np.float32)
  radius = 0.2
  env.teleport(np.array([-0.6, 0.0], dtype=np.float32), zero_velocity=True)
  env.step(np.zeros(2, dtype=np.float32))
  env.add_obstacle(center, radius)
  push = np.array([0.01, 0.0], dtype=np.float32)  # drive toward +x into obstacle
  for _ in range(60):
    ts = env.step(push)
    dist = np.linalg.norm(ts.observation['cur_pos'] - center)
    # never penetrates the obstacle interior (small tolerance)
    assert dist >= radius - 1e-4


def test_obstacle_add_remove():
  env = EnhancedPoint2D()
  np.random.seed(0)
  env.reset()
  oid = env.add_obstacle(np.array([0.0, 0.0], dtype=np.float32), 0.1)
  assert oid in env._obstacles
  env.remove_obstacle(oid)
  assert oid not in env._obstacles
  env.add_obstacle(np.array([0.1, 0.1], dtype=np.float32), 0.1)
  env.clear_obstacles()
  assert len(env._obstacles) == 0


def test_random_action_full_prob_diverges_and_is_reproducible():
  actions = _fixed_actions(40)
  # prob=1.0 => agent actions ignored; trajectory driven by RNG only
  env1 = EnhancedPoint2D()
  env1.set_random_action(prob=1.0, scale=0.005, seed=42)
  r1 = _rollout(env1, actions, seed=1)
  base = _rollout(Point2D(), actions, seed=1)
  assert not np.allclose(r1[-1]['cur_pos'], base[-1]['cur_pos'])
  # same feature seed + same env seed => reproducible
  env2 = EnhancedPoint2D()
  env2.set_random_action(prob=1.0, scale=0.005, seed=42)
  r2 = _rollout(env2, actions, seed=1)
  for a, b in zip(r1, r2):
    np.testing.assert_array_equal(a['cur_pos'], b['cur_pos'])


def test_render_smoke():
  env = EnhancedPoint2D()
  np.random.seed(0)
  env.reset()
  env.add_obstacle(np.array([0.0, 0.0], dtype=np.float32), 0.15)
  env.set_bias_force(np.array([0.003, 0.0], dtype=np.float32))
  env.step(np.zeros(2, dtype=np.float32))
  img = env.render(title='smoke')
  assert img.ndim == 3 and img.shape[2] == 3
  assert img.shape[0] > 0 and img.shape[1] > 0
