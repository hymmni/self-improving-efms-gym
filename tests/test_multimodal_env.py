"""Tests for the multimodal reaching map (phase 2, step 0)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_env import (
    generate_multimodal_dataset, MultiModalPoint2D, routed_pd_controller,
    OBSTACLE_CENTER, OBSTACLE_RADIUS)


def test_reset_places_obstacle_and_fixed_region():
  env = MultiModalPoint2D(jitter=0.08)
  np.random.seed(0)
  ts = env.reset()
  assert len(env._obstacles) == 1
  # start is in the lower region, goal in the upper region
  assert ts.observation['cur_pos'][1] < 0
  assert ts.observation['goal_pos'][1] > 0


def test_dataset_is_balanced_and_avoids_obstacle():
  eps, tuples, sides = generate_multimodal_dataset(num_episodes=40, seed=0)
  assert sides['left'] == sides['right'] == 20
  # no demonstration step penetrates the obstacle interior
  for e in eps:
    d = np.linalg.norm(e.observation['cur_pos'] - OBSTACLE_CENTER, axis=1)
    assert d.min() >= OBSTACLE_RADIUS - 1e-3


def test_steps_to_go_is_bimodal():
  eps, _, _ = generate_multimodal_dataset(num_episodes=40, seed=0)
  lens = np.array([e.observation['cur_pos'].shape[0] for e in eps])
  left, right = lens[0::2], lens[1::2]
  # the two routes have clearly separated lengths (bimodal STG at the start)
  assert right.mean() - left.mean() > 3.0
  # each mode is tight (near-deterministic per side)
  assert left.std() < 2.0 and right.std() < 2.0


def test_routed_controller_sides_differ():
  # from the same start, left vs right waypoint produce different first actions
  np.random.seed(0)
  env = MultiModalPoint2D(jitter=0.0)
  ts = env.reset()
  o = ts.observation
  a_left = routed_pd_controller(o['cur_pos'], o['cur_vel'], o['goal_pos'], 'left')
  a_right = routed_pd_controller(o['cur_pos'], o['cur_vel'], o['goal_pos'], 'right')
  assert not np.allclose(a_left, a_right)
  assert a_left[0] < a_right[0]  # left steers -x, right steers +x
