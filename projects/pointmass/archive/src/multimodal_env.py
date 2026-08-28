"""Multimodal reaching map for the variance-reward experiments.

A single obstacle sits between a (jittered) fixed start and goal, so a successful
path must detour LEFT or RIGHT around it. Demonstrations are generated ~50/50 on
each side, which makes the steps-to-go label distribution *bimodal* at the
decision region: from near the start, both a left and a right route reach the
goal in a similar number of steps, so a distribution predictor trained on this
data must place mass on two modes (spec section 3 requirement).

Reuses phase-1 assets (EnhancedPoint2D obstacle mechanics, pd_controller) rather
than defining a new environment (ADR-005).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmass_core import pd_controller, DataTuple  # noqa: E402
from envs_enhanced import EnhancedPoint2D            # noqa: E402
import jax                                           # noqa: E402


# Fixed geometry of the map (world coords in [-1, 1]^2).
GOAL = np.array([0.0, 0.7], dtype=np.float32)
START = np.array([0.0, -0.7], dtype=np.float32)
OBSTACLE_CENTER = np.array([0.0, 0.0], dtype=np.float32)
OBSTACLE_RADIUS = 0.35
# Asymmetric detours: the left route is short, the right route is long. This is
# deliberate — it makes the steps-to-go label BIMODAL near the start (a
# center-start state is reached by both a short-left and a long-right demo), which
# is exactly the multimodal signal the experiments need. A symmetric map would
# make both routes equal length and the distribution unimodal.
LEFT_WAYPOINT = np.array([-0.5, 0.0], dtype=np.float32)
RIGHT_WAYPOINT = np.array([1.0, 0.0], dtype=np.float32)


class MultiModalPoint2D(EnhancedPoint2D):
  """EnhancedPoint2D with a fixed obstacle between a jittered start and goal."""

  def __init__(self, jitter: float = 0.08):
    super().__init__()
    self._jitter = jitter

  def reset(self):
    # Set the fixed (jittered) start/goal instead of the parent's random sample.
    self._goal_pos = (GOAL + np.random.uniform(
        -self._jitter, self._jitter, size=2)).astype(np.float32)
    start = (START + np.random.uniform(
        -self._jitter, self._jitter, size=2)).astype(np.float32)
    self._cur_pos = start.copy()
    self._cur_vel = np.zeros(2, dtype=np.float32)
    self._cur_episode_traj = [start.copy()]

    # (re)install the single fixed obstacle
    self._obstacles = {}
    self._next_obstacle_id = 0
    self.add_obstacle(OBSTACLE_CENTER, OBSTACLE_RADIUS)

    self._step_count = 0
    self._intervention_log = []
    self._pending_teleport = None

    import dm_env
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST, reward=None, discount=None,
        observation={'cur_pos': self._cur_pos.copy(),
                     'cur_vel': self._cur_vel.copy(),
                     'goal_pos': self._goal_pos.copy()})


def routed_pd_controller(cur_pos, cur_vel, goal_pos, side, obstacle_center=None,
                         obstacle_radius=None):
  """PD controller that detours around the obstacle via a side waypoint.

  side: 'left' or 'right'. Steers toward the side waypoint until the mass has
  cleared the obstacle band (y above the obstacle top), then steers to the goal.
  """
  oc = OBSTACLE_CENTER if obstacle_center is None else obstacle_center
  orad = OBSTACLE_RADIUS if obstacle_radius is None else obstacle_radius
  waypoint = LEFT_WAYPOINT if side == 'left' else RIGHT_WAYPOINT
  cleared = cur_pos[1] > oc[1] + orad * 0.5
  # also switch to goal once we're horizontally past the waypoint's x extent
  near_waypoint = np.linalg.norm(cur_pos - waypoint) < 0.2
  target = goal_pos if (cleared or near_waypoint) else waypoint
  return pd_controller(cur_pos, cur_vel, target)


def generate_multimodal_dataset(num_episodes=4000, jitter=0.08,
                                max_steps=200, discard_thresh=10, seed=0):
  """Generates ~50/50 left/right detour demonstrations.

  Returns (episodes, all_tuples) in the same DataTuple format as
  pointmass_core.generate_dataset, with time_to_success labeled per step.
  """
  np.random.seed(seed)
  env = MultiModalPoint2D(jitter=jitter)
  episodes = []
  side_counts = {'left': 0, 'right': 0}
  attempts = 0
  max_attempts = num_episodes * 4  # guard against a broken routing config
  while len(episodes) < num_episodes:
    attempts += 1
    if attempts > max_attempts:
      raise RuntimeError(
          f'only {len(episodes)}/{num_episodes} episodes succeeded after '
          f'{attempts} attempts — routing config likely broken')
    side = 'left' if (len(episodes) % 2 == 0) else 'right'
    ts = env.reset()
    cur_obs = ts.observation
    traj = []
    steps = 0
    while (not env.success()) and steps < max_steps:
      act = routed_pd_controller(
          cur_obs['cur_pos'], cur_obs['cur_vel'], cur_obs['goal_pos'], side)
      ts = env.step(act)
      traj.append(DataTuple(
          observation=cur_obs, action=act, time_to_success=0.,
          reward=ts.reward if ts.reward is not None else 0., discount=1.,
          next_observation=ts.observation))
      cur_obs = ts.observation
      steps += 1

    if not env.success() or len(traj) < discard_thresh:
      continue  # drop failed / too-short detours

    new_traj = jax.tree.map(lambda *xs: np.stack(xs, dtype=np.float32), *traj)
    new_traj = new_traj._replace(
        time_to_success=np.arange(len(traj) - 1, -1, -1, dtype=np.float32))
    episodes.append(new_traj)
    side_counts[side] += 1

  all_tuples = jax.tree.map(
      lambda *xs: np.concatenate(xs, dtype=np.float32), *episodes)
  return episodes, all_tuples, side_counts
