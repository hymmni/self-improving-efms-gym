"""Enhanced pointmass environment with intervention hooks.

`EnhancedPoint2D` subclasses `pointmass_core.Point2D` (ADR-005: additive,
non-invasive extension) and adds four perturbation mechanisms used to probe how
the TIMER steps-to-go prediction distribution reacts (see references/context_2.md):

  1. teleport            -- instantaneous displacement of the point mass
  2. bias force          -- constant external force added every physics substep
  3. obstacles           -- dynamically created circular walls the mass cannot enter
  4. random actions       -- stochastic replacement of the agent action

Design invariant (most important): with NO intervention active, `EnhancedPoint2D`
is bit-for-bit identical to `Point2D` for the same seed and action sequence. Every
intervention is guarded so that, when inactive, it neither changes the numerics nor
consumes RNG.
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Intervention labels/annotations are Korean; DejaVu Sans (matplotlib default)
# has no Hangul glyphs and would render them as tofu boxes.
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pointmass_core import (
    Point2D,
    BOUNDS_X,
    BOUNDS_Y,
    DPI,
    RENDER_HEIGHT_INCHES,
)
import dm_env


class Obstacle(NamedTuple):
  center: np.ndarray  # (2,)
  radius: float


class EnhancedPoint2D(Point2D):
  """Point2D with teleport / bias-force / obstacle / random-action hooks."""

  def __init__(self):
    super().__init__()
    self._bias_force: Optional[np.ndarray] = None
    self._obstacles: dict = {}          # id -> Obstacle
    self._next_obstacle_id: int = 0
    self._rand_prob: float = 0.0
    self._rand_scale: float = 0.0
    self._rng: Optional[np.random.Generator] = None
    self._pending_teleport: Optional[Tuple[np.ndarray, bool]] = None
    self._step_count: int = 0
    self._intervention_log: List[tuple] = []

  # -- reset also clears per-episode bookkeeping (interventions persist) -------
  def reset(self):
    ts = super().reset()
    self._step_count = 0
    self._intervention_log = []
    self._pending_teleport = None
    return ts

  # ------------------------------------------------------------------ teleport
  def teleport(self, pos: np.ndarray, zero_velocity: bool = False) -> None:
    """Queue an instantaneous move to `pos`, applied at the start of the next
    `step()` (before physics). Optionally zero the velocity. Logged with the
    step index at which it will take effect."""
    self._pending_teleport = (np.asarray(pos, dtype=np.float32).copy(),
                              zero_velocity)

  # ---------------------------------------------------------------- bias force
  def set_bias_force(self, force: Optional[np.ndarray]) -> None:
    """Constant force added to the action on every physics substep. None clears
    it (restoring exact parent behavior)."""
    if force is None:
      self._bias_force = None
    else:
      self._bias_force = np.asarray(force, dtype=np.float32).copy()
    self._intervention_log.append(
        (self._step_count, 'bias_force',
         None if force is None else tuple(float(x) for x in force)))

  # ------------------------------------------------------------------ obstacles
  def add_obstacle(self, center: np.ndarray, radius: float) -> int:
    obs_id = self._next_obstacle_id
    self._next_obstacle_id += 1
    self._obstacles[obs_id] = Obstacle(
        np.asarray(center, dtype=np.float32).copy(), float(radius))
    self._intervention_log.append(
        (self._step_count, 'add_obstacle', (obs_id, tuple(center), radius)))
    return obs_id

  def remove_obstacle(self, obstacle_id: int) -> None:
    if obstacle_id in self._obstacles:
      del self._obstacles[obstacle_id]
      self._intervention_log.append(
          (self._step_count, 'remove_obstacle', obstacle_id))

  def clear_obstacles(self) -> None:
    self._obstacles = {}
    self._intervention_log.append((self._step_count, 'clear_obstacles', None))

  # -------------------------------------------------------------- random action
  def set_random_action(self, prob: float, scale: float,
                        seed: Optional[int] = None) -> None:
    """With probability `prob`, replace the agent action with a
    N(0, scale^2) sample each step. prob=0.0 disables it (parent behavior,
    no RNG consumed). Uses a private Generator so global np.random is untouched."""
    self._rand_prob = float(prob)
    self._rand_scale = float(scale)
    if self._rng is None or seed is not None:
      self._rng = np.random.default_rng(seed)

  def _maybe_random_action(self, action):
    # Guarded so that a disabled feature consumes no RNG and returns the exact
    # same array object the caller passed (preserving bit-exactness).
    if self._rand_prob <= 0.0:
      return action
    if self._rng is None:
      self._rng = np.random.default_rng()
    if self._rng.random() < self._rand_prob:
      rand = self._rng.normal(0.0, self._rand_scale, size=2).astype(np.float32)
      self._intervention_log.append((self._step_count, 'random_action', rand))
      return rand
    return action

  # ------------------------------------------------------------- collision math
  def _resolve_obstacle_collisions(self) -> None:
    for obs in self._obstacles.values():
      delta = self._cur_pos - obs.center
      dist = float(np.linalg.norm(delta))
      if dist < obs.radius:
        if dist > 1e-8:
          direction = delta / dist
        else:
          direction = np.array([1.0, 0.0], dtype=np.float32)
        # Project to the obstacle surface and stop (wall-like).
        self._cur_pos = (obs.center + direction * obs.radius).astype(np.float32)
        self._cur_vel = np.zeros(2, dtype=np.float32)

  # ------------------------------------------------------------------- step
  # From: pointmass_core.Point2D.step (notebook cell 4)
  # MODIFIED: intervention hooks (teleport / random action / bias force /
  # obstacle collisions) are woven into the copied physics loop. Each hook is a
  # no-op when its feature is inactive, so the numerics reduce exactly to the
  # parent's `for substep: vel += action; pos += vel`.
  def step(self, action):
    # 1) pending teleport (before physics)
    if self._pending_teleport is not None:
      pos, zero_vel = self._pending_teleport
      from_pos = self._cur_pos.copy()
      self._cur_pos = pos.copy()
      if zero_vel:
        self._cur_vel = np.zeros(2, dtype=np.float32)
      self._intervention_log.append(
          (self._step_count, 'teleport', (tuple(from_pos), tuple(pos), zero_vel)))
      self._cur_episode_traj.append(self._cur_pos.copy())
      self._pending_teleport = None

    # 2) random-action substitution (no-op / no RNG when disabled)
    action = self._maybe_random_action(action)

    # 3) physics substeps (parent loop + guarded bias & collision hooks)
    bias = self._bias_force
    for _ in range(self._physics_substeps):
      if bias is not None:
        self._cur_vel += action + bias
      else:
        self._cur_vel += action
      self._cur_pos += self._cur_vel
      if self._obstacles:
        self._resolve_obstacle_collisions()

    # 4) build TimeStep (verbatim from parent)
    cur_pos_copy = self._cur_pos.copy()
    cur_vel_copy = self._cur_vel.copy()
    obs = {
        'cur_pos': cur_pos_copy,
        'cur_vel': cur_vel_copy,
        'goal_pos': self._goal_pos.copy()}

    if self.success():
      step_type = dm_env.StepType.LAST
    else:
      step_type = dm_env.StepType.MID
    ts = dm_env.TimeStep(
        step_type=step_type,
        reward=-1. * np.linalg.norm(self._cur_pos - self._goal_pos),
        discount=1.,
        observation=obs,)

    self._cur_episode_traj.append(cur_pos_copy)
    self._step_count += 1
    return ts

  @property
  def intervention_log(self) -> List[tuple]:
    return list(self._intervention_log)

  # ------------------------------------------------------------------- render
  # From: pointmass_core.Point2D.render (notebook cell 4)
  # MODIFIED: draws active obstacles (gray filled circles) and the bias force
  # (arrow from the current position) on top of the parent's rendering.
  def render(self, title: str = '', points: Optional[np.ndarray] = None,
             goal_pos: Optional[np.ndarray] = None):
    fig, ax = plt.subplots(
        figsize=(RENDER_HEIGHT_INCHES, RENDER_HEIGHT_INCHES), dpi=DPI)
    ax.set_xlim(BOUNDS_X[0], BOUNDS_X[1])
    ax.set_ylim(BOUNDS_Y[0], BOUNDS_Y[1])
    ax.set_aspect('equal')

    if points is None:
      points = np.array(self._cur_episode_traj)
      cur_pos = self._cur_pos
    else:
      cur_pos = points[-1]
    if goal_pos is None:
      goal_pos = self._goal_pos

    # obstacles first (under the trajectory)
    for obs in self._obstacles.values():
      ax.add_patch(patches.Circle(
          (obs.center[0], obs.center[1]), obs.radius,
          facecolor='gray', edgecolor='black', alpha=0.5, linewidth=2))

    ax.plot(points[:, 0], points[:, 1], marker='.', color='blue',
            markersize=16, linewidth=4)
    ax.scatter(goal_pos[0], goal_pos[1], marker='*', s=200, color='orange',
               linewidths=8)
    ax.scatter(cur_pos[0], cur_pos[1], marker='o', s=100, color='red',
               linewidths=8)

    circle = patches.Circle(
        (goal_pos[0], goal_pos[1]), self._success_radius,
        edgecolor='green', linestyle='--', linewidth=4, fill=False)
    ax.add_patch(circle)

    # bias force arrow: fixed on-screen length in the bias direction (the raw
    # vector is tiny -- e.g. 2e-5 -- and invisible at true scale, since it is
    # added every physics substep and accumulates over the episode instead of
    # being drawn to scale). The exact vector is printed as text so magnitude
    # is never implied by the arrow's length.
    if self._bias_force is not None:
      norm = float(np.linalg.norm(self._bias_force))
      if norm > 1e-12:
        arrow_len = 0.22
        dx = float(self._bias_force[0]) / norm * arrow_len
        dy = float(self._bias_force[1]) / norm * arrow_len
        ax.arrow(cur_pos[0], cur_pos[1], dx, dy,
                 head_width=0.045, head_length=0.045, fc='purple',
                 ec='purple', linewidth=3, zorder=6)
        ax.text(0.02, 0.02,
                f'외력: ({self._bias_force[0]:.1e}, {self._bias_force[1]:.1e})',
                transform=ax.transAxes, fontsize=10, color='purple',
                fontweight='bold', va='bottom')

    for spine in ax.spines.values():
      spine.set_linewidth(4)
    if title != '':
      ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    plt.close(fig)
    return image
