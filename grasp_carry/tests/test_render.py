"""Tests for the GraspCarry2D renderer (phase 3, step 4).

렌더러는 검증 도구이므로 여기서는: 프레임 형식, 그리퍼가 화면 밖으로 잘리지
않는지, 그리고 렌더링이 부작용(env 상태 변경) 없이 순수하게 읽기만 하는지를
검증한다.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy

from grasp_carry.scripts.record.record_carry import draw_env, render_frame, view_limits


def _hold_action(env):
  ex, ey, eth = env.gripper.pose
  return (ex, ey, eth, 0.0)


# ---------------------------------------------------------- 1. 프레임 형식
def test_render_frame_returns_hwc_uint8_array():
  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  env.reset(seed=0)
  frame = render_frame(env, action=_hold_action(env))
  assert frame.dtype == np.uint8
  assert frame.ndim == 3
  assert frame.shape[2] == 3
  assert frame.shape[0] > 0 and frame.shape[1] > 0


# --------------------------------------------------- 2. 그리퍼가 화면 안에
def test_gripper_polygons_fit_inside_view_limits():
  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  policy = ScriptedCarryPolicy(speed=90.0, config=cfg)  # 큰 리드 -> PD 오버슈트 유발

  for seed in (0, 5, 10):
    env.reset(seed=seed)
    policy.reset()
    for _ in range(30):
      action = policy(env)
      _, _, term, trunc, _ = env.step(action)
      (x_lo, x_hi), (bottom, top) = view_limits(env)
      for poly in env.gripper.polygons():
        assert np.all(poly[:, 0] >= x_lo - 1e-6)
        assert np.all(poly[:, 0] <= x_hi + 1e-6)
        assert np.all(poly[:, 1] <= bottom + 1e-6)
        assert np.all(poly[:, 1] >= top - 1e-6)
      if term or trunc:
        break


# ------------------------------------------------- 3. 렌더링은 부작용이 없다
def test_rendering_does_not_mutate_env_state():
  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  env.reset(seed=3)
  policy = ScriptedCarryPolicy(speed=30.0, config=cfg)
  for _ in range(15):
    _, _, term, trunc, _ = env.step(policy(env))
    if term or trunc:
      break

  pos_before = (float(env.block_body.position.x),
                float(env.block_body.position.y))
  angle_before = float(env.block_body.angle)
  ee_before = env.gripper.pose

  render_frame(env, action=_hold_action(env))
  fig_axes = None
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  draw_env(ax, env, action=_hold_action(env))
  plt.close(fig)

  pos_after = (float(env.block_body.position.x),
              float(env.block_body.position.y))
  angle_after = float(env.block_body.angle)
  ee_after = env.gripper.pose

  assert pos_before == pos_after
  assert angle_before == angle_after
  assert ee_before == ee_after
