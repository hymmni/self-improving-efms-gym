"""Tests for the GraspCarry2D gripper body (phase 3, step 1).

각 테스트는 최소한의 `pymunk.Space`(중력 포함)를 직접 만들어 쓴다. 환경/물체/
에피소드 로직은 step 2의 범위라 여기서는 그리퍼만 검증한다.
"""

import math
import os
import sys

import pymunk
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grasp_carry.config import CarryConfig
from src.grasp_carry.gripper import Gripper


def make_space(cfg):
  space = pymunk.Space()
  space.gravity = (0.0, cfg.gravity)      # +y가 아래
  space.iterations = cfg.solver_iterations
  return space


def run(gripper, space, cfg, n, closing, target_xy=None, target_angle=0.0):
  """substep을 n번 진행한다. 매 substep: 제어력 인가 → step → 속도 제한."""
  if target_xy is None:
    target_xy = (gripper.base.position.x, gripper.base.position.y)
  for _ in range(n):
    gripper.apply_pose_control(target_xy, target_angle)
    if closing is not None:
      gripper.apply_grip(closing)
    space.step(cfg.physics_dt)
    gripper.clamp_finger_speed()


def build(cfg=None, position=(256.0, 300.0)):
  cfg = cfg or CarryConfig()
  space = make_space(cfg)
  return cfg, space, Gripper(space, cfg, position)


# --------------------------------------------------------------- 1. 개도 범위
def test_gap_stays_within_configured_opening_range():
  cfg, space, g = build()
  seen = [g.gap]

  run(g, space, cfg, 400, closing=True)
  seen.append(g.gap)
  assert g.gap == pytest.approx(cfg.finger_opening_min, abs=1.0)

  run(g, space, cfg, 400, closing=False)
  seen.append(g.gap)
  assert g.gap == pytest.approx(cfg.finger_opening_max, abs=1.0)

  for gap in seen:
    assert cfg.finger_opening_min - 1.0 <= gap <= cfg.finger_opening_max + 1.0


# --------------------------------------------------------- 2. 물체 폭에서 멈춤
def test_closing_on_object_stops_at_object_width():
  cfg = CarryConfig()
  space = make_space(cfg)
  floor = pymunk.Segment(space.static_body, (0.0, cfg.floor_y),
                         (cfg.world_width, cfg.floor_y), 4.0)
  floor.friction = 0.9
  floor.elasticity = 0.0
  space.add(floor)

  w, h = 50.0, 120.0
  mass = 0.3
  body = pymunk.Body(mass, pymunk.moment_for_box(mass, (w, h)))
  body.position = (256.0, cfg.floor_y - h / 2.0 - 4.0)
  shape = pymunk.Poly.create_box(body, (w, h))
  shape.friction = 0.5
  shape.elasticity = 0.0
  space.add(body, shape)

  # 패드(베이스 아래 60~90mm)가 물체 윗부분을 물도록 베이스를 배치한다.
  base_y = (cfg.floor_y - h - 4.0) - cfg.finger_length + cfg.pad_length
  g = Gripper(space, cfg, (256.0, base_y))
  run(g, space, cfg, 600, closing=True, target_xy=(256.0, base_y))

  assert g.gap == pytest.approx(w, abs=4.0)


# ------------------------------------------------------------- 3. 중력 보상 부호
def test_gravity_compensation_lifts_toward_target_above():
  cfg, space, g = build()
  y0 = g.base.position.y
  target = (g.base.position.x, y0 - 40.0)     # y는 아래 방향 -> 위쪽 목표

  run(g, space, cfg, 500, closing=False, target_xy=target)

  assert g.base.position.y < y0                     # 실제로 올라갔다
  assert g.base.position.y == pytest.approx(target[1], abs=3.0)


# ---------------------------------------------------------------- 4. 회전 강성
def test_pose_control_restores_angle_after_external_torque():
  cfg, space, g = build()
  hold = (g.base.position.x, g.base.position.y)

  # 외부 토크로 손목을 비튼다(제어 없이). 한 바퀴를 넘기지 않을 만큼만 준다.
  for _ in range(60):
    g.base.torque = 2.0e4
    space.step(cfg.physics_dt)
  assert 0.05 < abs(g.base.angle) < 1.0

  run(g, space, cfg, 500, closing=False, target_xy=hold, target_angle=0.0)

  assert g.base.angle == pytest.approx(0.0, abs=0.02)


# ----------------------------------------------------------- 5. 손가락 속도 제한
def test_finger_speed_relative_to_base_is_clamped():
  cfg, space, g = build()
  vmax = cfg.finger_speed_max
  worst = 0.0
  for _ in range(300):
    g.apply_pose_control((g.base.position.x, g.base.position.y), 0.0)
    g.apply_grip(closing=True)
    space.step(cfg.physics_dt)
    g.clamp_finger_speed()
    axis = pymunk.Vec2d(math.cos(g.base.angle), math.sin(g.base.angle))
    for f in g.fingers:
      worst = max(worst, abs((f.velocity - g.base.velocity).dot(axis)))
  assert worst <= vmax + 1e-6


# ------------------------------------------------------------------ 6. 삼각 형상
def test_finger_polygons_are_triangles_with_vertical_inner_edge():
  cfg, space, g = build()
  polys = g.polygons()
  assert len(polys) == 4                      # 레일 플레이트, 스템, 손가락 2개

  cx = g.base.position.x
  for poly, side in zip(polys[2:], (-1.0, 1.0)):
    assert poly.shape == (3, 2)
    xs = sorted(float(p[0]) for p in poly)
    # 정확히 두 정점이 같은 x를 공유한다 = 안쪽 변이 수직
    if side < 0:
      inner = xs[1:]                          # 왼손가락: 안쪽 = 오른쪽 두 점
    else:
      inner = xs[:2]                          # 오른손가락: 안쪽 = 왼쪽 두 점
    assert inner[0] == pytest.approx(inner[1], abs=1e-6)
    assert abs(inner[0] - cx) == pytest.approx(g.gap / 2.0, abs=1e-6)
