"""Tests for the GraspCarry2D environment (phase 3, step 2).

정책·렌더링은 이후 step의 범위라 여기서는 환경의 계약만 검증한다:
은닉 물성의 비노출, 시드 결정성, 하드 스톱, 종료 판정, 월드 탈출 방지.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.config import CarryConfig
from grasp_carry.env import FRAME_FIELDS, GraspCarry2D


def hold_action(env):
  """그리퍼를 제자리에 두고 손을 벌린 액션 — 블록을 건드리지 않는다."""
  ex, ey, eth = env.gripper.pose
  return (ex, ey, eth, 0.0)


def place_block(env, center_x, bottom_y, angle):
  """블록의 **바닥**이 `bottom_y`에 오도록 강제 배치한다(0.5mm 띄움).

  세운 상태(angle=0)면 세로 반높이가 `block_h/2`, 눕힌 상태(angle=90deg)면
  `block_w/2`다.
  """
  body = env.block_body
  body.angle = angle
  half = env.block_w / 2.0 if abs(angle) > 1e-6 else env.block_h / 2.0
  body.position = (center_x, bottom_y - half - 0.5)
  body.velocity = (0.0, 0.0)
  body.angular_velocity = 0.0


def run_until_terminated(env, n=12):
  """정지 액션으로 최대 n 스텝 진행하고 마지막 (terminated, info)를 준다."""
  terminated, info = False, env._info()
  for _ in range(n):
    _, _, terminated, _, info = env.step(hold_action(env))
    if terminated:
      break
  return terminated, info


# ------------------------------------------------- 1. 은닉 물성이 관측에 없다
def test_hidden_properties_never_reach_the_observation():
  env = GraspCarry2D()
  env.reset(seed=0)

  # (a) 관측 스키마 자체에 은닉 물성 필드가 없다.
  for name in FRAME_FIELDS:
    assert 'mass' not in name and 'friction' not in name and 'mu' not in name
  assert len(FRAME_FIELDS) == 15
  assert env.obs_dim == 15 * env.cfg.obs_history

  # (b) 은닉값만 바꿔도 관측은 한 비트도 달라지지 않는다.
  before = env.observe_frame().copy()
  env.block_body.mass = env.block_body.mass * 5.0
  env.block_shape.friction = 0.99
  after = env.observe_frame()
  np.testing.assert_array_equal(before, after)

  # 그래도 진단용 info에는 실려 있어야 한다.
  info = env._info()
  assert info['mass'] > 0.0 and info['friction'] > 0.0


# ---------------------------------------------------------- 2. 시드 결정성
def test_reset_is_deterministic_given_the_seed():
  env = GraspCarry2D()

  def episode(seed):
    obs, _ = env.reset(seed=seed)
    return (env.block_w, env.block_h,
            float(env.block_body.position.x),
            env.src_box.inner_width, obs)

  a = episode(7)
  b = episode(7)
  for x, y in zip(a[:4], b[:4]):
    assert x == pytest.approx(y, abs=1e-12)
  np.testing.assert_array_equal(a[4], b[4])

  c = episode(8)
  assert (a[0], a[1], a[2], a[3]) != (c[0], c[1], c[2], c[3])


# ------------------------------------------------------------- 3. 관측 차원
def test_reset_observation_matches_obs_dim():
  env = GraspCarry2D()
  obs, info = env.reset(seed=1)
  assert obs.shape == (env.obs_dim,)
  assert obs.dtype == np.float32
  assert set(info) >= {'mass', 'friction', 'contact_length', 'is_held',
                       'n_drops'}


# --------------------------------------------- 4. 소스 박스 폭이 바깥폭을 걸친다
def test_source_box_width_straddles_the_gripper_outer_width():
  env = GraspCarry2D()
  gow = env.cfg.gripper_outer_width
  widths = []
  for seed in range(40):
    env.reset(seed=seed)
    widths.append(env.src_box.inner_width)
  # 한쪽만 나오면 항상 깊은/항상 얕은 파지가 되어 설계가 깨진 것이다.
  assert min(widths) < gow, 'no episode forces a shallow grasp'
  assert max(widths) > gow, 'no episode allows a deep grasp'


# --------------------------------------------------------------- 5. 하드 스톱
def test_base_never_descends_past_the_kinematic_limit():
  """아주 낮은 목표 y를 계속 줘도 베이스가 하강 한계를 넘지 않는다.

  블록을 치운 상태로도 검사한다. 블록이 있으면 그리퍼가 블록 위에 얹혀 한계에
  **닿지도 못한 채** 통과할 수 있어(= 무의미한 테스트), 한계가 실제로 구속
  조건이 되는 상황을 만들어야 한다.
  """
  env = GraspCarry2D()
  env.reset(seed=2)
  env.space.remove(env.block_body, env.block_shape)   # 방해물 제거
  mid_x = 0.5 * (env.src_box.right_outer + env.tgt_box.left_outer)
  def press(tx, n):
    for _ in range(n):
      env.step((tx, 10_000.0, 0.0, 0.0))
      base = env.gripper.base
      assert base.position.y <= env.max_descend_y(base.position.x) + 1e-6

  for tx in (env.src_box.center_x, mid_x, env.tgt_box.center_x):
    # 먼저 rim 위로 들어올려 이동한 뒤 누른다. 최대 하향력을 준 채 가로지르면
    # 그리퍼가 rim에 얹혀 마찰(0.5 * ee_force_max)에 걸려 끼고, 아래의
    # "한계에 도달했나" 확인이 이동 도중 상태를 재게 된다(실측: x=140.8에서
    # 속도 0으로 정지). 실제 정책도 들어올린 뒤 이동한다.
    for _ in range(25):
      env.step((tx, 0.0, 0.0, 0.0))
      base = env.gripper.base
      assert base.position.y <= env.max_descend_y(base.position.x) + 1e-6
    press(tx, 25)
    # 한계에 실제로 도달했는지 — 도달 못 하면 위 부등식은 공허하다.
    assert env.gripper.base.position.y == pytest.approx(
        env.max_descend_y(), abs=0.5)

  # 블록이 있는(= 정상) 상황에서도 불변식은 유지된다.
  env.reset(seed=2)
  for _ in range(20):
    env.step((env.src_box.center_x, 10_000.0, 0.0, 0.0))
    base = env.gripper.base
    assert base.position.y <= env.max_descend_y(base.position.x) + 1e-6


# ------------------------------------------------------ 6. 성공 판정(자세 무관)
@pytest.mark.parametrize('angle', [math.pi / 2.0, 0.0])
def test_block_settled_inside_target_box_counts_as_success(angle):
  """타겟 박스 바닥에 내려앉으면 **자세와 무관하게** 성공이다.

  `angle=0`(세워둔 키 큰 블록)이 중요하다: 중심 y가 타겟 박스 rim보다 **위**라
  성공 판정에 블록 중심을 쓰면 실패로 오판된다(실측된 버그).
  """
  env = GraspCarry2D()
  env.reset(seed=1)
  place_block(env, env.tgt_box.center_x, env.tgt_box.inner_floor_y, angle)
  if angle == 0.0:
    assert env.block_body.position.y < env.tgt_box.rim_y, (
        'seed 1 블록이 충분히 크지 않아 중심-y 버그를 잡지 못한다')
  terminated, info = run_until_terminated(env)
  assert terminated and info['outcome'] == 'success'


# --------------------------------------------------------------- 7. 넘어짐
def test_block_laid_down_outside_target_box_is_a_permanent_failure():
  env = GraspCarry2D()
  # 두 박스 사이 빈 바닥이 누운 블록보다 넓은 에피소드를 고른다(소스 박스
  # 내폭이 랜덤이라 시드에 따라 자리가 없을 수 있다).
  for seed in range(40):
    env.reset(seed=seed)
    gap = env.tgt_box.left_outer - env.src_box.right_outer
    if gap > env.block_h + 8.0:
      break
  else:
    pytest.fail('no seed leaves floor room for a lying block')

  center = 0.5 * (env.src_box.right_outer + env.tgt_box.left_outer)
  place_block(env, center, env.cfg.floor_y, math.pi / 2.0)
  terminated, info = run_until_terminated(env)
  assert terminated and info['outcome'] == 'tipped'
  assert env.is_tipped()


# ------------------------------------------------------------ 8. 월드 탈출 방지
def test_block_cannot_escape_the_world():
  """양끝 벽이 없으면 물체가 화면 밖으로 날아간다(실측된 실패 모드).

  박스 안에서 임펄스를 주면 박스 벽에 막혀 월드 벽을 시험하지 못하므로,
  rim보다 높은 공중에 놓고 던져 실제로 월드 끝까지 가게 한다.
  """
  env = GraspCarry2D()
  for sign in (1.0, -1.0):
    env.reset(seed=4)
    body = env.block_body
    body.position = (env.cfg.world_width / 2.0, 100.0)   # 박스 rim 위 공중
    body.velocity = (0.0, 0.0)
    # v = 2 m/s 짜리 수평 임펄스 (물리 substep당 4mm — 벽을 뚫지 않는다)
    body.apply_impulse_at_local_point((sign * 2000.0 * body.mass, 0.0))
    reach = []
    for _ in range(30):
      env.step(hold_action(env))
      x = float(body.position.x)
      assert 0.0 <= x <= env.cfg.world_width, f'block escaped to x={x}'
      reach.append(x)
    # 벽까지 실제로 갔는지 — 안 갔으면 위 단언은 공허하다.
    edge = max(reach) if sign > 0 else min(reach)
    assert abs(edge - (env.cfg.world_width if sign > 0 else 0.0)) < 60.0
