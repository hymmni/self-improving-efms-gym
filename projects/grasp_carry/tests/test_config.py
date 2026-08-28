"""Tests for GraspCarry2D config (phase 3, step 0)."""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.config import CarryConfig


def test_is_dataclass_with_all_defaults():
  assert dataclasses.is_dataclass(CarryConfig)
  for f in dataclasses.fields(CarryConfig):
    assert f.default is not dataclasses.MISSING or \
        f.default_factory is not dataclasses.MISSING, (
            f'field {f.name!r} has no default')


def test_unit_consistency():
  cfg = CarryConfig()
  assert cfg.grip_force == 12_000.0   # mN = 12 N
  assert cfg.gravity == 9810.0        # mm/s^2 = 9.81 m/s^2


def test_max_hold_mass_worst_friction_can_still_lift_heaviest_object():
  cfg = CarryConfig()
  mass_upper = cfg.object_mass_range[1]
  # 최악 마찰(범위 하한)에서도 가장 무거운(가장 채워진) 물체를 들 수 있어야
  # 태스크가 성립한다 — 그렇지 않으면 미끄러운 물체는 아예 들 수 없다.
  assert cfg.max_hold_mass(cfg.object_friction_range[0]) > mass_upper


def test_max_hold_mass_best_friction_has_ample_margin():
  cfg = CarryConfig()
  mass_upper = cfg.object_mass_range[1]
  assert cfg.max_hold_mass(cfg.object_friction_range[1]) > 2.0 * mass_upper


def test_src_box_width_range_straddles_gripper_outer_width():
  cfg = CarryConfig()
  lo, hi = cfg.src_box_width_range
  gow = cfg.gripper_outer_width
  # 범위가 그리퍼 바깥폭을 걸치지 않으면 항상 얕게 또는 항상 깊게만 잡혀
  # 설계 의도(파지 깊이의 에피소드 간 다양성)가 깨진다.
  assert lo < gow < hi


def test_tip_torque_capacity_scales_with_contact_length():
  cfg = CarryConfig()
  mu = 0.5
  short = cfg.tip_torque_capacity(contact_len=5.0, mu=mu)
  long = cfg.tip_torque_capacity(contact_len=cfg.pad_length, mu=mu)
  assert short < long
  assert short > 0.0


def test_ee_force_max_is_positive_and_derived():
  cfg = CarryConfig()
  expected = (cfg.assembly_mass + cfg.payload_mass) * (
      cfg.gravity + cfg.max_accel)
  assert cfg.ee_force_max == expected
  assert cfg.ee_force_max > 0.0
