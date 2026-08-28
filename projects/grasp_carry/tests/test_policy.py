"""Tests for the scripted carry policy (phase 3, step 3).

성능(성공률·스텝)은 `src/eval_carry.py`가 재는 것이고, 여기서는 정책의 **계약**만
검증한다: 액션 형식, `allow_regrasp` 스위치, 그리고 비특권 모드가 은닉 물성을
정말로 안 본다는 것.

세 번째가 이 phase에서 가장 중요하다. 비특권 정책이 은닉값을 몰래 참조하면
수집한 데모의 관측-행동 관계가 오염되고(같은 관측인데 은닉값에 따라 행동이
달라진다), 이 환경의 존재 이유인 "관측이 결과를 미결정한다"가 무너진다.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy


def rollout(env, policy, seed, max_steps=None):
  """한 에피소드를 굴리고 (거쳐간 phase 집합, 마지막 info)를 준다."""
  env.reset(seed=seed)
  policy.reset()
  phases, info = set(), env._info()
  for _ in range(max_steps or env.cfg.max_steps):
    action = policy(env)
    phases.add(policy.phase)
    _, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
      break
  return phases, info


# ------------------------------------------------------- 1. 액션 형식
def test_actions_are_four_dimensional_and_finite():
  """모든 스텝의 액션이 `(x, y, theta, grip)` 4차원 유한값이다.

  NaN/inf가 섞이면 pymunk가 조용히 물체를 월드 밖으로 날려버리고 실패가
  엉뚱한 곳에서 드러나므로, 액션 단계에서 잡는다.
  """
  env = GraspCarry2D()
  policy = ScriptedCarryPolicy()
  for seed in (0, 3, 7):
    env.reset(seed=seed)
    policy.reset()
    for _ in range(60):
      action = np.asarray(policy(env), dtype=np.float64)
      assert action.shape == (4,)
      assert np.all(np.isfinite(action)), f'non-finite action {action}'
      # grip은 이진 해석(> 0.5)이므로 값 자체도 [0, 1] 안에 있어야 한다.
      assert 0.0 <= action[3] <= 1.0
      _, _, terminated, truncated, _ = env.step(action)
      if terminated or truncated:
        break


# ------------------------------------------------- 2. allow_regrasp 스위치
def test_allow_regrasp_false_never_enters_the_relocate_phase():
  """`allow_regrasp=False`면 재파지 phase에 진입하지 않는다.

  같은 시드에서 `True`일 때는 실제로 진입하는지도 함께 단언한다 — 그 확인이
  없으면 "애초에 재파지가 필요 없는 시드였다"는 이유로 테스트가 헛돈다.
  """
  env = GraspCarry2D()
  regrasping = ScriptedCarryPolicy(allow_regrasp=True)
  direct = ScriptedCarryPolicy(allow_regrasp=False)

  # 재파지가 실제로 일어나는 시드를 찾는다(소스 박스 내폭이 랜덤이라 시드마다
  # 얕은/깊은 파지가 갈린다).
  for seed in range(12):
    phases, _ = rollout(env, regrasping, seed)
    if 'relocate' in phases:
      break
  else:
    pytest.fail('no seed in 0..11 triggers a regrasp')
  assert regrasping.regrasped and regrasping.n_regrasps >= 1

  phases, _ = rollout(env, direct, seed)
  assert 'relocate' not in phases
  assert not direct.regrasped and direct.n_regrasps == 0


# ------------------------------------------- 3. 비특권 모드는 은닉값을 안 본다
def _blind_info(env):
  """은닉 물성 키를 **제거한** `info`를 주도록 env를 감싼다."""
  original = env._info

  def stripped():
    info = original()
    info.pop('mass', None)
    info.pop('friction', None)
    return info

  env._info = stripped
  return original


def test_unprivileged_policy_never_reads_the_hidden_properties():
  """기본(`privileged=False`) 정책은 `info`의 은닉 물성 없이도 동작한다.

  그리고 `privileged=True`에서는 같은 조작이 `KeyError`를 내야 한다 — 그렇지
  않다면 은닉값을 지우는 조작 자체가 무효라 위 단언이 공허하다.
  """
  env = GraspCarry2D()
  _blind_info(env)

  policy = ScriptedCarryPolicy(privileged=False)
  phases, info = rollout(env, policy, seed=0)
  assert len(phases) > 1, 'policy never advanced past the first phase'
  assert info['outcome'] in ('success', 'tipped', 'timeout', 'running')

  with pytest.raises(KeyError):
    rollout(env, ScriptedCarryPolicy(privileged=True), seed=0)


def _speed_caps(privileged, mass, mu):
  """접촉 길이/지렛대를 격자로 고정한 채 `_speed_cap`이 고르는 리드를 뽑는다."""
  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  env.reset(seed=2)
  env._mass, env._mu = mass, mu       # 은닉값만 갈아끼운다(기하=관측은 그대로)
  policy = ScriptedCarryPolicy(privileged=privileged, config=cfg)
  return [round(policy._speed_cap(env, contact, arm), 9)
          for contact in (8.0, 20.0, 30.0)
          for arm in (5.0, 25.0, 60.0)]


def test_unprivileged_speed_choice_ignores_the_episode_hidden_values():
  """은닉값이 달라도 비특권 정책의 **속도 선택**은 한 비트도 안 바뀐다.

  키 제거(위 테스트)는 "안 읽는다"를 보지만, 은닉값이 정말로 행동을 못 바꾸는지는
  이렇게 직접 재는 편이 확실하다. 특권 모드에서는 반대로 달라져야 한다 —
  그 대조가 없으면 `_speed_cap`이 애초에 상수를 뱉는 경우와 구분되지 않는다.
  """
  light, heavy = (0.05, 0.60), (0.34, 0.26)   # (질량, 마찰) — 유리한 쪽/불리한 쪽
  assert _speed_caps(False, *light) == _speed_caps(False, *heavy)
  assert _speed_caps(True, *light) != _speed_caps(True, *heavy)
