"""Tests for `train_carry_si.py` (phase 4, step 3 — SI-EFM Stage-2 DDPO-SF loop).

env/정책 없이도 검증 가능한 순수 함수 3개만 목표로 한다:
  1. `compute_returns` — 감가 리턴 계산이 R_t = sum gamma^(i-t) r_i를 정확히
     내는지, gamma=1일 때 텔레스코핑으로 d(o_t)-d(o_last)가 되는지.
  2. `make_pair_index_table` — xs 안에서 (x_in, x_out, kk) 대응이 규칙과
     일치하는지, 쌍 개수가 n_steps-1인지.
  3. `compute_step_rewards`의 시그니처에 env의 outcome/terminated 등이 전혀
     들어가지 않는지 (구조적으로 오염 불가능함을 보장).
"""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_carry.scripts.train.train_carry_si import (compute_returns, compute_step_rewards,
                            make_pair_index_table)


# ---------------------------------------------------------- 1. returns 감가
def test_returns_discounting():
  rewards = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
  gamma = 0.9

  R = compute_returns(rewards, gamma)
  assert R.shape == (4,)

  # 직접 계산: R_t = sum_{i>=t} gamma^(i-t) r_i
  T = len(rewards)
  expected = np.zeros(T, dtype=np.float64)
  for t in range(T):
    expected[t] = sum(gamma ** (i - t) * rewards[i] for i in range(t, T))
  np.testing.assert_allclose(R, expected, rtol=1e-5, atol=1e-6)

  # 텔레스코핑: r_t = d_t - d_{t+1}이고 gamma=1이면 R_t = d_t - d_last
  d_vals = np.array([10.0, 7.0, 7.0, 9.0, 3.0], dtype=np.float32)  # T+1=5
  r = compute_step_rewards(d_vals)
  R1 = compute_returns(r, gamma=1.0)
  expected_telescope = d_vals[:-1] - d_vals[-1]
  np.testing.assert_allclose(R1, expected_telescope, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------ 2. pair indexing
def test_pair_indexing():
  n_steps = 10
  i_in, i_out, kk = make_pair_index_table(n_steps)

  assert len(i_in) == len(i_out) == len(kk) == n_steps - 1

  # 가짜 xs: xs[i][...] = i (값이 곧 인덱스). 모양은 (n_steps+1, B, D)라 치고
  # 여기선 스칼라로 단순화해도 인덱스 대응 자체는 값과 무관하게 성립해야 한다.
  xs_values = np.arange(n_steps + 1)

  for idx in range(n_steps - 1):
    expected_kk = n_steps - 1 - idx
    assert i_in[idx] == idx
    assert i_out[idx] == idx + 1
    assert kk[idx] == expected_kk
    # x_in -> x_out 전이가 kk 스텝에 해당한다는 규칙(ddpo.chain_logp와 동일)
    x_in_val = xs_values[i_in[idx]]
    x_out_val = xs_values[i_out[idx]]
    assert x_in_val == idx
    assert x_out_val == idx + 1

  # kk는 n_steps-1부터 1까지 정확히 한 번씩(내림차순), kk=0은 등장하지 않음
  assert list(kk) == list(range(n_steps - 1, 0, -1))
  assert 0 not in kk


# --------------------------------------------------- 3. 보상 계산에 ground truth 없음
def test_no_ground_truth_in_reward():
  sig = inspect.signature(compute_step_rewards)
  param_names = set(sig.parameters.keys())
  forbidden_substrings = ('outcome', 'terminated', 'truncated', 'success',
                          'info', 'done', 'reset')
  for name in param_names:
    lname = name.lower()
    for bad in forbidden_substrings:
      assert bad not in lname, (
          f'compute_step_rewards의 인자 {name!r}가 env ground-truth 신호를 '
          f'암시한다 — 보상은 d 값에서만 계산돼야 한다.')

  # 인자가 d_vals 하나뿐이라는 구조 자체를 확인 (env 관련 정보가 흘러들 여지가 없음)
  assert len(param_names) == 1

  # 동작 확인: 순수하게 d_vals만으로 r_t = d_t - d_{t+1}을 낸다
  d_vals = np.array([5.0, 3.0, 4.0], dtype=np.float32)
  r = compute_step_rewards(d_vals)
  np.testing.assert_allclose(r, np.array([2.0, -1.0], dtype=np.float32))
