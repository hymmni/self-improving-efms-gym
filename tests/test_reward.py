"""Tests for the reward functions (phase 2, step 1)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward import stepwise_reward, discounted_returns, reward_from_config


def test_baseline_equals_expectation_difference():
  mu = np.array([40., 38., 35., 30., 0.], dtype=np.float32)
  sigma2 = np.array([100., 90., 50., 10., 1.], dtype=np.float32)
  r = stepwise_reward(mu, sigma2, alpha=1.0, beta=0.0)
  np.testing.assert_allclose(r, mu[:-1] - mu[1:], rtol=1e-6)


def test_baseline_is_beta_zero_special_case():
  # spec section 7: baseline must be reproducible as beta=0, independent of
  # variant / eps / sigma2 values (even pathological ones).
  mu = np.array([10., 7., 3., 0.], dtype=np.float32)
  for sigma2 in [np.array([0., 0., 0., 0.]), np.array([1e-9, 5., 1e-9, 2.])]:
    r0 = stepwise_reward(mu, sigma2, beta=0.0, variant='change_rate')
    r1 = stepwise_reward(mu, sigma2, beta=0.0, variant='ratio', eps=1.0)
    np.testing.assert_array_equal(r0, r1)
    np.testing.assert_allclose(r0, mu[:-1] - mu[1:], rtol=1e-6)


def test_ours_adds_variance_term():
  mu = np.array([40., 38., 0.], dtype=np.float32)
  sigma2 = np.array([100., 50., 1.], dtype=np.float32)
  base = stepwise_reward(mu, sigma2, beta=0.0)
  ours = stepwise_reward(mu, sigma2, alpha=1.0, beta=1.0, variant='change_rate')
  # variance shrinks (100->50->1) so the variance term is positive -> ours > base
  assert np.all(ours >= base)
  assert np.any(ours > base)


def test_change_rate_and_ratio_relationship():
  # change_rate and (1 - ratio) are algebraically equal in exact arithmetic;
  # with the same eps they should match closely.
  mu = np.array([5., 4., 2., 0.], dtype=np.float32)
  sigma2 = np.array([80., 60., 30., 5.], dtype=np.float32)
  cr = stepwise_reward(mu, sigma2, alpha=0.0, beta=1.0, variant='change_rate', eps=1e-6)
  ra = stepwise_reward(mu, sigma2, alpha=0.0, beta=1.0, variant='ratio', eps=1e-6)
  np.testing.assert_allclose(cr, ra, atol=1e-4)


def test_variance_term_no_nan_when_sigma_zero():
  mu = np.array([3., 2., 1., 0.], dtype=np.float32)
  sigma2 = np.array([0., 0., 0., 0.], dtype=np.float32)
  r = stepwise_reward(mu, sigma2, alpha=1.0, beta=1.0, eps=1e-6,
                      variant='change_rate')
  assert np.all(np.isfinite(r))


def test_discounted_returns_matches_manual():
  rews = np.array([1., 2., 3.], dtype=np.float32)
  gamma = 0.9
  w = discounted_returns(rews, gamma)
  expected = np.array([1 + 0.9 * (2 + 0.9 * 3), 2 + 0.9 * 3, 3.],
                      dtype=np.float32)
  np.testing.assert_allclose(w, expected, rtol=1e-6)


def test_reward_from_config():
  mu = np.array([10., 5., 0.], dtype=np.float32)
  sigma2 = np.array([50., 20., 1.], dtype=np.float32)
  cfg = {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.9}
  w = reward_from_config(mu, sigma2, cfg)
  assert w.shape == (2,) and np.all(np.isfinite(w))
