"""Reward functions for the variance-reward experiments (spec section 1).

Operates on per-trajectory prediction arrays: `mu[t]` and `sigma2[t]` are the mean
and variance of the steps-to-go distribution the policy predicts at step t
(computed from the categorical predictor output on the same bin values used in
training — see stg_probe). Arrays include the terminal prediction, so a length
(T+1) trajectory yields T stepwise rewards.

  baseline (eq 1.1):  r_t = mu_t - mu_{t+1}
  ours     (eq 1.2):  r_t = alpha*(mu_t - mu_{t+1}) + beta * VARTERM

VARTERM is selected by `variant` (decision item 2.1):
  'change_rate':  (sigma2_t - sigma2_{t+1}) / (sigma2_t + eps)
  'ratio':         1 - sigma2_{t+1} / (sigma2_t + eps)
Both reward variance *reduction*; they are algebraically related but differ in
numerical behavior near sigma2_t -> 0, which is exactly what experiment 2.1/2.2
measure. `eps` (decision item 2.2) guards the division.

CRITICAL (spec section 7): baseline must be reproducible as the beta=0 special
case of `ours`. `stepwise_reward(..., beta=0)` returns exactly alpha*(mu_t-mu_{t+1}),
independent of variant/eps/sigma2.
"""

import numpy as np


def stepwise_reward(mu, sigma2, alpha=1.0, beta=0.0, eps=1e-6,
                    variant='change_rate'):
  """Per-step reward along one trajectory.

  mu, sigma2: (T+1,) arrays. Returns (T,) rewards.
  """
  mu = np.asarray(mu, dtype=np.float64)
  sigma2 = np.asarray(sigma2, dtype=np.float64)
  dmu = mu[:-1] - mu[1:]

  if beta == 0.0:
    # exact baseline (eq 1.1 when alpha=1); no dependence on sigma2/variant/eps
    return (alpha * dmu).astype(np.float32)

  s_t = sigma2[:-1]
  s_tp1 = sigma2[1:]
  if variant == 'change_rate':
    var_term = (s_t - s_tp1) / (s_t + eps)
  elif variant == 'ratio':
    var_term = 1.0 - s_tp1 / (s_t + eps)
  else:
    raise ValueError(f'unknown variant {variant!r}')

  return (alpha * dmu + beta * var_term).astype(np.float32)


def discounted_returns(rews, gamma):
  """Backward discounted accumulation (same as the notebook Stage-2 weights)."""
  rews = np.asarray(rews, dtype=np.float32)
  weights = np.empty_like(rews)
  temp = 0.0
  for i in range(rews.shape[0] - 1, -1, -1):
    temp = rews[i] + gamma * temp
    weights[i] = temp
  return weights


def reward_from_config(mu, sigma2, cfg):
  """Convenience: build stepwise reward + discounted returns from a config dict.

  cfg keys: alpha, beta, eps, variant, gamma.
  """
  rews = stepwise_reward(
      mu, sigma2,
      alpha=cfg.get('alpha', 1.0),
      beta=cfg.get('beta', 0.0),
      eps=cfg.get('eps', 1e-6),
      variant=cfg.get('variant', 'change_rate'))
  return discounted_returns(rews, cfg.get('gamma', 0.9))
