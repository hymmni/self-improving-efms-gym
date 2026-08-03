"""Minimal REINFORCE self-improvement with a pluggable reward (E2/E3).

Starts from an SFT checkpoint on the multimodal map and improves the policy
(action head) with REINFORCE. The reward per step is computed from the predictor
(distance head) distribution via src.reward — baseline (eq 1.1) or ours (eq 1.2).

Because the REINFORCE loss -mean(weight * log pi_action(a|s)) only has gradients
w.r.t. the action-head parameters, the distance head (the predictor) is frozen
automatically. So mu/sigma2 come from a fixed predictor throughout, which makes
the baseline-vs-ours comparison clean (only the reward differs).
"""

import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmass_core import (  # noqa: E402
    Point2D, build_continuous_act_discrete_dist_v0,
    build_discrete_distance_converter, make_normalizers)
from src.multimodal_env import MultiModalPoint2D  # noqa: E402
from src.reward import reward_from_config  # noqa: E402


def make_env_fn(map_name, jitter=0.08):
  if map_name == 'standard':
    return lambda: Point2D()
  return lambda: MultiModalPoint2D(jitter=jitter)


def build_from_checkpoint(ckpt_path):
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  c = ck['dc_config']
  dc = build_discrete_distance_converter(c['min_distance'], c['max_distance'],
                                         c['num_bins'])
  bin_vals = np.linspace(c['min_distance'], c['max_distance'], c['num_bins'] + 1,
                         endpoint=True, dtype=np.float32)[:-1]
  net = build_continuous_act_discrete_dist_v0(
      (256, 256, 256), 2, c['num_bins'], np.ones((4, 6), dtype=np.float32))
  normalize_obs, _, unnormalize_action = make_normalizers(ck['norm_stats'])
  return net, ck['params'], normalize_obs, unnormalize_action, dc, bin_vals, ck


def _mu_sigma2(probs, bin_vals):
  mu = float((bin_vals * probs).sum())
  var = float((bin_vals ** 2 * probs).sum() - mu ** 2)
  return mu, max(var, 0.0)


def collect_reinforce_data(net, params, normalize_obs, unnormalize_action,
                           bin_vals, reward_cfg, num_env_steps, key,
                           env_fn=None, jitter=0.08, max_steps=140):
  """Roll out episodes with the current policy; compute per-step reward and
  discounted returns via reward_from_config. Returns (obs, action, weight)
  arrays + episode stats."""

  @jax.jit
  def infer(p, x, rng):
    out = net.network.apply(p, x)
    act = net.sample_act(out.act_dist_params, rng)
    return act, out.dist_to_succ_dist_params.logits

  all_obs, all_act, all_w = [], [], []
  stats = {'success': [], 'len': []}
  total = 0
  env = env_fn() if env_fn is not None else MultiModalPoint2D(jitter=jitter)
  while total < num_env_steps:
    ts = env.reset()
    traj_obs, traj_act, traj_mu, traj_s2 = [], [], [], []
    ep_len = 0
    while (not env.success()) and ep_len < max_steps:
      obs = ts.observation
      no = normalize_obs(jax.tree.map(lambda x: x[None], obs))
      x = jnp.concatenate([no['cur_pos'], no['cur_vel'], no['goal_pos']], axis=-1)
      key, sub = jax.random.split(key)
      act, logits = infer(params, x, sub)
      probs = np.asarray(jax.nn.softmax(logits))[0]
      mu, s2 = _mu_sigma2(probs, bin_vals)
      traj_obs.append(np.asarray(x)[0])
      traj_act.append(np.asarray(act)[0])
      traj_mu.append(mu); traj_s2.append(s2)
      unn = unnormalize_action(act)
      ts = env.step(np.asarray(unn)[0])
      ep_len += 1
    # terminal prediction (for mu_{t+1} of the last step)
    obs = ts.observation
    no = normalize_obs(jax.tree.map(lambda x: x[None], obs))
    x = jnp.concatenate([no['cur_pos'], no['cur_vel'], no['goal_pos']], axis=-1)
    key, sub = jax.random.split(key)
    _, logits = infer(params, x, sub)
    probs = np.asarray(jax.nn.softmax(logits))[0]
    mu, s2 = _mu_sigma2(probs, bin_vals)
    traj_mu.append(mu); traj_s2.append(s2)

    if ep_len < 1:
      continue
    weights = reward_from_config(np.array(traj_mu), np.array(traj_s2), reward_cfg)
    all_obs.append(np.array(traj_obs, dtype=np.float32))
    all_act.append(np.array(traj_act, dtype=np.float32))
    all_w.append(weights.astype(np.float32))
    stats['success'].append(bool(env.success()))
    stats['len'].append(ep_len)
    total += ep_len

  obs = np.concatenate(all_obs); act = np.concatenate(all_act)
  w = np.concatenate(all_w)
  stats['success'] = np.array(stats['success'], dtype=np.float32)
  stats['len'] = np.array(stats['len'], dtype=np.float32)
  return obs, act, w, stats, key


def make_update(net, optimizer):
  def loss_fn(params, obs, act, weight):
    out = net.network.apply(params, obs)
    logp = net.act_log_prob(out.act_dist_params, act)
    # standardize weights for a stable policy gradient (advantage-style)
    w = (weight - jnp.mean(weight)) / (jnp.std(weight) + 1e-8)
    return -jnp.mean(w * logp)

  @jax.jit
  def update(params, opt_state, obs, act, weight):
    loss, grads = jax.value_and_grad(loss_fn)(params, obs, act, weight)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
  return update


# Action-head parameter modules (mlp/linear/linear_1) vs distance-head
# (mlp_1/linear_2). Built action-first in build_continuous_act_discrete_dist_v0.
_ACTION_HEAD_KEYS = ('mlp/', 'linear', 'linear_1')


def degrade_policy(params, noise_scale, seed):
  """Add Gaussian noise to the action-head params only (leaves the predictor /
  distance head intact), producing a controlled sub-optimal starting policy for
  E2/E3 so REINFORCE has clear room to improve."""
  if noise_scale <= 0:
    return params
  rng = np.random.default_rng(seed)
  new = {}
  for mod, sub in params.items():
    is_action = (mod in ('linear', 'linear_1')) or mod.startswith('mlp/')
    if is_action:
      new[mod] = {k: (np.asarray(v) +
                      rng.normal(0, noise_scale, size=np.asarray(v).shape)
                      ).astype(np.float32) for k, v in sub.items()}
    else:
      new[mod] = sub
  return new


def train_reinforce(ckpt_path, reward_cfg, num_iters=40, env_steps_per_iter=2048,
                    seed=0, lr=1e-4, jitter=0.08, init_noise=0.0,
                    map_name='multimodal'):
  net, params, nobs, unact, dc, bin_vals, ck = build_from_checkpoint(ckpt_path)
  params = degrade_policy(params, init_noise, seed)
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)
  update = make_update(net, optimizer)
  key = jax.random.PRNGKey(seed)
  env_fn = make_env_fn(map_name, jitter)

  curve = {'iter': [], 'success_rate': [], 'ep_len': [], 'loss': []}
  for it in range(num_iters):
    obs, act, w, stats, key = collect_reinforce_data(
        net, params, nobs, unact, bin_vals, reward_cfg, env_steps_per_iter, key,
        env_fn=env_fn, jitter=jitter)
    params, opt_state, loss = update(params, opt_state,
                                     jnp.asarray(obs), jnp.asarray(act),
                                     jnp.asarray(w))
    curve['iter'].append(it)
    curve['success_rate'].append(float(np.mean(stats['success'])))
    curve['ep_len'].append(float(np.mean(stats['len'])))
    curve['loss'].append(float(loss))
  return curve
