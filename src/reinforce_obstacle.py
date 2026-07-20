"""장애물 회피 환경(phase 3)에서 REINFORCE 자기개선 — phase 2 src/reinforce.py를
9차원 관측(장애물 필드 포함)에 맞게 이식.

phase 2(웨이포인트/멀티모달 맵)의 결론: baseline 대비 분산 보상("ours")이
견고하게 못 이김. 그런데 이 환경은 (1) 데모가 노이즈로 의도적으로 비효율적이라
자기개선 여지가 실재하고, (2) BC 정책 성공률이 이미 0.96~1.0로 천장에 가까워
"성공률" 대신 "에피소드 길이(효율)"가 더 의미 있는 지표다.

중요: eps 기본값 1e-6은 잘 보정된 예측기(σ²_t≈0인 상태)에서 보상이 폭주하는
버그가 있음이 확인된 바 있다(2026-07-13, 미수정 결정). 이 예측기는 well-
calibrated하므로(터미널 근처 σ²→0 다수 관측) 여기서는 eps를 안전한 값(기본
1.0)으로 올려 그 문제를 피한다.
"""

import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmass_core import build_continuous_act_discrete_dist_v0  # noqa: E402
from src.obstacle_env import ObstacleAvoidPoint2D  # noqa: E402
from src.reward import reward_from_config  # noqa: E402
from src.train_obstacle_predictor import (  # noqa: E402
    OBS_FIELDS, make_normalizers_obstacle)

CKPT_DEFAULT = 'checkpoints/obstacle/predictor.pkl'


def build_from_checkpoint(ckpt_path):
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  c = ck['dc_config']
  bin_vals = np.linspace(c['min_distance'], c['max_distance'], c['num_bins'] + 1,
                        endpoint=True, dtype=np.float32)[:-1]
  obs_dim = ck.get('obs_dim', 9)
  net = build_continuous_act_discrete_dist_v0(
      (256, 256, 256), 2, c['num_bins'], np.ones((4, obs_dim), dtype=np.float32))
  normalize_obs, _, unnormalize_action = make_normalizers_obstacle(ck['norm_stats'])
  return net, ck['params'], normalize_obs, unnormalize_action, bin_vals, ck


def _mu_sigma2(probs, bin_vals):
  mu = float((bin_vals * probs).sum())
  var = float((bin_vals ** 2 * probs).sum() - mu ** 2)
  return mu, max(var, 0.0)


def _concat(normed_obs):
  return jnp.concatenate([normed_obs[f] for f in OBS_FIELDS], axis=-1)


def collect_reinforce_data(net, params, normalize_obs, unnormalize_action,
                           bin_vals, reward_cfg, num_env_steps, key,
                           max_steps=300):
  @jax.jit
  def infer(p, x, rng):
    out = net.network.apply(p, x)
    act = net.sample_act(out.act_dist_params, rng)
    return act, out.dist_to_succ_dist_params.logits

  all_obs, all_act, all_w = [], [], []
  stats = {'success': [], 'len': []}
  total = 0
  while total < num_env_steps:
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    traj_obs, traj_act, traj_mu, traj_s2 = [], [], [], []
    ep_len = 0
    while (not env.success()) and ep_len < max_steps:
      obs = ts.observation
      no = normalize_obs(jax.tree.map(lambda x: x[None], obs))
      x = _concat(no)
      key, sub = jax.random.split(key)
      act, logits = infer(params, x, sub)
      probs = np.asarray(jax.nn.softmax(logits))[0]
      mu, s2 = _mu_sigma2(probs, bin_vals)
      traj_obs.append(np.asarray(x)[0])
      traj_act.append(np.asarray(act)[0])
      traj_mu.append(mu); traj_s2.append(s2)
      unn = unnormalize_action(act)
      ts = env.step(np.asarray(unn, dtype=np.float32)[0])
      ep_len += 1
    obs = ts.observation
    no = normalize_obs(jax.tree.map(lambda x: x[None], obs))
    x = _concat(no)
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
    w = (weight - jnp.mean(weight)) / (jnp.std(weight) + 1e-8)
    return -jnp.mean(w * logp)

  @jax.jit
  def update(params, opt_state, obs, act, weight):
    loss, grads = jax.value_and_grad(loss_fn)(params, obs, act, weight)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
  return update


def train_reinforce(ckpt_path, reward_cfg, num_iters=40, env_steps_per_iter=2048,
                    seed=0, lr=1e-4, max_steps=300):
  net, params, nobs, unact, bin_vals, ck = build_from_checkpoint(ckpt_path)
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)
  update = make_update(net, optimizer)
  key = jax.random.PRNGKey(seed)

  curve = {'iter': [], 'success_rate': [], 'ep_len': [], 'loss': []}
  for it in range(num_iters):
    obs, act, w, stats, key = collect_reinforce_data(
        net, params, nobs, unact, bin_vals, reward_cfg, env_steps_per_iter,
        key, max_steps=max_steps)
    params, opt_state, loss = update(params, opt_state,
                                     jnp.asarray(obs), jnp.asarray(act),
                                     jnp.asarray(w))
    curve['iter'].append(it)
    curve['success_rate'].append(float(np.mean(stats['success'])))
    curve['ep_len'].append(float(np.mean(stats['len'])))
    curve['loss'].append(float(loss))
  return curve
