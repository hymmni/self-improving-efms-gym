"""2-장애물 환경 steps-to-go 예측기 + BC 정책 학습.

train_obstacle_predictor.py(1-장애물)의 재현성 있는 구조를 그대로 따르되
환경/컨트롤러/경로만 교체한다. compute_stats/make_normalizers_obstacle/
concat_obs는 필드명 기반으로 동작해 차원(2 vs 4)에 무관하게 그대로 재사용
가능하므로 import해서 쓴다 (1-장애물 코드는 건드리지 않음, additive).

관측 12차원: cur_pos(2) cur_vel(2) goal_pos(2) obstacle_rel_pos(4, 가까운
장애물 순) obstacle_radius(2).

실행:
  python -m src.train_two_obstacle_predictor
"""

import argparse
import os
import pickle
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from pointmass_core import (
    build_continuous_act_discrete_dist_v0,
    build_discrete_distance_converter,
)
from src.obstacle_env import TwoObstacleAvoidPoint2D, demo_action_pf_two
from src.train_obstacle_predictor import (
    OBS_FIELDS, compute_stats, make_normalizers_obstacle, concat_obs,
)

EP_CAP = 500
MAX_DISTANCE = EP_CAP
NUM_BINS = MAX_DISTANCE

DATA_PATH = 'data/two_obstacle_demos.pkl'
CKPT_PATH = 'checkpoints/two_obstacle/predictor.pkl'


# ------------------------------------------------------------------ dataset
def generate_demos(num_episodes, seed0=4000, noise_std=1.5e-4):
  obs_lists = {k: [] for k in OBS_FIELDS}
  acts, ttgs, ep_ids = [], [], []
  n_discard = 0
  t0 = time.time()
  ep = 0
  attempt = 0
  while ep < num_episodes:
    np.random.seed(seed0 + attempt)
    attempt += 1
    env = TwoObstacleAvoidPoint2D()
    ts = env.reset()
    ep_obs, ep_act = [], []
    step = 0
    while not env.success() and step < EP_CAP:
      act = demo_action_pf_two(ts.observation, noise_std=noise_std)
      ep_obs.append(ts.observation)
      ep_act.append(np.asarray(act, dtype=np.float32))
      ts = env.step(act)
      step += 1
    if not env.success() or len(ep_obs) < 10:
      n_discard += 1
      continue
    n = len(ep_obs)
    for k in OBS_FIELDS:
      obs_lists[k] += [o[k] for o in ep_obs]
    acts += ep_act
    ttgs.append(np.arange(n - 1, -1, -1, dtype=np.float32))
    ep_ids.append(np.full(n, ep, dtype=np.int32))
    ep += 1
    if ep % 500 == 0:
      print(f'  {ep}/{num_episodes} episodes '
            f'({time.time() - t0:.0f}s, 폐기 {n_discard})', flush=True)

  data = dict(
      observation={k: np.stack(v).astype(np.float32)
                   for k, v in obs_lists.items()},
      action=np.stack(acts).astype(np.float32),
      time_to_success=np.concatenate(ttgs),
      episode_id=np.concatenate(ep_ids),
      meta=dict(num_episodes=num_episodes, noise_std=noise_std,
                n_discard=n_discard, ep_cap=EP_CAP, seed0=seed0,
                env='TwoObstacleAvoidPoint2D'),
  )
  print(f'생성 완료: {len(data["action"])} transitions / {num_episodes} eps '
        f'(폐기 {n_discard}, {time.time() - t0:.0f}s)')
  return data


def concat_obs2(obs):
  return jnp.concatenate([jnp.asarray(obs[k]) for k in OBS_FIELDS], axis=-1)


# ------------------------------------------------------------------ training
class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=10000)
  ap.add_argument('--steps', type=int, default=32768)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval-episodes', type=int, default=100)
  args = ap.parse_args()

  if os.path.exists(DATA_PATH):
    print(f'기존 데이터셋 사용: {DATA_PATH}')
    with open(DATA_PATH, 'rb') as fp:
      data = pickle.load(fp)
  else:
    data = generate_demos(args.episodes)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'wb') as fp:
      pickle.dump(data, fp)
    print(f'저장: {DATA_PATH}')
  N = len(data['action'])
  lengths = np.bincount(data['episode_id'])
  print(f'transitions={N}  ep길이 p50={np.percentile(lengths,50):.0f} '
        f'p99={np.percentile(lengths,99):.0f} max={lengths.max()}')
  assert lengths.max() <= MAX_DISTANCE, 'MAX_DISTANCE보다 긴 에피소드 존재'

  stats = compute_stats(data)
  normalize_obs, normalize_action, unnormalize_action = \
      make_normalizers_obstacle(stats)
  rng = np.random.default_rng(args.seed)
  n_ep = int(data['episode_id'].max()) + 1
  val_eps = set(rng.choice(n_ep, size=max(n_ep // 20, 1), replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs2(normalize_obs(data['observation'])))
  act_n = np.asarray(normalize_action(data['action']))
  ttg = data['time_to_success']
  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_act = jnp.asarray(act_n[~val_mask])
  tr_ttg = jnp.asarray(ttg[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_act = jnp.asarray(act_n[val_mask])
  va_ttg = jnp.asarray(ttg[val_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps, obs_dim={tr_obs.shape[-1]})')

  nets = build_continuous_act_discrete_dist_v0(
      (256, 256, 256), 2, NUM_BINS,
      np.ones((4, tr_obs.shape[-1]), dtype=np.float32))
  dc = build_discrete_distance_converter(0, MAX_DISTANCE, NUM_BINS)
  bin_vals = np.linspace(0, MAX_DISTANCE, NUM_BINS + 1,
                         endpoint=True, dtype=np.float32)[:-1]
  optimizer = optax.adam(args.lr)

  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = nets.network.init(sub)
  state = TrainState(params, optimizer.init(params))

  def loss_fn(params, bo, ba, bt):
    preds = nets.network.apply(params, bo)
    bc = -jnp.mean(nets.act_log_prob(preds.act_dist_params, ba))
    dl = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, bt))
    return bc + dl, (bc, dl)

  @jax.jit
  def train_step(state, key):
    idx = jax.random.randint(key, (args.batch,), 0, tr_obs.shape[0])
    (l, (bc, dl)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, tr_obs[idx], tr_act[idx], tr_ttg[idx])
    updates, opt_state = optimizer.update(g, state.opt_state)
    return (TrainState(optax.apply_updates(state.params, updates), opt_state),
            l, bc, dl)

  @jax.jit
  def val_metrics(params):
    preds = nets.network.apply(params, va_obs)
    logits = preds.dist_to_succ_dist_params.logits
    exp = jnp.sum(jax.nn.softmax(logits) * bin_vals[None, :], axis=-1)
    mae = jnp.mean(jnp.abs(exp - va_ttg))
    nll = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, va_ttg))
    bc = -jnp.mean(nets.act_log_prob(preds.act_dist_params, va_act))
    return mae, nll, bc

  t0 = time.time()
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l, bc, dl = train_step(state, sub)
    if step % 4096 == 0 or step == 1:
      mae, nll, vbc = val_metrics(state.params)
      print(f'step {step:6d}  loss={float(l):.3f} '
            f'(bc={float(bc):.3f} dist={float(dl):.3f})  '
            f'val: MAE={float(mae):.2f} NLL={float(nll):.3f} '
            f'bc={float(vbc):.3f}  ({time.time() - t0:.0f}s)', flush=True)

  def _policy(params, norm_concat, rng):
    preds = nets.network.apply(params, norm_concat)
    act = nets.sample_act(preds.act_dist_params, rng)
    return act
  policy = jax.jit(_policy, backend='cpu')

  succ, ep_lens = 0, []
  key_eval = jax.random.PRNGKey(1234)
  for e in range(args.eval_episodes):
    np.random.seed(90000 + e)
    env = TwoObstacleAvoidPoint2D()
    ts = env.reset()
    step = 0
    while not env.success() and step < EP_CAP:
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None],
                                        ts.observation))
      key_eval, sub = jax.random.split(key_eval)
      act = np.asarray(policy(state.params, np.asarray(concat_obs2(norm)),
                              sub))[0]
      ts = env.step(np.asarray(unnormalize_action(act), dtype=np.float32))
      step += 1
    succ += int(env.success())
    ep_lens.append(step)
  success_rate = succ / args.eval_episodes
  mae, nll, vbc = val_metrics(state.params)
  print(f'\n최종: 성공률 {success_rate:.2f} ({args.eval_episodes} eps, '
        f'길이 중앙값 {np.median(ep_lens):.0f})  '
        f'val MAE={float(mae):.2f} NLL={float(nll):.3f}')

  os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
  with open(CKPT_PATH, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': stats,
        'dc_config': {'min_distance': 0, 'max_distance': MAX_DISTANCE,
                      'num_bins': NUM_BINS},
        'obs_fields': list(OBS_FIELDS),
        'obs_dim': int(tr_obs.shape[-1]),
        'meta': {
            'env': 'TwoObstacleAvoidPoint2D', 'noise_std': 1.5e-4,
            'episodes': args.episodes, 'steps': args.steps,
            'seed': args.seed,
            'val_mae': float(mae), 'val_nll': float(nll),
            'policy_success_rate': success_rate,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {CKPT_PATH}')


if __name__ == '__main__':
  main()
