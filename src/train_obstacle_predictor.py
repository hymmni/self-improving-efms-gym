"""장애물 회피 환경(5필드 관측) steps-to-go 예측기 + BC 정책 학습 (phase 3).

파이프라인 (train_sft.py의 구조를 5필드 관측에 맞게 재작성):
  1. 데모 생성: ObstacleAvoidPoint2D + demo_action_pf(노이즈 1.5e-4, 채택값).
     성공 못 한 에피소드(500스텝 상한)는 버림(~1%). 라벨 = time_to_success
     (원본 generate_dataset과 동일하게 arange(len-1,-1,-1)).
     생성 결과는 data/obstacle_demos.pkl로 저장해 재사용(골 입력 ablation 등).
  2. 관측 9차원 concat: cur_pos(2) cur_vel(2) goal_pos(2) obstacle_rel_pos(2)
     obstacle_radius(1). 정규화는 원본 make_normalizers 방식을 확장
     (goal_pos는 원본처럼 cur_pos 통계 공유, 장애물 필드는 자체 통계).
  3. 네트워크: pointmass_core.build_continuous_act_discrete_dist_v0 재사용
     (MLP 256x3, bin_size=1 -> num_bins = max_distance = 420, p99 길이 407 근거).
     bin_size=1이므로 원본 loss의 raw-ttg-as-class-label이 정확히 성립
     (phase-2에서 확인한 잠재 버그 우회와 동일).
  4. 학습 루프: 단일 디바이스 jit (PretrainLearner는 3필드 concat 하드코딩
     + pmap 구조라 재사용하지 않음). Adam 3e-4, minibatch 256, 32768 step
     (train_sft와 동일 하이퍼).
  5. 평가: held-out 전이 MAE/NLL + BC 정책 성공률(100 에피소드).

실행:
  python -m src.train_obstacle_predictor            # 전체 (생성 포함)
  python -m src.train_obstacle_predictor --episodes 10000 --steps 32768
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
from src.obstacle_env import ObstacleAvoidPoint2D, demo_action_pf, demo_action

OBS_FIELDS = ('cur_pos', 'cur_vel', 'goal_pos', 'obstacle_rel_pos',
              'obstacle_radius')
EP_CAP = 500              # 데모 에피소드 스텝 상한 (넘으면 폐기)
MAX_DISTANCE = EP_CAP     # 성공 에피소드는 정의상 EP_CAP 이하 -> 라벨이 bin
NUM_BINS = MAX_DISTANCE   # 범위를 절대 못 넘음. bin_size=1.

DATA_PATH = 'data/obstacle_demos.pkl'
CKPT_PATH = 'checkpoints/obstacle/predictor.pkl'


# ------------------------------------------------------------------ dataset
def generate_demos(num_episodes, seed0=3000, noise_std=1.5e-4, action_fn=None):
  """action_fn(obs)->action. 기본(None)이면 기존 노이즈 PF 컨트롤러.
  action_fn=demo_action(접선점 조준)을 넘기면 노이즈 없는 '깔끔한' 데모가 된다
  (2026-07-20: 노이즈가 태스크 무관 불확실성을 얼마나 만드는지 분리하기 위한
  대조군 — --controller tangent)."""
  if action_fn is None:
    action_fn = lambda obs: demo_action_pf(obs, noise_std=noise_std)
  obs_lists = {k: [] for k in OBS_FIELDS}
  acts, ttgs, ep_ids = [], [], []
  n_discard = 0
  t0 = time.time()
  ep = 0
  attempt = 0
  while ep < num_episodes:
    np.random.seed(seed0 + attempt)
    attempt += 1
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    ep_obs, ep_act = [], []
    step = 0
    while not env.success() and step < EP_CAP:
      act = action_fn(ts.observation)
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
    # From: pointmass_core.generate_dataset — time_to_success 라벨링 방식 동일
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
                n_discard=n_discard, ep_cap=EP_CAP, seed0=seed0),
  )
  print(f'생성 완료: {len(data["action"])} transitions / {num_episodes} eps '
        f'(폐기 {n_discard}, {time.time() - t0:.0f}s)')
  return data


# -------------------------------------------------------------- normalizers
def compute_stats(data):
  obs, act = data['observation'], data['action']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  stats['cur_pos_mean'], stats['cur_pos_std'] = ms(obs['cur_pos'])
  stats['cur_vel_mean'], stats['cur_vel_std'] = ms(obs['cur_vel'])
  stats['obstacle_rel_pos_mean'], stats['obstacle_rel_pos_std'] = \
      ms(obs['obstacle_rel_pos'])
  stats['obstacle_radius_mean'], stats['obstacle_radius_std'] = \
      ms(obs['obstacle_radius'])
  stats['act_mean'], stats['act_std'] = ms(act)
  return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def make_normalizers_obstacle(stats):
  """원본 make_normalizers의 5필드 확장. goal_pos는 원본과 동일하게 cur_pos
  통계를 공유한다(같은 좌표 공간)."""
  def normalize_obs(obs):
    return {
        'cur_pos':
            (obs['cur_pos'] - stats['cur_pos_mean']) / stats['cur_pos_std'],
        'cur_vel':
            (obs['cur_vel'] - stats['cur_vel_mean']) / stats['cur_vel_std'],
        'goal_pos':
            (obs['goal_pos'] - stats['cur_pos_mean']) / stats['cur_pos_std'],
        'obstacle_rel_pos':
            (obs['obstacle_rel_pos'] - stats['obstacle_rel_pos_mean'])
            / stats['obstacle_rel_pos_std'],
        'obstacle_radius':
            (obs['obstacle_radius'] - stats['obstacle_radius_mean'])
            / stats['obstacle_radius_std'],
    }

  def normalize_action(a):
    return (a - stats['act_mean']) / stats['act_std']

  def unnormalize_action(a):
    return a * stats['act_std'] + stats['act_mean']

  return normalize_obs, normalize_action, unnormalize_action


def concat_obs(obs):
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
  ap.add_argument('--controller', choices=['pf', 'tangent'], default='pf',
                  help='pf=노이즈 PF(기존 채택값), tangent=노이즈 없는 접선점'
                       ' 조준(대조군, data/ckpt에 _clean 접미사)')
  args = ap.parse_args()

  suffix = '' if args.controller == 'pf' else '_clean'
  data_path = f'data/obstacle_demos{suffix}.pkl'
  ckpt_path = f'checkpoints/obstacle{suffix}/predictor.pkl'
  action_fn = None if args.controller == 'pf' else (lambda obs: demo_action(obs))

  # ---- 1) 데이터 (있으면 재사용)
  if os.path.exists(data_path):
    print(f'기존 데이터셋 사용: {data_path}')
    with open(data_path, 'rb') as fp:
      data = pickle.load(fp)
  else:
    data = generate_demos(args.episodes, action_fn=action_fn)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as fp:
      pickle.dump(data, fp)
    print(f'저장: {data_path}')
  N = len(data['action'])
  lengths = np.bincount(data['episode_id'])
  print(f'transitions={N}  ep길이 p50={np.percentile(lengths,50):.0f} '
        f'p99={np.percentile(lengths,99):.0f} max={lengths.max()}')
  assert lengths.max() <= MAX_DISTANCE, 'MAX_DISTANCE보다 긴 에피소드 존재'

  # ---- 2) 정규화/분할 (held-out은 에피소드 단위 5%)
  stats = compute_stats(data)
  normalize_obs, normalize_action, unnormalize_action = \
      make_normalizers_obstacle(stats)
  rng = np.random.default_rng(args.seed)
  n_ep = int(data['episode_id'].max()) + 1
  val_eps = set(rng.choice(n_ep, size=max(n_ep // 20, 1), replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))
  act_n = np.asarray(normalize_action(data['action']))
  ttg = data['time_to_success']
  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_act = jnp.asarray(act_n[~val_mask])
  tr_ttg = jnp.asarray(ttg[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_act = jnp.asarray(act_n[val_mask])
  va_ttg = jnp.asarray(ttg[val_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps)')

  # ---- 3) 네트워크 (원본 빌더 재사용, 입력 9차원)
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
    # bin_size=1이므로 raw ttg가 곧 클래스 인덱스 (원본 loss와 동일 형태)
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

  # ---- 4) BC 정책 성공률 평가 (cpu jit — 원본 rollout과 동일 방식)
  def _policy(params, norm_concat, rng):
    preds = nets.network.apply(params, norm_concat)
    act = nets.sample_act(preds.act_dist_params, rng)
    return act
  policy = jax.jit(_policy, backend='cpu')

  succ, ep_lens = 0, []
  key_eval = jax.random.PRNGKey(1234)
  for e in range(args.eval_episodes):
    np.random.seed(90000 + e)
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    step = 0
    while not env.success() and step < EP_CAP:
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None],
                                        ts.observation))
      key_eval, sub = jax.random.split(key_eval)
      act = np.asarray(policy(state.params, np.asarray(concat_obs(norm)), sub))[0]
      ts = env.step(np.asarray(unnormalize_action(act), dtype=np.float32))
      step += 1
    succ += int(env.success())
    ep_lens.append(step)
  success_rate = succ / args.eval_episodes
  mae, nll, vbc = val_metrics(state.params)
  print(f'\n최종: 성공률 {success_rate:.2f} ({args.eval_episodes} eps, '
        f'길이 중앙값 {np.median(ep_lens):.0f})  '
        f'val MAE={float(mae):.2f} NLL={float(nll):.3f}')

  # ---- 5) 체크포인트
  os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
  with open(ckpt_path, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': stats,
        'dc_config': {'min_distance': 0, 'max_distance': MAX_DISTANCE,
                      'num_bins': NUM_BINS},
        'obs_fields': list(OBS_FIELDS),
        'obs_dim': int(tr_obs.shape[-1]),
        'meta': {
            'env': 'ObstacleAvoidPoint2D', 'controller': args.controller,
            'noise_std': (0.0 if args.controller == 'tangent' else 1.5e-4),
            'episodes': args.episodes, 'steps': args.steps,
            'seed': args.seed,
            'val_mae': float(mae), 'val_nll': float(nll),
            'policy_success_rate': success_rate,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {ckpt_path}')


if __name__ == '__main__':
  main()
