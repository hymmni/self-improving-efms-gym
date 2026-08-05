r"""GraspCarry2D 액션-조건부 STG 예측기 — 조건 A/B를 예측기로 직접 재현.

`src/train_carry_predictor.py`의 STG 헤드는 관측만 보고 분포를 낸다(액션
비의존). 그래서 "같은 상태에서 빠르게 갈지 느리게 갈지 고르면 분포가
어떻게 달라지는가"(`calibrate_carry.py`의 조건 A/B)를 직접 물어볼 수
없었다. 이 스크립트는 `concat(관측, 액션)`을 입력으로 받는 범주형 분포
하나로 **성공확률과 성공 시 스텝 분포를 동시에** 낸다:

  마지막 bin(=max_steps)을 "실패" 클래스로 쓴다. 성공 에피소드의 스텝은
  기존처럼 `L-1-t`로, 실패 에피소드의 모든 스텝은 `max_steps`로 라벨링돼
  있다(`collect_carry_demos.py --keep-failures`).

    P(성공) = 1 - softmax(logits)[..., 실패bin]
    E[스텝|성공] = 실패bin 제외하고 재정규화한 뒤 기댓값
    q0.8[스텝|성공] = 같은 재정규화 분포의 0.8분위수

데이터: `collect_carry_demos.py --keep-failures` → `data/grasp_carry_demos_v3.pkl`

실행:
  python -m src.train_carry_qstg --data data/grasp_carry_demos_v3.pkl
"""

import argparse
import os
import pickle
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

OBS_FIELDS = ('frame',)


# -------------------------------------------------------------- normalizers
def compute_stats(data):
  obs, act = data['observation'], data['action']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  for f in OBS_FIELDS:
    stats[f'{f}_mean'], stats[f'{f}_std'] = ms(obs[f])
  stats['act_mean'], stats['act_std'] = ms(act)
  return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def make_normalizers(stats):
  def normalize_obs(obs):
    return {f: (obs[f] - stats[f'{f}_mean']) / stats[f'{f}_std']
            for f in OBS_FIELDS}

  def normalize_action(a):
    return (a - stats['act_mean']) / stats['act_std']

  return normalize_obs, normalize_action


def concat_obs(obs):
  return jnp.concatenate([jnp.asarray(obs[f]) for f in OBS_FIELDS], axis=-1)


# ------------------------------------------------------------------ network
def build_qstg_net(layer_sizes, obs_act_dim, num_bins):
  """concat(정규화된 관측, 정규화된 액션) -> 범주형 logits(마지막=실패bin)."""
  def _net(x):
    h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                    activate_final=True)(x)
    return hk.Linear(num_bins, with_bias=False)(h)

  tn = hk.without_apply_rng(hk.transform(_net))

  def init(rng):
    return tn.init(rng, jnp.zeros((2, obs_act_dim), jnp.float32))

  return tn.apply, init


def split_success_fail(logits, fail_bin):
  """logits(..., num_bins) -> (p_succ, 성공 bin만 재정규화한 확률(..., fail_bin))."""
  probs = jax.nn.softmax(logits, axis=-1)
  p_fail = probs[..., fail_bin]
  p_succ = 1.0 - p_fail
  succ_probs = probs[..., :fail_bin] / jnp.maximum(p_succ[..., None], 1e-6)
  return p_succ, succ_probs


def succ_mean_quantile(succ_probs, bin_vals, q=0.8):
  mean = jnp.sum(succ_probs * bin_vals[None, :], axis=-1)
  cdf = jnp.cumsum(succ_probs, axis=-1)
  # cdf가 q를 처음 넘는 bin의 값 — 이산 분위수.
  idx = jnp.argmax(cdf >= q, axis=-1)
  quant = bin_vals[idx]
  return mean, quant


def succ_cvar(succ_probs, bin_vals, alpha=0.8):
  """CVaR_alpha — 최악 (1-alpha) 구간(꼬리)만 재정규화한 기댓값.

  `jnp.argmax(cdf>=q)`(분위수)는 계단함수라 그래디언트가 안 흐른다. 여기서는
  각 bin이 [alpha, 1] 구간과 겹치는 확률질량만 `jnp.clip`으로 뽑아 쓴다 —
  clip은 미분 가능(경계에서 subgradient)하므로 액터를 역전파로 학습시키는
  용도로 쓸 수 있다.
  """
  cdf = jnp.cumsum(succ_probs, axis=-1)
  cdf_prev = cdf - succ_probs
  overlap = jnp.clip(cdf, alpha, 1.0) - jnp.clip(cdf_prev, alpha, 1.0)
  mass = jnp.sum(overlap, axis=-1)
  cvar = jnp.sum(overlap * bin_vals[None, :], axis=-1) / jnp.maximum(mass, 1e-6)
  return cvar


class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--steps', type=int, default=16384)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--warmup', type=int, default=500)
  ap.add_argument('--eval-every', type=int, default=500)
  ap.add_argument('--patience', type=int, default=12)
  ap.add_argument('--weight-decay', type=float, default=1e-4)
  ap.add_argument('--no-early-stop', action='store_true')
  ap.add_argument('--success-only', action='store_true',
                  help=('실패 트랜지션을 학습에서 제외한다 — STG 예측기를 '
                        '성공 데모만으로 학습하는 기존 관례(SI-EFM, '
                        'dp_policy.collect_rollouts)를 같은 아키텍처로 재현하는 '
                        '비교군이다. 이렇게 학습하면 실패 bin이 데이터에 아예 '
                        '없으므로 P(성공)이 구조적으로 ~1로 붕괴하고, 모델은 '
                        '"성공한다고 치면 몇 스텝"만 말할 수 있게 된다. '
                        '검증셋은 (공정 비교를 위해) 실패 포함 그대로 둔다.'))
  ap.add_argument('--tag', default='')
  args = ap.parse_args()

  ckpt_path = f'checkpoints/grasp_carry_qstg{args.tag}/predictor.pkl'

  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  assert data['meta'].get('keep_failures'), (
      '이 스크립트는 --keep-failures로 모은 데이터가 필요하다 '
      '(collect_carry_demos.py --keep-failures)')
  N = len(data['action'])
  max_steps = int(data['meta']['max_steps'])
  fail_bin = int(data['meta']['failure_bin'])
  NUM_BINS = fail_bin + 1                    # 0..max_steps-1 성공 + max_steps 실패
  print(f'transitions={N}  에피소드={len(np.unique(data["episode_id"]))}  '
        f'성공율={data["is_success"].mean():.1%}  '
        f'outcomes={data["meta"]["outcomes"]}')

  stats = compute_stats(data)
  normalize_obs, normalize_action = make_normalizers(stats)
  rng = np.random.default_rng(args.seed)
  ep_ids = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids, size=max(len(ep_ids) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))
  act_n = np.asarray(normalize_action(data['action']))
  x_all = np.concatenate([obs_c, act_n], axis=-1)
  ttg = data['time_to_success']
  is_succ = data['is_success']

  # 학습 마스크에서만 실패를 뺀다. 검증셋은 양쪽 모델이 **같은** 것을 보도록
  # 실패를 포함한 채로 둔다 — 그래야 "성공만으로 배운 모델이 실패를 못
  # 맞힌다"가 같은 척도 위에서 드러난다.
  tr_mask = ~val_mask
  if args.success_only:
    tr_mask = tr_mask & is_succ
  tr_x = jnp.asarray(x_all[tr_mask]); tr_ttg = jnp.asarray(ttg[tr_mask])
  va_x = jnp.asarray(x_all[val_mask]); va_ttg = jnp.asarray(ttg[val_mask])
  va_succ = jnp.asarray(is_succ[val_mask])
  print(f'train {tr_x.shape[0]} / val {va_x.shape[0]} transitions '
        f'(val {len(val_eps)} eps, 입력 {tr_x.shape[-1]}차원 = 관측+액션)'
        + ('  [success-only 학습]' if args.success_only else ''))

  XDIM = int(tr_x.shape[-1])
  apply_fn, init_fn = build_qstg_net((256, 256, 256), XDIM, NUM_BINS)
  bin_vals = jnp.arange(NUM_BINS, dtype=jnp.float32)   # bin_size=1

  sched_lr = optax.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup,
      decay_steps=max(args.steps, args.warmup + 1), end_value=args.lr * 0.05)
  optimizer = optax.adamw(sched_lr, b1=0.95, b2=0.999,
                          weight_decay=args.weight_decay)
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = init_fn(sub)
  state = TrainState(params, optimizer.init(params))

  def loss_fn(p, bx, bt):
    logits = apply_fn(p, bx)
    return -jnp.mean(tfd.Categorical(logits=logits).log_prob(bt))

  @jax.jit
  def train_step(state, key):
    idx = jax.random.randint(key, (args.batch,), 0, tr_x.shape[0])
    l, g = jax.value_and_grad(loss_fn)(state.params, tr_x[idx], tr_ttg[idx])
    updates, opt_state = optimizer.update(g, state.opt_state, state.params)
    return TrainState(optax.apply_updates(state.params, updates), opt_state), l

  @jax.jit
  def val_metrics(p):
    logits = apply_fn(p, va_x)
    nll = -jnp.mean(tfd.Categorical(logits=logits).log_prob(va_ttg))
    p_succ, succ_probs = split_success_fail(logits, fail_bin)
    mean, _ = succ_mean_quantile(succ_probs, bin_vals[:fail_bin])
    # 성공 확률 예측 정확도 — 결정 임계 0.5
    succ_acc = jnp.mean((p_succ > 0.5) == va_succ)
    # 성공 트랜지션에서만 MAE(실패 스텝의 "몇 스텝"은 정의가 없다)
    succ_mask = va_succ
    mae = jnp.sum(jnp.where(succ_mask, jnp.abs(mean - va_ttg), 0.0)) / \
        jnp.maximum(jnp.sum(succ_mask), 1.0)
    return nll, succ_acc, mae

  t0 = time.time()
  best = dict(loss=np.inf, step=0, params=jax.device_get(state.params),
              nll=np.nan, succ_acc=np.nan, mae=np.nan)
  since = 0
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l = train_step(state, sub)
    if step % args.eval_every == 0 or step == 1:
      nll, succ_acc, mae = val_metrics(state.params)
      vloss = float(nll)
      mark = ''
      if vloss < best['loss'] - 1e-4:
        best = dict(loss=vloss, step=step, params=jax.device_get(state.params),
                    nll=float(nll), succ_acc=float(succ_acc), mae=float(mae))
        since = 0; mark = '  *best'
      else:
        since += 1
      print(f'step {step:6d}  train={float(l):.3f}  '
            f'val: NLL={float(nll):.3f} 성공분류정확도={float(succ_acc):.3f} '
            f'MAE(성공만)={float(mae):.2f}  ({time.time()-t0:.0f}s){mark}',
            flush=True)
      if (not args.no_early_stop) and since >= args.patience:
        print(f'조기 종료: {args.patience}회 개선 없음 (best step={best["step"]})')
        break
  if not args.no_early_stop:
    state = TrainState(jax.tree.map(jnp.asarray, best['params']), state.opt_state)
  else:
    best['step'] = args.steps
  print(f'\n[최적] step={best["step"]}  val NLL={best["nll"]:.3f}  '
        f'성공분류정확도={best["succ_acc"]:.3f}  MAE(성공만)={best["mae"]:.2f}')

  os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
  with open(ckpt_path, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': stats,
        'num_bins': NUM_BINS,
        'fail_bin': fail_bin,
        'max_steps': max_steps,
        'obs_fields': list(OBS_FIELDS),
        'frame_fields': data['meta']['frame_fields'],
        'obs_history': data['meta']['obs_history'],
        'action_fields': data['meta']['action_fields'],
        'meta': {
            'env': 'grasp_carry/GraspCarry2D (action-conditioned)',
            'data': args.data, 'steps': args.steps, 'seed': args.seed,
            'success_only': bool(args.success_only),
            'best_step': int(best['step']), 'weight_decay': args.weight_decay,
            'val_nll': float(best['nll']), 'val_succ_acc': float(best['succ_acc']),
            'val_mae': float(best['mae']),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {ckpt_path}')


if __name__ == '__main__':
  main()
