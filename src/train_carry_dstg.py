r"""GraspCarry2D 관측-only STG 예측기 d(o,g) := E[steps-to-go | o, g] (phase 4, step 1).

논문(SI-EFM) 식(1)의 거리 함수는 관측과 목표만 입력받는다 — 액션은 입력이
아니다. 이 스크립트는 그 `d`를 두 가지 버전으로 학습한다:

  succ: 성공 에피소드만 학습(논문 원안. Stage-1 데모가 성공 시연이므로 구조적
        으로 실패를 표현할 수 없다). num_bins=200 (0..199).
  fail (--include-failures): 실패 에피소드까지 포함. 마지막 bin(=200)을
        "실패" 클래스로 따로 두어 d가 "이 상태는 실패할 확률이 높다"를
        표현하게 한다. num_bins=201 (0..199 성공 + 200 실패).

두 버전은 데이터 소스·아키텍처·하이퍼파라미터를 전부 동일하게 맞춘다 —
나중에 나오는 정책 성능 차이를 "실패를 아는가" 한 가지 요인으로 돌리기 위함.

정규화기·데이터 로딩·val 분할 방식은 `src/train_carry_predictor.py`를,
실패 bin 라벨링 규칙은 `src/train_carry_qstg.py`(`split_success_fail`)를
따른다. 아키텍처는 `src/diffusion_act.py`의 STG 헤드
(`hk.nets.MLP(...) -> hk.Linear(num_bins, with_bias=False)`)와 동일하다.

데이터: `data/grasp_carry_demos_v3.pkl` (`collect_carry_demos.py --keep-failures`)

실행:
  python -m src.train_carry_dstg --data data/grasp_carry_demos_v3.pkl \
      --out checkpoints/grasp_carry_dstg_succ/predictor.pkl
  python -m src.train_carry_dstg --data data/grasp_carry_demos_v3.pkl \
      --include-failures --out checkpoints/grasp_carry_dstg_fail/predictor.pkl
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
DEFAULT_LAYER_SIZES = (256, 256, 256)


# -------------------------------------------------------------- normalizers
# From: src/train_carry_predictor.py (compute_stats / make_normalizers) —
# 관측 정규화만 필요하므로 액션 통계는 뺐다.
def compute_stats(data):
  obs = data['observation']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  for f in OBS_FIELDS:
    stats[f'{f}_mean'], stats[f'{f}_std'] = ms(obs[f])
  return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def make_normalizers(stats):
  def normalize_obs(obs):
    return {f: (obs[f] - stats[f'{f}_mean']) / stats[f'{f}_std']
            for f in OBS_FIELDS}
  return normalize_obs


def concat_obs(obs):
  return jnp.concatenate([jnp.asarray(obs[f]) for f in OBS_FIELDS], axis=-1)


# ------------------------------------------------------------------ network
def build_dstg_net(layer_sizes, obs_dim, num_bins):
  """관측만 받아 카테고리컬 logits을 내는 MLP.

  구조는 src/diffusion_act.py의 STG 헤드와 동일:
    hk.nets.MLP(layer_sizes, relu, activate_final=True) -> hk.Linear(num_bins, with_bias=False)
  반환: (apply_fn, init_fn) — src/train_carry_qstg.py의 build_qstg_net과 같은 형태.
  """
  def _net(obs):
    h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                    activate_final=True)(obs)
    return hk.Linear(num_bins, with_bias=False)(h)

  tn = hk.without_apply_rng(hk.transform(_net))

  def init(rng):
    return tn.init(rng, jnp.zeros((2, obs_dim), jnp.float32))

  return tn.apply, init


class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--include-failures', action='store_true',
                  help=('실패 transition까지 포함해 학습하고 마지막 bin을 '
                        '실패 클래스로 둔다. 없으면 성공 transition만 사용 '
                        '(논문 원안).'))
  ap.add_argument('--steps', type=int, default=16384)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--warmup', type=int, default=500)
  ap.add_argument('--eval-every', type=int, default=500)
  ap.add_argument('--patience', type=int, default=12)
  ap.add_argument('--weight-decay', type=float, default=1e-4)
  ap.add_argument('--no-early-stop', action='store_true')
  ap.add_argument('--out', required=True, help='체크포인트 저장 경로')
  args = ap.parse_args()

  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  N = len(data['action'])
  max_steps = int(data['meta']['max_steps'])
  is_succ = data['is_success']
  ttg = data['time_to_success']
  print(f'transitions={N}  에피소드={len(np.unique(data["episode_id"]))}  '
        f'성공율={is_succ.mean():.1%}  outcomes={data["meta"]["outcomes"]}  '
        f'실패 ttg 유일값={np.unique(ttg[~is_succ])}')

  if args.include_failures:
    fail_bin = max_steps
    NUM_BINS = max_steps + 1     # 0..199 성공 + 200 실패
  else:
    fail_bin = None
    NUM_BINS = max_steps         # 0..199

  # ---- 정규화 / 분할 (held-out은 에피소드 단위 10%, train_carry_predictor.py와 동일)
  stats = compute_stats(data)
  normalize_obs = make_normalizers(stats)
  rng = np.random.default_rng(args.seed)
  ep_ids = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids, size=max(len(ep_ids) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))

  if args.include_failures:
    keep_mask = np.ones(N, dtype=bool)
  else:
    keep_mask = is_succ

  tr_mask = (~val_mask) & keep_mask
  # 검증셋도 같은 keep_mask로 거른다. succ 버전(num_bins=200)은 라벨 200을
  # 표현할 bin이 아예 없어 실패 transition을 검증에 섞으면 "구조적으로 못
  # 맞히는" 오차가 그대로 MAE를 부풀린다 — 참조 체크포인트(v2 데이터)도
  # 애초에 실패 에피소드가 없는 데이터로 검증했으므로 조건이 맞다.
  va_mask = val_mask & keep_mask

  tr_obs = jnp.asarray(obs_c[tr_mask]); tr_ttg = jnp.asarray(ttg[tr_mask])
  va_obs = jnp.asarray(obs_c[va_mask]); va_ttg = jnp.asarray(ttg[va_mask])
  va_succ = jnp.asarray(is_succ[va_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps, 관측 {tr_obs.shape[-1]}차원)'
        + ('  [실패 포함 학습, num_bins=%d]' % NUM_BINS if args.include_failures
           else '  [성공만 학습, num_bins=%d]' % NUM_BINS))

  OBS_DIM = int(tr_obs.shape[-1])
  apply_fn, init_fn = build_dstg_net(DEFAULT_LAYER_SIZES, OBS_DIM, NUM_BINS)
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

  def loss_fn(p, bo, bt):
    logits = apply_fn(p, bo)
    return -jnp.mean(tfd.Categorical(logits=logits).log_prob(bt))

  @jax.jit
  def train_step(state, key):
    idx = jax.random.randint(key, (args.batch,), 0, tr_obs.shape[0])
    l, g = jax.value_and_grad(loss_fn)(state.params, tr_obs[idx], tr_ttg[idx])
    updates, opt_state = optimizer.update(g, state.opt_state, state.params)
    return TrainState(optax.apply_updates(state.params, updates), opt_state), l

  @jax.jit
  def val_metrics(p):
    logits = apply_fn(p, va_obs)
    nll = -jnp.mean(tfd.Categorical(logits=logits).log_prob(va_ttg))
    probs = jax.nn.softmax(logits, axis=-1)
    exp = jnp.sum(probs * bin_vals[None, :], axis=-1)
    mae = jnp.mean(jnp.abs(exp - va_ttg))
    if fail_bin is not None:
      pred_fail = jnp.argmax(logits, axis=-1) == fail_bin
      fail_acc = jnp.mean(pred_fail == (~va_succ))
    else:
      fail_acc = jnp.float32(jnp.nan)
    return mae, nll, fail_acc

  t0 = time.time()
  best = dict(loss=np.inf, step=0, params=jax.device_get(state.params),
              mae=np.nan, nll=np.nan, fail_acc=np.nan)
  since = 0
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l = train_step(state, sub)
    if step % args.eval_every == 0 or step == 1:
      mae, nll, fail_acc = val_metrics(state.params)
      vloss = float(nll)
      mark = ''
      if vloss < best['loss'] - 1e-4:
        best = dict(loss=vloss, step=step, params=jax.device_get(state.params),
                    mae=float(mae), nll=float(nll), fail_acc=float(fail_acc))
        since = 0; mark = '  *best'
      else:
        since += 1
      print(f'step {step:6d}  train={float(l):.3f}  '
            f'val: MAE={float(mae):.2f} NLL={float(nll):.3f}'
            + (f' 실패분류정확도={float(fail_acc):.3f}' if fail_bin is not None else '')
            + f'  ({time.time()-t0:.0f}s){mark}', flush=True)
      if (not args.no_early_stop) and since >= args.patience:
        print(f'조기 종료: {args.patience}회 개선 없음 (best step={best["step"]})')
        break
  if not args.no_early_stop:
    state = TrainState(jax.tree.map(jnp.asarray, best['params']), state.opt_state)
  else:
    best['step'] = args.steps
  print(f'\n[최적] step={best["step"]}  val MAE={best["mae"]:.2f}  NLL={best["nll"]:.3f}')

  val_fail_base_rate = float(jnp.mean(~va_succ))
  meta = {
      'data': args.data, 'include_failures': bool(args.include_failures),
      'steps': args.steps, 'seed': args.seed,
      'best_step': int(best['step']), 'weight_decay': args.weight_decay,
      'val_mae': float(best['mae']), 'val_nll': float(best['nll']),
      'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  }
  if args.include_failures:
    meta['val_fail_accuracy'] = float(best['fail_acc'])
    meta['val_fail_base_rate'] = val_fail_base_rate
    print(f'val_fail_accuracy={best["fail_acc"]:.3f}  '
          f'val_fail_base_rate(다수결 기준선)={val_fail_base_rate:.3f}  '
          f'차이={best["fail_acc"] - val_fail_base_rate:+.3f}')
    if best['fail_acc'] - val_fail_base_rate < 0.03:
      print('[경고] 실패 분류 정확도가 다수결 기준선을 +0.03 이상 넘지 못함 '
            '— 예측기가 실패를 유의미하게 배우지 못했을 수 있음.')

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': {'frame_mean': stats['frame_mean'],
                       'frame_std': stats['frame_std']},
        'obs_dim': OBS_DIM,
        'num_bins': NUM_BINS,
        'fail_bin': fail_bin,
        'max_steps': max_steps,
        'layer_sizes': DEFAULT_LAYER_SIZES,
        'meta': meta,
    }, fp)
  print(f'체크포인트 저장: {args.out}')

  if not args.include_failures:
    ref_mae = 15.04
    ratio = float(best['mae']) / ref_mae
    print(f'\n참조(grasp_carry_diff100, v2 성공 데모) val_mae={ref_mae:.2f} 대비 '
          f'이 succ 버전 val_mae={float(best["mae"]):.2f}  비율={ratio:.2f}x'
          + ('  [자릿수 일치]' if 0.5 <= ratio <= 2.0 else '  [경고: 2배 이상 벗어남]'))


if __name__ == '__main__':
  main()
