"""PushT steps-to-go 예측기 + BC 정책 (phase 4).

2D 장애물 환경(phase 3)에서 확인된 구조적 한계 때문에 태스크를 옮겼다:
  - 완전관측 + 결정론적 전문가라 p(a|s)가 구성상 단봉 → 다봉성 검증이 공허했음
  - STG의 σ²가 μ와 중복이고 기하학으로 재구성 가능 → 신호로서 무가치
PushT는 사람 데모라 같은 상황에서 미는 방향이 실제로 갈리고(실측: 이웃 액션
방향 집중도 R<0.7인 상태가 34%), 완료 시점이 26~241로 9배 가변이라 STG 라벨이
퇴화하지 않는다.

데이터: lerobot/pusht_keypoints → data/pusht_demos.pkl (scripts로 변환)
  관측 18차원 = agent_pos(2) + env_state(16, T블록 키포인트 8개)
  액션 2차원 = 목표 위치(픽셀 좌표 0~512)
  라벨 = coverage>=0.80 첫 도달까지 남은 스텝

실행:
  python -m src.train_pusht_predictor --steps 32768
  python -m src.train_pusht_predictor --act-head mixture --n-mix 3
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
    build_mixture_act_discrete_dist_v0,
)
from src.diffusion_act import build_diffusion_act_chunk, diffusion_loss

OBS_FIELDS = ('agent_pos', 'env_state')
EP_CAP = 300              # gym-pusht 기본 상한
MAX_DISTANCE = EP_CAP     # STG 라벨은 정의상 이보다 작다
NUM_BINS = MAX_DISTANCE   # bin_size=1 → raw ttg가 곧 클래스 인덱스

DATA_PATH = 'data/pusht_demos.pkl'
SUCCESS_COVERAGE = 0.80   # 데이터 변환 때 쓴 완료 기준과 동일하게


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

  def unnormalize_action(a):
    return a * stats['act_std'] + stats['act_mean']

  return normalize_obs, normalize_action, unnormalize_action


def concat_obs(obs):
  return jnp.concatenate([jnp.asarray(obs[f]) for f in OBS_FIELDS], axis=-1)


class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default=DATA_PATH)
  ap.add_argument('--steps', type=int, default=32768)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval-episodes', type=int, default=50)
  ap.add_argument('--act-head', choices=['gaussian', 'mixture', 'diffusion'],
                  default='gaussian')
  ap.add_argument('--chunk', type=int, default=1,
                  help='액션 청크 길이 H (한 번에 예측할 스텝 수). '
                       '1이면 기존 한 스텝 예측.')
  ap.add_argument('--action-horizon', type=int, default=0,
                  help='청크 중 실제로 실행할 스텝 수 A (0이면 H//2). '
                       'Diffusion Policy 기본은 H=16, A=8.')
  ap.add_argument('--diffusion-steps', type=int, default=100)
  # From: real-stanford/diffusion_policy (train_diffusion_unet_lowdim_workspace)
  ap.add_argument('--ema-power', type=float, default=0.75,
                  help='EMA power (공식 DP 기본 0.75). 0이면 EMA 끔.')
  ap.add_argument('--minmax-action', action='store_true',
                  help='액션을 [-1,1] min-max로 정규화(공식 DP 방식). '
                       'DDPM은 데이터가 [-1,1]이라고 가정하므로 디퓨전에 권장.')
  ap.add_argument('--obs-steps', type=int, default=1,
                  help='관측 이력 길이 (공식 DP는 2)')
  ap.add_argument('--warmup', type=int, default=500)
  ap.add_argument('--select-by', choices=['nll_bc', 'nll'], default='nll_bc',
                  help='조기종료/best 선택 기준. nll=STG(dist) NLL 단독(STG 연구용)')
  ap.add_argument('--backbone', choices=['unet', 'mlp'], default='unet',
                  help='디퓨전 노이즈 예측기 백본. unet=공식 ConditionalUnet1D')
  ap.add_argument('--n-mix', type=int, default=3)
  ap.add_argument('--eval-every', type=int, default=500,
                  help='검증 주기. 데이터가 작아 최적점이 이르게 오므로 촘촘히.')
  ap.add_argument('--patience', type=int, default=12,
                  help='이 횟수만큼 검증 손실 개선이 없으면 조기 종료')
  ap.add_argument('--weight-decay', type=float, default=1e-4)
  ap.add_argument('--no-early-stop', action='store_true',
                  help='조기 종료 끄고 --steps 끝까지. 디퓨전은 dist 헤드가 '
                       '검증손실을 지배해 너무 일찍 멈추므로 필요.')
  ap.add_argument('--tag', default='', help='체크포인트 경로 접미사(시드 비교용)')
  args = ap.parse_args()

  if args.act_head == 'mixture':
    head_suffix = f'_mix{args.n_mix}'
  elif args.act_head == 'diffusion':
    head_suffix = f'_diff{args.diffusion_steps}'
  else:
    head_suffix = ''
  H = max(1, args.chunk)
  A_EXEC = args.action_horizon if args.action_horizon > 0 else max(1, H // 2)
  if H > 1:
    head_suffix += f'_h{H}a{A_EXEC}'
  ckpt_path = f'checkpoints/pusht{head_suffix}{args.tag}/predictor.pkl'

  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  N = len(data['action'])
  lengths = np.bincount(data['episode_id'])
  lengths = lengths[lengths > 0]
  print(f'transitions={N}  에피소드={len(lengths)}  '
        f'길이 p50={np.percentile(lengths,50):.0f} '
        f'범위=[{lengths.min()},{lengths.max()}]')
  assert data['time_to_success'].max() < MAX_DISTANCE, 'STG가 bin 범위 초과'

  # ---- 정규화 / 분할 (held-out은 에피소드 단위 10%)
  stats = compute_stats(data)
  if args.minmax_action:
    # From: diffusion_policy/model/common/normalizer.py (output_min=-1, max=1)
    amin = data['action'].min(0); amax = data['action'].max(0)
    rng_ = np.maximum(amax - amin, 1e-4)
    stats['act_mean'] = ((amin + amax) / 2).astype(np.float32)
    stats['act_std'] = (rng_ / 2).astype(np.float32)   # (a-mean)/std ∈ [-1,1]
  normalize_obs, normalize_action, unnormalize_action = make_normalizers(stats)
  rng = np.random.default_rng(args.seed)
  ep_ids = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids, size=max(len(ep_ids) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))
  if args.obs_steps > 1:
    # 과거 obs_steps개를 이어붙임(에피소드 시작은 첫 관측 반복)
    eid_o = data['episode_id']; stacked = []
    for e in np.unique(eid_o):
      idx = np.where(eid_o == e)[0]; seg = obs_c[idx]; T = len(idx)
      hs = [seg[np.maximum(np.arange(T) - k, 0)] for k in range(args.obs_steps-1, -1, -1)]
      stacked.append(np.concatenate(hs, axis=-1))
    obs_c = np.concatenate(stacked, axis=0)
    print(f'관측 이력 {args.obs_steps}스텝 → 관측 {obs_c.shape[-1]}차원')
  act_n = np.asarray(normalize_action(data['action']))
  ttg = data['time_to_success']
  if H > 1:
    # 각 전이 i에 대해 앞으로 H스텝 액션을 쌓는다. 에피소드 끝은 마지막 액션 반복.
    eid = data['episode_id']
    chunks = np.repeat(act_n[:, None, :], H, axis=1)
    starts = {}
    for e in np.unique(eid):
      idx = np.where(eid == e)[0]
      starts[e] = idx
    for e, idx in starts.items():
      seg = act_n[idx]                       # (T,2)
      T = len(idx)
      for h in range(H):
        src = np.minimum(np.arange(T) + h, T - 1)
        chunks[idx, h] = seg[src]
    act_n = chunks.reshape(len(act_n), -1)   # (N, H*2)
    print(f'액션 청킹: H={H} 실행={A_EXEC} → 액션 라벨 {act_n.shape[-1]}차원')
  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_act = jnp.asarray(act_n[~val_mask])
  tr_ttg = jnp.asarray(ttg[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_act = jnp.asarray(act_n[val_mask])
  va_ttg = jnp.asarray(ttg[val_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps, 관측 {tr_obs.shape[-1]}차원)')

  # ---- 네트워크
  ACT_FLAT = int(act_n.shape[-1])          # H*2
  OBS_DIM = int(tr_obs.shape[-1])
  dummy = np.ones((4, OBS_DIM), dtype=np.float32)
  is_diff = args.act_head == 'diffusion'
  if is_diff:
    nets = build_diffusion_act_chunk((256, 256, 256), ACT_FLAT, NUM_BINS,
                                     OBS_DIM,
                                     n_diffusion_steps=args.diffusion_steps,
                                     backbone=args.backbone,
                                     horizon=H, act_dim=2)
  elif args.act_head == 'mixture':
    nets = build_mixture_act_discrete_dist_v0((256, 256, 256), ACT_FLAT,
                                              NUM_BINS, dummy, n_mix=args.n_mix)
  else:
    nets = build_continuous_act_discrete_dist_v0((256, 256, 256), ACT_FLAT,
                                                 NUM_BINS, dummy)
  bin_vals = np.linspace(0, MAX_DISTANCE, NUM_BINS + 1,
                         endpoint=True, dtype=np.float32)[:-1]
  sched_lr = optax.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup,
      decay_steps=max(args.steps, args.warmup + 1), end_value=args.lr * 0.05)
  optimizer = optax.adamw(sched_lr, b1=0.95, b2=0.999,
                          weight_decay=args.weight_decay)
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = nets.init(sub) if is_diff else nets.network.init(sub)
  state = TrainState(params, optimizer.init(params))
  use_ema = args.ema_power > 0
  ema_params = jax.tree.map(jnp.array, params) if use_ema else None

  @jax.jit
  def ema_update(ema, new, step):
    # From: diffusion_policy/model/diffusion/ema_model.py (inv_gamma=1, power)
    decay = jnp.clip(1.0 - (1.0 + step) ** (-args.ema_power), 0.0, 0.9999)
    return jax.tree.map(lambda e, n: e * decay + n * (1.0 - decay), ema, new)

  def loss_fn(p, bo, ba, bt, key):
    if is_diff:
      bc = diffusion_loss(nets, p, bo, ba, key)          # 노이즈 예측 MSE
      logits = nets.dist_logits(p, bo)
      dl = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(bt.shape[0]),
                                                bt.astype(jnp.int32)])
    else:
      preds = nets.network.apply(p, bo)
      bc = -jnp.mean(nets.act_log_prob(preds.act_dist_params, ba))
      dl = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, bt))
    return bc + dl, (bc, dl)

  @jax.jit
  def train_step(state, key):
    key, sub = jax.random.split(key)
    idx = jax.random.randint(key, (args.batch,), 0, tr_obs.shape[0])
    (l, (bc, dl)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, tr_obs[idx], tr_act[idx], tr_ttg[idx], sub)
    updates, opt_state = optimizer.update(g, state.opt_state, state.params)
    return (TrainState(optax.apply_updates(state.params, updates), opt_state),
            l, bc, dl)

  @jax.jit
  def val_metrics(p, key):
    if is_diff:
      logits = nets.dist_logits(p, va_obs)
      bc = diffusion_loss(nets, p, va_obs, va_act, key)
    else:
      preds = nets.network.apply(p, va_obs)
      logits = preds.dist_to_succ_dist_params.logits
      bc = -jnp.mean(nets.act_log_prob(preds.act_dist_params, va_act))
    exp = jnp.sum(jax.nn.softmax(logits) * bin_vals[None, :], axis=-1)
    mae = jnp.mean(jnp.abs(exp - va_ttg))
    nll = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(va_ttg.shape[0]),
                                               va_ttg.astype(jnp.int32)])
    return mae, nll, bc

  t0 = time.time()
  best = dict(loss=np.inf, step=0, params=jax.device_get(state.params),
              mae=np.nan, nll=np.nan, bc=np.nan)
  since = 0
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l, bc, dl = train_step(state, sub)
    if use_ema:
      ema_params = ema_update(ema_params, state.params, step)
    if step % args.eval_every == 0 or step == 1:
      key, vk = jax.random.split(key)
      eval_p = ema_params if use_ema else state.params
      mae, nll, vbc = val_metrics(eval_p, vk)
      vloss = float(nll) if args.select_by == 'nll' else float(nll) + float(vbc)
      mark = ''
      if vloss < best['loss'] - 1e-4:
        best = dict(loss=vloss, step=step,
                    params=jax.device_get(eval_p),
                    mae=float(mae), nll=float(nll), bc=float(vbc))
        since = 0; mark = '  *best'
      else:
        since += 1
      print(f'step {step:6d}  train={float(l):.3f}  '
            f'val: MAE={float(mae):.2f} NLL={float(nll):.3f} '
            f'bc={float(vbc):.3f} 합={vloss:.3f}'
            f'  ({time.time()-t0:.0f}s){mark}', flush=True)
      if (not args.no_early_stop) and since >= args.patience:
        print(f'조기 종료: {args.patience}회 개선 없음 (best step={best["step"]})')
        break
  # 최적 파라미터 복원 — 이후 롤아웃/저장은 전부 이걸로
  if not args.no_early_stop:
    state = TrainState(jax.tree.map(jnp.asarray, best['params']), state.opt_state)
  else:
    best['step'] = args.steps
    if use_ema:
      state = TrainState(ema_params, state.opt_state)
      print(f'(조기 종료 비활성: EMA 파라미터 사용, power={args.ema_power})')
    else:
      print('(조기 종료 비활성: 마지막 파라미터 사용)')
  print(f'\n[최적] step={best["step"]}  val MAE={best["mae"]:.2f} '
        f'NLL={best["nll"]:.3f} bc={best["bc"]:.3f}')

  # ---- BC 정책 롤아웃 평가 (gym-pusht)
  import gymnasium as gym
  import gym_pusht  # noqa: F401

  if is_diff:
    _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
    def get_chunk(p, concat, rng_):
      return np.asarray(_sample(p, jnp.asarray(concat), rng_))[0]
  else:
    def _pf(p, concat, rng_):
      preds = nets.network.apply(p, concat)
      return nets.sample_act_mode(preds.act_dist_params, rng_)
    _pol = jax.jit(_pf, backend='cpu')
    def get_chunk(p, concat, rng_):
      return np.asarray(_pol(p, concat, rng_))[0]

  env = gym.make('gym_pusht/PushT-v0', obs_type='environment_state_agent_pos')
  succ, lens, covs = 0, [], []
  kk = jax.random.PRNGKey(1234)
  for e in range(args.eval_episodes):
    obs, info = env.reset(seed=10000 + e)
    done = False; n = 0; best_cov = 0.0; hit = False
    hist = None            # 관측 이력 버퍼 (obs_steps개)
    while not done and n < EP_CAP:
      o = {'agent_pos': np.asarray(obs['agent_pos'], np.float32)[None],
           'env_state': np.asarray(obs['environment_state'], np.float32)[None]}
      cur = np.asarray(concat_obs(normalize_obs(o)))       # (1, 18)
      if args.obs_steps > 1:
        if hist is None:
          hist = [cur] * args.obs_steps                    # 시작은 첫 관측 반복
        else:
          hist = hist[1:] + [cur]
        c = np.concatenate(hist, axis=-1)                  # (1, 18*obs_steps)
      else:
        c = cur
      kk, s2 = jax.random.split(kk)
      flat = get_chunk(state.params, c, s2)          # (H*2,)
      chunk = flat.reshape(H, 2)
      # 청크 앞 A_EXEC개만 실행하고 다시 예측 (receding horizon)
      for h in range(A_EXEC):
        if done or n >= EP_CAP:
          break
        a = np.asarray(unnormalize_action(chunk[h]), np.float32)
        obs, r, term, trunc, info = env.step(a)
        best_cov = max(best_cov, float(info.get('coverage', r)))
        n += 1
        if best_cov >= SUCCESS_COVERAGE and not hit:
          hit = True; lens.append(n)
        done = term or trunc
    succ += int(hit); covs.append(best_cov)
  env.close()
  rate = succ / args.eval_episodes
  kk, vk = jax.random.split(kk)
  mae, nll, vbc = val_metrics(state.params, vk)
  print(f'\n최종: coverage>={SUCCESS_COVERAGE} 도달률 {rate:.2f} '
        f'({args.eval_episodes} eps, 도달 시 스텝 중앙 '
        f'{np.median(lens) if lens else float("nan"):.0f})  '
        f'최고 coverage 중앙={np.median(covs):.3f}  '
        f'val MAE={float(mae):.2f} NLL={float(nll):.3f}')

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
            'env': 'gym_pusht/PushT-v0', 'data': args.data,
            'act_head': args.act_head, 'n_mix': args.n_mix,
            'chunk': H, 'action_horizon': A_EXEC,
            'diffusion_steps': args.diffusion_steps,
            'ema_power': args.ema_power, 'minmax_action': args.minmax_action,
            'obs_steps': args.obs_steps, 'backbone': args.backbone,
            'steps': args.steps, 'seed': args.seed,
            'best_step': int(best['step']), 'weight_decay': args.weight_decay,
            'success_coverage': SUCCESS_COVERAGE,
            'val_mae': float(mae), 'val_nll': float(nll),
            'policy_success_rate': rate,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {ckpt_path}')


if __name__ == '__main__':
  main()
