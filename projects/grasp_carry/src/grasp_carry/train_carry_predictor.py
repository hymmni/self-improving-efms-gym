"""GraspCarry2D steps-to-go 예측기 + BC 정책 (phase 3 후속).

`src/train_pusht_predictor.py`의 최소 버전이다 — 믹스처 액션 헤드, 청킹 등
부가 옵션을 빼고 범주형 STG 헤드 + (가우시안|디퓨전) 액션 헤드만 남겼다.

**가우시안 헤드는 이 데이터에서 성공률 2%로 사실상 못 쓴다.** 원인은 데이터
자체에 있다 — `explore_range`로 수집해서 **같은 상태에서 다른 속도를 고른
경우**가 의도적으로 섞여 있는데(다봉 p(a|s)), 단일 가우시안은 그 평균을
뱉어 phase 전환에 필요한 정확한 목표 위치를 못 낸다. PushT에서 같은 문제를
디퓨전/믹스처 헤드로 풀었던 것과 동일한 처방이라 디퓨전 헤드를 추가했다
(`src/diffusion_act.py`, `real-stanford/diffusion_policy` 이식).
`--horizon H`(기본 1, 기존 동작과 동일)로 액션 청킹을 켤 수 있다. H>1이면
`--exec-horizon`(기본 H)만큼만 실행하고 새 관측으로 재추론하는 receding
horizon 방식으로 롤아웃한다(square_assembly/3D 태스크와 같은 패턴). 2026-08-10:
원래는 "매 스텝 새 절대 목표를 받는 구조라 청킹이 불필요하다"고 H=1로
고정했었는데, 청킹을 쓰더라도 매 스텝 재추론해야 하는 게 아니라 exec-horizon
개만 실행 후 재추론하면 되므로 그 논리가 청킹 자체를 막을 근거는 아니었다.

데이터: `collect_carry_demos.py` → `data/grasp_carry_demos_v2.pkl`
  관측 60차원 = `env.observe_frame()`(15필드) x `obs_history`(4) 스택,
    `env.step()`/`env.reset()`이 실제로 내주는 값 그대로(은닉 물성 없음).
  액션 4차원 = (x, y, theta, grip) 절대 목표 pose. theta는 이 정책에서
    항상 0(회전 미사용)이라 학습에 죽은 차원으로 남지만 해는 없다.
  라벨 = 성공까지 남은 스텝(`time_to_success`). 에피소드는 성공 시점에서
    잘려 있다(성공하지 못한 에피소드는 애초에 데이터에 없음).

체크포인트는 `stg_probe.GenericSTGProbe`와 호환되지 않는다 — 그 프로브의
정규화기(`pointmass_core.make_normalizers`)가 원래 pointmass 3필드
(cur_pos/cur_vel/goal_pos)에 하드코딩돼 있어 단일 'frame' 필드를 못 읽는다.
이 스크립트는 자체 정규화기로 학습·평가·저장을 전부 자기완결적으로 한다.

실행:
  python -m grasp_carry.train_carry_predictor --data data/grasp_carry_demos_v2.pkl
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

from grasp_carry.diffusion_act import build_diffusion_act_chunk, diffusion_loss

OBS_FIELDS = ('frame',)


# -------------------------------------------------------------- normalizers
def compute_stats(data, minmax_action=False):
  obs, act = data['observation'], data['action']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  for f in OBS_FIELDS:
    stats[f'{f}_mean'], stats[f'{f}_std'] = ms(obs[f])
  if minmax_action:
    # From: diffusion_policy/model/common/normalizer.py (output_min=-1, max=1)
    # DDPM은 데이터가 대략 [-1,1]이라고 가정하므로 디퓨전 헤드에 권장.
    amin, amax = act.min(0), act.max(0)
    span = np.maximum(amax - amin, 1e-4)
    stats['act_mean'] = ((amin + amax) / 2).astype(np.float32)
    stats['act_std'] = (span / 2).astype(np.float32)
  else:
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


def build_action_chunks(action, episode_id, horizon):
  """각 transition마다 미래 horizon스텝의 액션을 이어붙인 청크 (N, horizon*A)를
  만든다. 에피소드 끝을 넘어가면 그 에피소드의 마지막 액션을 반복해 패딩한다
  (real-stanford/diffusion_policy의 표준 관행)."""
  N, A = action.shape
  uniq_eids, ep_start, ep_count = np.unique(episode_id, return_index=True,
                                            return_counts=True)
  ep_end_excl = np.repeat(ep_start + ep_count, ep_count)   # transition별 소속 에피소드의 끝(배타적) 인덱스
  chunks = np.empty((N, horizon, A), dtype=action.dtype)
  for h in range(horizon):
    idx = np.minimum(np.arange(N) + h, ep_end_excl - 1)
    chunks[:, h, :] = action[idx]
  return chunks   # (N, horizon, A) -- 정규화 후 호출부에서 평탄화한다


class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default='data/grasp_carry_demos_v2.pkl')
  ap.add_argument('--steps', type=int, default=16384)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval-episodes', type=int, default=50)
  ap.add_argument('--warmup', type=int, default=500)
  ap.add_argument('--eval-every', type=int, default=500,
                  help='검증 주기. 데이터가 작아 최적점이 이르게 오므로 촘촘히.')
  ap.add_argument('--patience', type=int, default=12,
                  help='이 횟수만큼 검증 손실 개선이 없으면 조기 종료')
  ap.add_argument('--weight-decay', type=float, default=1e-4)
  # From: real-stanford/diffusion_policy (train_diffusion_unet_lowdim_workspace)
  ap.add_argument('--ema-power', type=float, default=0.75,
                  help='EMA power (0이면 EMA 끔).')
  ap.add_argument('--act-head', choices=['gaussian', 'diffusion'],
                  default='gaussian',
                  help=('gaussian은 이 데이터(탐색 수집이라 다봉)에서 '
                        '성공률 2%%로 사실상 못 쓴다. diffusion을 쓰면 '
                        '--minmax-action도 같이 켜는 걸 권장한다.'))
  ap.add_argument('--diffusion-steps', type=int, default=100)
  ap.add_argument('--backbone', choices=['unet', 'mlp'], default='mlp',
                  help='디퓨전 노이즈 예측기 백본. 청크 길이가 1(H=1)이라 '
                       '시간축 conv인 unet보다 mlp가 자연스럽다.')
  ap.add_argument('--minmax-action', action='store_true',
                  help='액션을 [-1,1] min-max로 정규화(공식 DP 방식).')
  ap.add_argument('--horizon', type=int, default=1,
                  help=('액션 청크 길이 H. 기본 1(기존 동작, 청킹 없음). H>1이면 '
                        '디퓨전 헤드가 미래 H스텝을 한 번에 예측하고, 롤아웃(평가)은 '
                        '--exec-horizon만큼만 실행 후 재추론한다.'))
  ap.add_argument('--exec-horizon', type=int, default=None,
                  help='롤아웃 시 청크에서 실제로 실행할 스텝 수. 기본 None -> --horizon과 동일.')
  ap.add_argument('--no-early-stop', action='store_true')
  ap.add_argument('--tag', default='', help='체크포인트 경로 접미사')
  args = ap.parse_args()
  is_diff = args.act_head == 'diffusion'
  exec_horizon = args.exec_horizon if args.exec_horizon is not None else args.horizon
  if exec_horizon > args.horizon:
    raise ValueError(f'--exec-horizon({exec_horizon})은 --horizon({args.horizon})보다 클 수 없다.')

  head_suffix = f'_diff{args.diffusion_steps}' if is_diff else ''
  ckpt_path = f'checkpoints/grasp_carry{head_suffix}{args.tag}/predictor.pkl'

  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  N = len(data['action'])
  lengths = np.bincount(data['episode_id'])
  lengths = lengths[lengths > 0]
  max_steps = int(data['meta']['max_steps'])
  print(f'transitions={N}  에피소드={len(lengths)}  '
        f'길이 p50={np.percentile(lengths, 50):.0f} '
        f'범위=[{lengths.min()},{lengths.max()}]  max_steps(bin 상한)={max_steps}')
  print(f'수집 outcomes={data["meta"]["outcomes"]}')
  assert data['time_to_success'].max() < max_steps, 'STG가 bin 범위 초과'

  MAX_DISTANCE = max_steps
  NUM_BINS = max_steps

  # ---- 정규화 / 분할 (held-out은 에피소드 단위 10%)
  stats = compute_stats(data, minmax_action=args.minmax_action)
  normalize_obs, normalize_action, unnormalize_action = make_normalizers(stats)
  rng = np.random.default_rng(args.seed)
  ep_ids = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids, size=max(len(ep_ids) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))
  act_chunks = build_action_chunks(data['action'], data['episode_id'], args.horizon)  # (N, H, A)
  act_n = np.asarray(normalize_action(act_chunks)).reshape(N, -1)                     # (N, H*A)
  ttg = data['time_to_success']

  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_act = jnp.asarray(act_n[~val_mask])
  tr_ttg = jnp.asarray(ttg[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_act = jnp.asarray(act_n[val_mask])
  va_ttg = jnp.asarray(ttg[val_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps, 관측 {tr_obs.shape[-1]}차원)')

  # ---- 네트워크
  ACT_DIM = int(data['action'].shape[-1])       # 스텝당(raw) 액션 차원 -- 항상 4, horizon과 무관
  CHUNK_FLAT_DIM = ACT_DIM * args.horizon        # 청크 평탄화 차원 (H=1이면 ACT_DIM과 동일)
  OBS_DIM = int(tr_obs.shape[-1])
  dummy = np.ones((4, OBS_DIM), dtype=np.float32)
  if is_diff:
    nets = build_diffusion_act_chunk((256, 256, 256), CHUNK_FLAT_DIM, NUM_BINS,
                                     OBS_DIM, n_diffusion_steps=args.diffusion_steps,
                                     backbone=args.backbone, horizon=args.horizon,
                                     act_dim=ACT_DIM)
  else:
    if args.horizon != 1:
      raise ValueError('--horizon>1은 diffusion 헤드에서만 지원한다(gaussian은 미지원).')
    # 2026-09-02: gaussian 헤드(networks.py)만 tensorflow-probability에
    # 의존한다 — 이 환경은 jax==0.9.2/tfp==0.25.0이 비호환이라(레거시
    # jax.interpreters.xla 재노출 API가 사라짐) 모듈 최상단에서 import하면
    # concat_obs 등 tfp를 전혀 안 쓰는 롤아웃/추론 유틸까지 import가 깨진다.
    # gaussian 경로(여기)만 실제로 필요하므로 지연 import로 스코프를 좁힌다
    # — jax/tfp 버전 비호환 자체는 그대로 남아있어 이 경로는 여전히 깨져 있다.
    from grasp_carry.networks import build_continuous_act_discrete_dist_v0
    nets = build_continuous_act_discrete_dist_v0((256, 256, 256), ACT_DIM,
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
      vloss = float(nll) + float(vbc)
      mark = ''
      if vloss < best['loss'] - 1e-4:
        best = dict(loss=vloss, step=step, params=jax.device_get(eval_p),
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

  # ---- BC 정책 롤아웃 평가 (GraspCarry2D, 학습 데이터와 겹치지 않는 시드)
  from grasp_carry.config import CarryConfig
  from grasp_carry.env import GraspCarry2D

  if is_diff:
    _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
    def _pol(p, concat, rng_):
      return _sample(p, jnp.asarray(concat), rng_)
  else:
    def _pf(p, concat, rng_):
      preds = nets.network.apply(p, concat)
      return nets.sample_act_mode(preds.act_dist_params, rng_)
    _pol = jax.jit(_pf, backend='cpu')

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  succ, lens, outcomes = 0, [], {}
  kk = jax.random.PRNGKey(1234)
  eval_seed0 = 900000    # 수집 스크립트의 seed0(기본 0)과 안 겹치는 홀드아웃 구간
  for e in range(args.eval_episodes):
    obs, info = env.reset(seed=eval_seed0 + e)
    step_count = 0
    done = False
    while step_count < cfg.max_steps and not done:
      o = {'frame': np.asarray(obs, np.float32)[None]}
      c = np.asarray(concat_obs(normalize_obs(o)))
      kk, s2 = jax.random.split(kk)
      chunk_n = np.asarray(_pol(state.params, c, s2))[0].reshape(args.horizon, ACT_DIM)
      chunk = np.asarray(unnormalize_action(chunk_n), np.float32)
      # receding horizon: 청크 중 exec_horizon개만 실제로 실행하고 새 관측으로 재추론
      for h in range(min(exec_horizon, args.horizon)):
        if step_count >= cfg.max_steps:
          break
        obs, r, term, trunc, info = env.step(chunk[h])
        step_count += 1
        if term or trunc:
          done = True
          break
    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    if info['outcome'] == 'success':
      succ += 1
      lens.append(info['steps'])
  env.close() if hasattr(env, 'close') else None
  rate = succ / args.eval_episodes
  print(f'\n최종: BC 정책 성공률 {rate:.2f} ({args.eval_episodes}eps) '
        f'outcomes={outcomes}  성공 시 스텝 중앙 '
        f'{np.median(lens) if lens else float("nan"):.0f}')

  os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
  with open(ckpt_path, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': stats,
        'dc_config': {'min_distance': 0, 'max_distance': MAX_DISTANCE,
                      'num_bins': NUM_BINS},
        'obs_fields': list(OBS_FIELDS),
        'obs_dim': OBS_DIM,
        'meta': {
            'env': 'grasp_carry/GraspCarry2D', 'data': args.data,
            'act_head': args.act_head, 'diffusion_steps': args.diffusion_steps,
            'backbone': args.backbone, 'minmax_action': args.minmax_action,
            'ema_power': args.ema_power, 'horizon': args.horizon,
            'exec_horizon': exec_horizon, 'act_dim': ACT_DIM,
            'steps': args.steps, 'seed': args.seed,
            'best_step': int(best['step']), 'weight_decay': args.weight_decay,
            'val_mae': float(best['mae']), 'val_nll': float(best['nll']),
            'policy_success_rate': rate, 'eval_outcomes': outcomes,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {ckpt_path}')


if __name__ == '__main__':
  main()
