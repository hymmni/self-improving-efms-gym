r"""SI-EFM Stage-2(Algorithm 1) — DDPO-SF로 파지·운반 디퓨전 정책을 자기개선한다.

배경(코드가 아니라 여기 남기는 이유는 이 스크립트의 존재 이유 자체가
알고리즘 선택의 근거이기 때문이다) — 논문 Algorithm 1은:

  1. Stage-1 체크포인트를 하나 복사해 얼린다(보상/성공 판정 전용).
  2. 현재 정책으로 on-policy 롤아웃을 모은다.
  3. r_t = d(o_t,g) - d(o_{t+1},g)  (식 2)
  4. R_t = sum_{i>=t} gamma^(i-t) r_i
  5. REINFORCE: loss = -c * mean(R_t * log p(a_t|o_t,g))
  6. 그 배치는 한 번 쓰고 버린다(off-policy·부트스트래핑을 피하는 설계).

논문 정책은 이산 토큰이라 log p(a|o)가 바로 나오지만, 이 레포의 정책은 연속
디퓨전(`src/diffusion_act.py`)이라 닫힌 형태가 없다. `src/ddpo.py`(phase 4
step 0)가 역확산 100단계 각각을 가우시안 전이로 보고 단계별 로그확률의 합으로
대체한다(DDPO, Black et al. 2023) — 이 알고리즘이 DDPO-SF(score function,
바닐라 REINFORCE)인 이유는 데이터 재사용이 없어 논문의 on-policy 설계와
정확히 맞기 때문이다.

보상/성공 판정은 `src/carry_stg_reward.py`(phase 4 step 2)가 감싼 얼려진
관측-only STG 예측기(`src/train_carry_dstg.py`, phase 4 step 1)에서 나온다.
환경의 outcome/terminated/truncated는 어디에도 학습 신호로 들어가지 않는다
(로그에만 쓴다) — 논문 핵심 주장("외부 감독 없는 자기개선")이 걸려 있다.

실행:
  python train_carry_si.py \
      --policy-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
      --d-ckpt checkpoints/grasp_carry_dstg_succ/predictor.pkl \
      --statistic mean \
      --iterations 100 --episodes-per-iter 32 \
      --gamma 0.9 --reinforce-scale 5e-2 --lr 1e-5 \
      --termination learned \
      --out checkpoints/grasp_carry_si_mean_succ/predictor.pkl
"""

import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.diffusion_act import build_diffusion_act_chunk
from src.ddpo import build_ddpo
from src.carry_stg_reward import StgReward, calibrate_threshold, _val_episode_ids
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D

# From: run_bc_stg_guided.py load_bc_policy() — 대상 체크포인트가 학습된 고정 구조.
LAYER_SIZES = (256, 256, 256)
DRIFT_N_OBS = 256
DRIFT_SEED = 12345
DRIFT_SAMPLE_SEED = 999_999


# ---------------------------------------------------------------- 순수 함수부
# (테스트가 직접 부르는 부분 — env/정책 없이 검증 가능해야 한다)

def compute_step_rewards(d_vals: np.ndarray) -> np.ndarray:
  """식(2): r_t = d(o_t,g) - d(o_{t+1},g).

  d_vals: (T+1,) 한 에피소드의 d(o_0..o_T). 반환 (T,).

  주의: 인자로 받는 것은 d값뿐이다 — env의 outcome/terminated/success는
  여기 어디에도 없다. 논문의 핵심 주장("외부 감독 없는 자기개선")이 이
  경계에 걸려 있으므로, 이 함수의 시그니처 자체가 그 경계다.
  """
  d_vals = np.asarray(d_vals, dtype=np.float64)
  return (d_vals[:-1] - d_vals[1:]).astype(np.float32)


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
  """R_t = sum_{i>=t} gamma^(i-t) * r_i, 역순 누적으로 계산. rewards: (T,) -> (T,)."""
  rewards = np.asarray(rewards, dtype=np.float64)
  T = rewards.shape[0]
  returns = np.zeros(T, dtype=np.float64)
  acc = 0.0
  for t in range(T - 1, -1, -1):
    acc = rewards[t] + gamma * acc
    returns[t] = acc
  return returns.astype(np.float32)


def make_pair_index_table(n_steps: int):
  """`xs`(n_steps+1, ...) 안에서 학습에 쓰는 (i_in, i_out, kk) 대응표.

  `src/ddpo.py`의 `chain_logp` scan body(kk = n_steps-1-i, i=0..n_steps-2)와
  정확히 같은 규칙: xs[i] --(kk=n_steps-1-i)--> xs[i+1]. kk=0(i=n_steps-1)은
  결정론적 전이라 제외 — 그래서 표의 길이는 n_steps-1이다.

  반환: (i_in, i_out, kk) 각각 (n_steps-1,) int 배열.
  """
  i_in = np.arange(n_steps - 1)
  i_out = i_in + 1
  kk = (n_steps - 1) - i_in
  return i_in, i_out, kk


# ----------------------------------------------------------------- 체크포인트

def load_policy_for_training(ckpt_path: str):
  """`run_bc_stg_guided.load_bc_policy`와 같은 방식으로 nets를 복원하고,
  DDPO 함수를 그 위에 얹는다."""
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  nets = build_diffusion_act_chunk(
      LAYER_SIZES, act_dim, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=1, act_dim=act_dim)
  ddpo = build_ddpo(nets, act_dim)
  return ck, nets, ddpo, act_dim


def _dist_head_keys(params) -> list:
  """From: tests/test_ddpo.py _is_dist_head_key — STG(dist) 헤드는 haiku가
  생성 순서상 가장 큰 접미사를 붙인 mlp_2/*, linear_2 뿐이다."""
  return [k for k in params.keys()
          if k.startswith('mlp_2') or k == 'linear_2']


# --------------------------------------------------------------------- 롤아웃

def collect_episode(nets, ddpo, params, reward: StgReward, env: GraspCarry2D,
                    seed: int, key, act_mean, act_std, frame_mean, frame_std,
                    termination: str, max_steps: int, sample_fn):
  """정책 롤아웃 1개를 수집한다. 그래디언트 없음(inference만).

  반환: obs_n_all (T, obs_dim) — 각 스텝에서 정책이 실제로 조건으로 쓴 정규화
  관측, xs_all (T, n_steps+1, act_dim), d_vals (T+1,), outcome(str),
  episode_len(T).
  """
  env.reset(seed=seed)
  obs_raw_list = [np.asarray(env._stacked_obs(), dtype=np.float32)]
  obs_n_list = []
  xs_list = []
  outcome = 'running'

  for _ in range(max_steps):
    obs_raw_t = obs_raw_list[-1]
    obs_n_t = (obs_raw_t - frame_mean) / frame_std
    obs_n_list.append(obs_n_t)

    key, sub = jax.random.split(key)
    x_final, xs = sample_fn(params, jnp.asarray(obs_n_t[None]), sub)
    act_n = np.asarray(x_final)[0]
    act = act_n * act_std + act_mean
    xs_list.append(np.asarray(xs)[:, 0, :])          # (n_steps+1, act_dim)

    _, _, term, trunc, info = env.step(act)
    obs_raw_next = np.asarray(env._stacked_obs(), dtype=np.float32)
    obs_raw_list.append(obs_raw_next)
    outcome = info['outcome']

    if termination == 'learned':
      pred_succ = bool(reward.success(obs_raw_next[None])[0])
      stop = pred_succ or term or trunc
    else:  # 'env'
      stop = term or trunc
    if stop:
      break

  obs_raw_all = np.stack(obs_raw_list)                # (T+1, obs_dim)
  d_vals = reward.d(obs_raw_all)                       # (T+1,) — 보상은 항상 d로만
  obs_n_all = np.stack(obs_n_list)                      # (T, obs_dim)
  xs_all = np.stack(xs_list)                            # (T, n_steps+1, act_dim)
  return obs_n_all, xs_all, d_vals, outcome, key


# --------------------------------------------------------------------- 학습

def build_train_step(ddpo, dist_head_keys):
  def loss_fn(params, obs_b, x_in_b, x_out_b, kk_b, R_b, reinforce_scale):
    lp = ddpo.step_logp(params, obs_b, x_in_b, x_out_b, kk_b)      # (B,)
    return -reinforce_scale * jnp.mean(R_b * lp)

  grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=())

  def train_step(params, opt_state, optimizer, obs_b, x_in_b, x_out_b, kk_b,
                R_b, reinforce_scale):
    loss, grads = grad_fn(params, obs_b, x_in_b, x_out_b, kk_b, R_b,
                          reinforce_scale)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads

  return train_step


def _assert_dist_head_frozen(params_before, params_after, dist_head_keys):
  for k in dist_head_keys:
    for pname, arr_after in params_after[k].items():
      arr_before = params_before[k][pname]
      if not np.array_equal(np.asarray(arr_before), np.asarray(arr_after)):
        raise RuntimeError(
            f'STG(dist) 헤드 파라미터 {k}/{pname}가 DDPO 업데이트로 바뀌었다 — '
            f'chain_logp/step_logp 그래디언트가 dist 헤드로 새고 있다는 뜻이다. '
            f'src/ddpo.py의 헤드 분리 로직(_is_dist_head_key)이 이 체크포인트의 '
            f'파라미터 키 구조와 어긋났을 가능성이 높다. 즉시 중단.')


# --------------------------------------------------------------------- 진단

def _load_drift_obs(data_path, frame_mean, frame_std):
  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  frames = data['observation']['frame']
  rng = np.random.default_rng(DRIFT_SEED)
  idx = rng.choice(len(frames), size=min(DRIFT_N_OBS, len(frames)),
                   replace=False)
  obs_raw = frames[idx].astype(np.float32)
  obs_n = (obs_raw - frame_mean) / frame_std
  return jnp.asarray(obs_n)


def _drift_metric(sample_fn, params, base_params, drift_obs_n):
  key = jax.random.PRNGKey(DRIFT_SAMPLE_SEED)
  x_cur, _ = sample_fn(params, drift_obs_n, key)
  x_base, _ = sample_fn(base_params, drift_obs_n, key)
  return float(jnp.mean(jnp.linalg.norm(x_cur - x_base, axis=-1)))


# ------------------------------------------------------------------------ main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--policy-ckpt',
                  default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--d-ckpt', required=True)
  ap.add_argument('--statistic', choices=['mean', 'cvar'], default='mean')
  ap.add_argument('--cvar-alpha', type=float, default=0.8)
  ap.add_argument('--iterations', type=int, default=100)
  ap.add_argument('--episodes-per-iter', type=int, default=32)
  ap.add_argument('--gamma', type=float, default=0.9)
  ap.add_argument('--reinforce-scale', type=float, default=5e-2)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--termination', choices=['learned', 'env'], default='learned')
  ap.add_argument('--advantage-norm', action='store_true', default=False)
  ap.add_argument('--logp-batch', type=int, default=4096)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  if args.seed0 >= 900_000:
    raise ValueError('--seed0가 평가 시드 대역(900000+)과 겹친다 — 학습 시드는 '
                     '그 아래 값을 써라.')

  ck, nets, ddpo, act_dim = load_policy_for_training(args.policy_ckpt)
  params = ck['params']
  base_params = jax.tree.map(np.copy, params)          # 드리프트 비교용 고정 사본
  norm_stats = ck['norm_stats']
  act_mean, act_std = norm_stats['act_mean'], norm_stats['act_std']
  frame_mean, frame_std = norm_stats['frame_mean'], norm_stats['frame_std']
  n_steps = nets.n_steps
  dist_head_keys = _dist_head_keys(params)

  reward = StgReward(args.d_ckpt, statistic=args.statistic,
                     cvar_alpha=args.cvar_alpha)
  # reward.meta['data']를 쓴다(하드코딩된 별도 경로 대신) — 캘리브레이션/드리프트
  # 진단이 항상 이 --d-ckpt가 실제로 학습된 데이터와 일치하게 강제하기 위함.
  # 예전엔 DRIFT_DATA_PATH가 'data/grasp_carry_demos_v3.pkl'로 고정돼 있어서,
  # v4/v5 계열 d-ckpt를 쓸 때도 v3로 캘리브레이션해 문턱이 완전히 어긋났었다
  # (f1=0.151 — --termination learned가 이 잘못된 문턱을 파고들며 붕괴, 2026-08-09).
  calib_data_path = reward.meta['data']
  with open(calib_data_path, 'rb') as fp:
    calib_data = pickle.load(fp)
  val_eps = _val_episode_ids(calib_data, seed=reward.meta['seed'])
  best_s, calib_metrics = calibrate_threshold(reward, calib_data, val_eps)
  reward.threshold = best_s
  print(f'[calib] threshold s={best_s:.3f}  f1={calib_metrics["f1"]:.3f}  '
        f'(data={calib_data_path}, held-out {len(val_eps)} episodes)')

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  optimizer = optax.adam(args.lr)
  opt_state = optimizer.init(params)
  train_step = build_train_step(ddpo, dist_head_keys)
  sample_fn = jax.jit(ddpo.sample_with_trace)

  drift_obs_n = _load_drift_obs(calib_data_path, frame_mean, frame_std)

  rng_key = jax.random.PRNGKey(args.seed0 + 1_000_000)   # 정책 샘플링 전용 스트림
  env_seed_counter = args.seed0

  first_update_checked = False
  n_pairs_per_step = n_steps - 1

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

  for it in range(1, args.iterations + 1):
    t_iter_start = time.time()

    ep_obs_n, ep_xs, ep_R, ep_len, ep_outcome = [], [], [], [], []
    for _ in range(args.episodes_per_iter):
      obs_n_all, xs_all, d_vals, outcome, rng_key = collect_episode(
          nets, ddpo, params, reward, env, env_seed_counter, rng_key,
          act_mean, act_std, frame_mean, frame_std, args.termination,
          cfg.max_steps, sample_fn)
      env_seed_counter += 1

      rewards = compute_step_rewards(d_vals)
      returns = compute_returns(rewards, args.gamma)

      ep_obs_n.append(obs_n_all)
      ep_xs.append(xs_all)
      ep_R.append(returns)
      ep_len.append(len(rewards))
      ep_outcome.append(outcome)

    obs_n_all = np.concatenate(ep_obs_n, axis=0)         # (N, obs_dim)
    xs_all = np.concatenate(ep_xs, axis=0)                # (N, n_steps+1, act_dim)
    R_all = np.concatenate(ep_R, axis=0)                   # (N,)
    N = obs_n_all.shape[0]

    if args.advantage_norm:
      R_train = (R_all - R_all.mean()) / (R_all.std() + 1e-8)
    else:
      R_train = R_all

    # ---- (env-step, kk) 쌍 나열 → 셔플 → 미니배치 순회 (이 배치로 딱 한 번)
    i_in, i_out, kk = make_pair_index_table(n_steps)       # 각 (n_steps-1,)
    j_idx = np.repeat(np.arange(N), n_pairs_per_step)
    p_idx = np.tile(np.arange(n_pairs_per_step), N)
    kk_all = kk[p_idx]
    i_in_all = i_in[p_idx]
    i_out_all = i_out[p_idx]

    perm = np.random.default_rng(args.seed0 + it).permutation(len(j_idx))
    j_idx, i_in_all, i_out_all, kk_all = (a[perm] for a in
                                          (j_idx, i_in_all, i_out_all, kk_all))

    n_total_pairs = len(j_idx)
    losses = []
    for start in range(0, n_total_pairs, args.logp_batch):
      sl = slice(start, start + args.logp_batch)
      bj, bi_in, bi_out, bkk = j_idx[sl], i_in_all[sl], i_out_all[sl], kk_all[sl]

      obs_b = jnp.asarray(obs_n_all[bj])
      x_in_b = jnp.asarray(xs_all[bj, bi_in])
      x_out_b = jnp.asarray(xs_all[bj, bi_out])
      kk_b = jnp.asarray(bkk, dtype=jnp.int32)
      R_b = jnp.asarray(R_train[bj], dtype=jnp.float32)

      if not first_update_checked:
        params_before = jax.tree.map(np.copy, params)

      params, opt_state, loss, _ = train_step(
          params, opt_state, optimizer, obs_b, x_in_b, x_out_b, kk_b, R_b,
          args.reinforce_scale)
      losses.append(float(loss))

      if not first_update_checked:
        _assert_dist_head_frozen(params_before, params, dist_head_keys)
        first_update_checked = True
        print('[check] 첫 업데이트 후 STG(dist) 헤드 파라미터 불변 확인됨.')

    # ---- 로그 -----------------------------------------------------------
    drift = _drift_metric(sample_fn, params, base_params, drift_obs_n)
    n_env_succ = sum(1 for o in ep_outcome if o == 'success')
    total_steps = sum(ep_len)
    demos_per_1k = 1000.0 * n_env_succ / max(total_steps, 1)
    iter_time = time.time() - t_iter_start

    print(f'it={it:4d}  R_mean={R_all.mean():+8.4f}  |R|_mean={np.abs(R_all).mean():7.4f}  '
          f'env_succ_rate={n_env_succ / args.episodes_per_iter:5.1%}  '
          f'ep_len_mean={np.mean(ep_len):6.1f}  demos/1k={demos_per_1k:6.2f}  '
          f'drift_L2={drift:7.4f}  loss={np.mean(losses):+9.5f}  '
          f'time={iter_time:5.1f}s', flush=True)

    if not np.isfinite(R_all).all() or not np.isfinite(drift):
      raise RuntimeError(
          f'it={it}: 리턴 또는 드리프트가 비유한(NaN/inf)이 됐다 — 중단. '
          f'R_all finite={np.isfinite(R_all).all()}  drift={drift}')

  out_ckpt = dict(ck)
  out_ckpt['params'] = params
  out_ckpt['meta'] = dict(ck['meta'])
  out_ckpt['meta'].update(dict(
      si_d_ckpt=args.d_ckpt,
      si_statistic=args.statistic,
      si_gamma=args.gamma,
      si_reinforce_scale=args.reinforce_scale,
      si_lr=args.lr,
      si_iterations=args.iterations,
      si_episodes_per_iter=args.episodes_per_iter,
      si_termination=args.termination,
      si_advantage_norm=args.advantage_norm,
      si_base_ckpt=args.policy_ckpt,
  ))
  with open(args.out, 'wb') as fp:
    pickle.dump(out_ckpt, fp)
  print(f'저장: {args.out}')


if __name__ == '__main__':
  main()
