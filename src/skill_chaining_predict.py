"""스킬 체이닝 2단계 — "갈림길까지 몇 스텝 남았나"를 미리 예측하는 보조 헤드.

1단계(skill_chaining_detect.py)는 반응형이다: 지금 갈림길 안에 있는지만
안다. 2단계는 예측형이다: 갈림길이 곧 풀릴지 미리 내다본다.

핵심 아이디어(사후 라벨링, HER과 동일한 메커니즘): 미리 정해진 목표 위치가
없어도 된다. 정책을 굴려서(재학습 없음, 기존 메인 예측기 그대로) 1단계
탐지기로 각 에피소드의 '갈림길 진입→해소' 구간을 사후에 찾고, 그 구간
안의 각 스텝에 "실제로 해소됐던 시점까지 거꾸로 센 스텝 수"를 라벨로
붙인다 — 원본 STG 라벨(arange(len-1,-1,-1), 실제 성공 시점부터 거꾸로 셈)과
정확히 같은 방식이며, 대상만 '최종 골'에서 '이번 갈림길의 해소 시점'으로
바뀐 것뿐이다. 갈림길 밖의 스텝은 라벨이 정의되지 않으므로 학습에서 뺀다
(마스킹).

보조 헤드는 메인 예측기와 같은 9차원 관측(위치·속도·골·장애물)을 입력으로
쓰는 별도의 작은 네트워크로 학습한다(백본 공유는 프로덕션 최적화이지 개념
검증에 필수는 아니라 생략 — 학습 인프라 재사용을 위해 build_continuous_
act_discrete_dist_v0을 그대로 쓰되 행동 헤드 손실은 무시하고 거리 헤드만
학습한다).

실행:
  python -m src.skill_chaining_predict --episodes 500 --steps 8192
"""

import argparse
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pointmass_core import build_continuous_act_discrete_dist_v0
from src.obstacle_env import ObstacleAvoidPoint2D
from src.probe_generic import GenericSTGProbe
from src.skill_chaining_detect import detect_segments, refine_resolve_point
from src.train_obstacle_predictor import OBS_FIELDS, concat_obs

CKPT_MAIN = 'checkpoints/obstacle/predictor.pkl'
CKPT_AUX = 'checkpoints/obstacle/skill_boundary_predictor.pkl'


# ------------------------------------------------------------------ rollout
def rollout_full(probe, seed, max_steps=300):
  """skill_chaining_detect.rollout과 동일하되 관측 dict 전체도 기록."""
  np.random.seed(seed)
  env = ObstacleAvoidPoint2D()
  ts = env.reset()
  key = jax.random.PRNGKey(seed)
  positions, varis, obs_list = [env._cur_pos.copy()], [], []
  step = 0
  while (not env.success()) and step < max_steps:
    obs = ts.observation
    obs_list.append(obs)
    key, sub = jax.random.split(key)
    act_norm, logits = probe._logits_and_act(obs, sub)
    varis.append(probe._record(step, obs, logits).variance)
    action = probe.unnormalize_action(act_norm)[0]
    ts = env.step(np.asarray(action, dtype=np.float32))
    positions.append(env._cur_pos.copy())
    step += 1
  obs_list.append(ts.observation)
  obs = ts.observation
  key, sub = jax.random.split(key)
  _, logits = probe._logits_and_act(obs, sub)
  varis.append(probe._record(step, obs, logits).variance)
  return (np.array(positions), np.array(varis), obs_list, env._cur_obstacle,
          env._goal_pos.copy(), env.success())


# --------------------------------------------------------- hindsight dataset
def build_dataset(probe, n_episodes, seed0, factor, hysteresis, smooth,
                  min_gap, max_steps=300, refine=True, refine_window=8):
  obs_lists = {k: [] for k in OBS_FIELDS}
  labels = []
  ep_ids = []
  n_forks = 0
  shifts = []  # 정밀화로 경계가 원래 지점에서 얼마나 이동했는지(진단용)
  for k in range(n_episodes):
    pos, varis, obs_list, obst, goal, succ = rollout_full(
        probe, seed0 + k, max_steps)
    if len(pos) < 10:
      continue
    boundaries, _, entries = detect_segments(varis, factor, smooth,
                                              hysteresis, min_gap)
    for entry, resolve in zip(entries, boundaries):
      if refine:
        resolve_r = refine_resolve_point(varis, resolve, refine_window)
        resolve_r = max(resolve_r, entry + 1)  # entry보다 앞으로 못 감
        shifts.append(resolve_r - resolve)
        resolve = resolve_r
      n_forks += 1
      for t in range(entry, resolve):
        for f in OBS_FIELDS:
          obs_lists[f].append(obs_list[t][f])
        labels.append(resolve - t)  # 원본 time_to_success와 동일한 방식
        ep_ids.append(k)
  if refine and shifts:
    shifts = np.array(shifts)
    print(f'경계 정밀화: 평균 이동 {shifts.mean():+.1f} step '
          f'(표준편차 {shifts.std():.1f}, |이동|>=1인 비율 '
          f'{(np.abs(shifts) >= 1).mean():.0%})')

  data = dict(
      observation={f: np.stack(v).astype(np.float32)
                   for f, v in obs_lists.items()},
      label=np.array(labels, dtype=np.float32),
      episode_id=np.array(ep_ids, dtype=np.int32),
  )
  print(f'{n_episodes}개 에피소드 롤아웃 -> 갈림길 {n_forks}개 -> '
        f'학습 샘플 {len(labels)}개')
  return data


# ------------------------------------------------------------------ training
def train_aux(data, steps, batch, lr, seed, max_label):
  num_bins = int(max_label)  # bin_size=1
  bin_vals = np.arange(num_bins, dtype=np.float32)

  rng = np.random.default_rng(seed)
  n_ep = int(data['episode_id'].max()) + 1
  val_eps = set(rng.choice(n_ep, size=max(n_ep // 5, 1), replace=False)
               .tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(data['observation']))
  labels = np.clip(data['label'], 0, num_bins - 1)
  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_lab = jnp.asarray(labels[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_lab = jnp.asarray(labels[val_mask])
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} (val {len(val_eps)} eps)')

  nets = build_continuous_act_discrete_dist_v0(
      (128, 128), 2, num_bins, np.ones((4, obs_c.shape[-1]), dtype=np.float32))
  optimizer = optax.adam(lr)
  key = jax.random.PRNGKey(seed)
  key, sub = jax.random.split(key)
  params = nets.network.init(sub)
  opt_state = optimizer.init(params)

  def loss_fn(params, bo, bl):
    preds = nets.network.apply(params, bo)
    return -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, bl))

  @jax.jit
  def step_fn(params, opt_state, key):
    idx = jax.random.randint(key, (batch,), 0, tr_obs.shape[0])
    l, g = jax.value_and_grad(loss_fn)(params, tr_obs[idx], tr_lab[idx])
    updates, opt_state = optimizer.update(g, opt_state)
    return optax.apply_updates(params, updates), opt_state, l

  @jax.jit
  def val_mae(params):
    preds = nets.network.apply(params, va_obs)
    p = jax.nn.softmax(preds.dist_to_succ_dist_params.logits, axis=-1)
    exp = jnp.sum(p * bin_vals[None, :], axis=-1)
    return jnp.mean(jnp.abs(exp - va_lab))

  for i in range(1, steps + 1):
    key, sub = jax.random.split(key)
    params, opt_state, l = step_fn(params, opt_state, sub)
    if i % 2048 == 0 or i == 1:
      print(f'  step {i:5d}  loss={float(l):.3f}  val MAE={float(val_mae(params)):.2f}')
  return params, nets, bin_vals, float(val_mae(params))


# --------------------------------------------------------------------- demo
def demo_figure(probe, aux_params, aux_nets, bin_vals, out_path, seeds,
                factor, hysteresis, smooth, min_gap, max_steps=300):
  @jax.jit
  def aux_predict(params, concat):
    preds = aux_nets.network.apply(params, concat)
    p = jax.nn.softmax(preds.dist_to_succ_dist_params.logits, axis=-1)
    return jnp.sum(p * bin_vals[None, :], axis=-1)

  n = len(seeds)
  fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
  if n == 1:
    axes = [axes]
  for ax, seed in zip(axes, seeds):
    pos, varis, obs_list, obst, goal, succ = rollout_full(probe, seed, max_steps)
    boundaries, thresh, entries = detect_segments(
        varis, factor, smooth, hysteresis, min_gap)
    boundaries = [max(refine_resolve_point(varis, b), e + 1)
                 for b, e in zip(boundaries, entries)]
    T = len(obs_list)
    concats = np.asarray(concat_obs({f: np.stack([o[f] for o in obs_list])
                                     for f in OBS_FIELDS}))
    pred = np.asarray(aux_predict(aux_params, jnp.asarray(concats)))

    # 실제 '다음 해소까지 남은 스텝' (사후, 갈림길 구간에서만 정의)
    true_remaining = np.full(T, np.nan)
    for entry, resolve in zip(entries, boundaries):
      for t in range(entry, resolve):
        true_remaining[t] = resolve - t

    ax.plot(pred, color='C0', lw=1.6, label='예측(2단계 보조헤드)')
    ax.plot(true_remaining, color='black', lw=1.8, ls='-',
            label='실제 남은 스텝(사후, 갈림길 구간만)')
    for b in boundaries:
      ax.axvline(b, color='red', ls='--', lw=1.0, alpha=0.7)
    ax.set_xlabel('step'); ax.set_ylabel('갈림길 해소까지 남은 스텝(예측)')
    ax.set_title(f'seed {seed} ({"성공" if succ else "실패"}) — '
                 f'빨강 점선=1단계가 찾은 실제 해소 시점', fontsize=9)
    ax.legend(fontsize=7)
  fig.suptitle('스킬 체이닝 2단계 — 갈림길 해소를 미리 내다보는 보조 헤드',
               fontsize=13)
  fig.tight_layout()
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(out_path, dpi=130)
  plt.close(fig)
  print(f'-> {out_path}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT_MAIN)
  ap.add_argument('--episodes', type=int, default=500)
  ap.add_argument('--factor', type=float, default=1.4)
  ap.add_argument('--hysteresis', type=float, default=0.85)
  ap.add_argument('--smooth', type=int, default=3)
  ap.add_argument('--min-gap', type=int, default=4)
  ap.add_argument('--steps', type=int, default=8192)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='results/obstacle_env/skill_chaining_tier2.png')
  args = ap.parse_args()

  probe = GenericSTGProbe(args.checkpoint)

  # 체크포인트 종류(노이즈/깨끗함)별로 사후 라벨 데이터셋·보조헤드 경로를 분리
  # -- 안 그러면 서로 다른 --checkpoint 실행이 같은 파일을 덮어쓴다.
  suffix = '_clean' if 'clean' in args.checkpoint else ''
  data_path = f'data/skill_boundary_hindsight{suffix}.pkl'
  ckpt_aux = f'checkpoints/obstacle{suffix}/skill_boundary_predictor.pkl'
  if os.path.exists(data_path):
    print(f'기존 사후 라벨 데이터셋 재사용: {data_path}')
    with open(data_path, 'rb') as fp:
      data = pickle.load(fp)
  else:
    data = build_dataset(probe, args.episodes, args.seed, args.factor,
                         args.hysteresis, args.smooth, args.min_gap)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as fp:
      pickle.dump(data, fp)

  labels = data['label']
  print(f'라벨 분포: 중앙값={np.median(labels):.0f} p90={np.percentile(labels,90):.0f} '
        f'p99={np.percentile(labels,99):.0f} max={labels.max():.0f}')
  max_label = min(100, int(np.percentile(labels, 99.5)) + 5)

  params, nets, bin_vals, val_mae = train_aux(
      data, args.steps, args.batch, args.lr, args.seed, max_label)
  print(f'\n최종 val MAE: {val_mae:.2f} (라벨 범위 0~{max_label})')

  os.makedirs(os.path.dirname(ckpt_aux), exist_ok=True)
  with open(ckpt_aux, 'wb') as fp:
    pickle.dump({'params': jax.device_get(params), 'max_label': max_label,
                'val_mae': val_mae}, fp)
  print(f'보조 헤드 체크포인트 저장: {ckpt_aux}')

  demo_figure(probe, params, nets, bin_vals, args.out,
             seeds=[1, 2, 5, 8], factor=args.factor,
             hysteresis=args.hysteresis, smooth=args.smooth,
             min_gap=args.min_gap)


if __name__ == '__main__':
  main()
