"""2-장애물 게이트 시나리오에서도 불확실성이 진짜 갈림길 신호인지 검증.

exp_reducible_uncertainty.py와 같은 방법(앙상블 전분산 분해)을 12차원
관측(장애물 2개)에 적용한다. 핵심 질문: 게이트(두 장애물 사이 좁은 틈)에서
- 막힌 두 방향(각 장애물 뒤)에서만 뜨겁고 틈 자체는 차가운가(단일 장애물과
  같은 패턴)?
- 아니면 틈을 정밀하게 조준해야 하는 것 자체가 별도의 불확실성을 만드는가?

--mode gate  : 게이트 장면 2D 지도 + 기하학적 차단(둘 중 하나라도 막힘) 겹침
--mode ring  : 게이트를 가로지르는 직선을 따라(문설주 사이) 단면 스캔
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
import matplotlib.patches as patches

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pointmass_core import build_continuous_act_discrete_dist_v0
from src.train_obstacle_predictor import compute_stats, make_normalizers_obstacle
from src.train_two_obstacle_predictor import (
    DATA_PATH, MAX_DISTANCE, NUM_BINS, OBS_FIELDS, concat_obs2)
from src.exp_reducible_uncertainty import is_blocked_point, deco as deco1

CACHE_PATH = 'results/obstacle_env/ensemble_cache_two.pkl'
BIN_VALS = np.linspace(0, MAX_DISTANCE, NUM_BINS + 1, dtype=np.float32)[:-1]


# --------------------------------------------------------------- ensemble
def train_subset(data, stats, frac, steps, batch, lr, seed):
  normalize_obs, normalize_action, _ = make_normalizers_obstacle(stats)
  n_ep = int(data['episode_id'].max()) + 1
  rng = np.random.default_rng(seed)
  keep = set(rng.choice(n_ep, size=max(int(n_ep * frac), 1),
                        replace=False).tolist())
  mask = np.isin(data['episode_id'], list(keep))
  obs_c = jnp.asarray(np.asarray(concat_obs2(normalize_obs(
      data['observation'])))[mask])
  act_n = jnp.asarray(np.asarray(normalize_action(data['action']))[mask])
  ttg = jnp.asarray(data['time_to_success'][mask])

  nets = build_continuous_act_discrete_dist_v0(
      (256, 256, 256), 2, NUM_BINS,
      np.ones((4, obs_c.shape[-1]), dtype=np.float32))
  optimizer = optax.adam(lr)
  key = jax.random.PRNGKey(seed)
  key, sub = jax.random.split(key)
  params = nets.network.init(sub)
  opt_state = optimizer.init(params)

  def loss_fn(params, bo, ba, bt):
    preds = nets.network.apply(params, bo)
    bc = -jnp.mean(nets.act_log_prob(preds.act_dist_params, ba))
    dl = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, bt))
    return bc + dl

  @jax.jit
  def step(params, opt_state, key):
    idx = jax.random.randint(key, (batch,), 0, obs_c.shape[0])
    g = jax.grad(loss_fn)(params, obs_c[idx], act_n[idx], ttg[idx])
    updates, opt_state = optimizer.update(g, opt_state)
    return optax.apply_updates(params, updates), opt_state

  for _ in range(steps):
    key, sub = jax.random.split(key)
    params, opt_state = step(params, opt_state, sub)
  print(f'  seed={seed}: {obs_c.shape[0]} transitions ({len(keep)} eps) 학습완료')
  return params, nets


def get_ensemble(n_models, frac, steps, batch, lr, cache_path, refresh=False):
  if os.path.exists(cache_path) and not refresh:
    with open(cache_path, 'rb') as fp:
      cached = pickle.load(fp)
    if cached['n_models'] == n_models and cached['frac'] == frac:
      print(f'앙상블 캐시 사용: {cache_path}')
      with open(DATA_PATH, 'rb') as fp:
        data = pickle.load(fp)
      _, nets = train_subset(data, compute_stats(data), frac, 0, batch, lr, 0)
      return [jax.tree.map(jnp.asarray, p) for p in cached['params']], nets

  with open(DATA_PATH, 'rb') as fp:
    data = pickle.load(fp)
  stats = compute_stats(data)
  print(f'{n_models}개 모델 학습 (각 데이터 {int(frac*100)}%)...')
  models, nets = [], None
  for k in range(n_models):
    p, nets = train_subset(data, stats, frac, steps, batch, lr, seed=200 + k)
    models.append(p)
  os.makedirs(os.path.dirname(cache_path), exist_ok=True)
  with open(cache_path, 'wb') as fp:
    pickle.dump({'n_models': n_models, 'frac': frac,
                'params': [jax.device_get(p) for p in models]}, fp)
  return models, nets


def make_mu_var_fn(nets):
  bins = jnp.asarray(BIN_VALS)

  def mu_var(params, concats):
    preds = nets.network.apply(params, concats)
    p = jax.nn.softmax(preds.dist_to_succ_dist_params.logits, axis=-1)
    mu = jnp.sum(bins * p, axis=-1)
    var = jnp.sum(bins ** 2 * p, axis=-1) - mu ** 2
    return mu, var

  return jax.jit(mu_var, backend='cpu')


# ------------------------------------------------------------------ scene
def make_obs(pos, goal, obstacles, speed):
  """obstacles: [(center, radius), ...] 정확히 2개, 전달된 순서 그대로 사용.

  2026-07-20: 예전엔 거리순 정렬(학습 시 매 스텝 재정렬과 동일 규칙)을
  재현했으나, 그 매 스텝 재정렬 자체가 등거리선에서 인위적 불연속을 만드는
  아티팩트로 확인되어 환경 쪽을 '배치시 고정 순서'로 고쳤다. 여기서도 같은
  고정 순서를 그대로 따라야 하므로 정렬을 제거한다 — 호출자가 넘긴 순서가
  곧 '배치 순서'다.
  """
  d = goal - pos
  nrm = np.linalg.norm(d)
  vel = (speed * d / nrm if nrm > 1e-6 else np.zeros(2)).astype(np.float32)
  rel = np.concatenate([(c - pos).astype(np.float32) for c, _ in obstacles])
  radii = np.array([r for _, r in obstacles], dtype=np.float32)
  return {'cur_pos': pos.astype(np.float32), 'cur_vel': vel, 'goal_pos': goal,
          'obstacle_rel_pos': rel, 'obstacle_radius': radii}


def eval_scene(models, mu_var_fn, normalize_obs, goal, obstacles, speed, grid):
  xs = np.linspace(-1, 1, grid, dtype=np.float32)
  concats = []
  for y_ in xs:
    for x_ in xs:
      obs = make_obs(np.array([x_, y_], dtype=np.float32), goal, obstacles,
                     speed)
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
      concats.append(np.asarray(concat_obs2(norm))[0])
  C = jnp.asarray(np.stack(concats))
  mus, vars_ = [], []
  for p in models:
    mu, var = mu_var_fn(p, C)
    mus.append(np.asarray(mu).reshape(grid, grid))
    vars_.append(np.asarray(var).reshape(grid, grid))
  mus = np.stack(mus); vars_ = np.stack(vars_)
  return mus.std(0), np.sqrt(vars_.mean(0)), xs


def blocked_by_either(xs, goal, obstacles):
  gs = len(xs)
  mask = np.zeros((gs, gs), dtype=bool)
  for i, y_ in enumerate(xs):
    for j, x_ in enumerate(xs):
      p = np.array([x_, y_], dtype=np.float32)
      mask[i, j] = any(is_blocked_point(p, goal, c, r) for c, r in obstacles)
  return mask


def deco2(ax, obstacles, goal):
  for c, r in obstacles:
    ax.add_patch(patches.Circle(c, r, facecolor='none', edgecolor='white',
                                lw=1.6))
  ax.scatter(*goal, marker='*', s=150, color='lime', edgecolor='k', zorder=5,
             linewidths=0.6)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--mode', choices=['gate', 'ring'], default='gate')
  ap.add_argument('--models', type=int, default=4)
  ap.add_argument('--frac', type=float, default=1.0)
  ap.add_argument('--steps', type=int, default=32768)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--grid', type=int, default=61)
  ap.add_argument('--refresh-cache', action='store_true')
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  models, nets = get_ensemble(args.models, args.frac, args.steps, args.batch,
                              args.lr, CACHE_PATH, args.refresh_cache)
  with open(DATA_PATH, 'rb') as fp:
    data = pickle.load(fp)
  normalize_obs, _, _ = make_normalizers_obstacle(compute_stats(data))
  mu_var_fn = make_mu_var_fn(nets)
  gs = args.grid

  # ---- 게이트 장면: 두 장애물이 (0,0) 좌우로 마주보며 통과 폭 0.3짜리 문
  goal = np.array([0.6, 0.6], dtype=np.float32)
  gap = 0.28
  r0 = r1 = 0.16
  obstacles = [(np.array([0.0, r0 + gap / 2], np.float32), r0),
               (np.array([0.0, -(r1 + gap / 2)], np.float32), r1)]

  if args.mode == 'gate':
    epistemic, aleatoric, xs = eval_scene(models, mu_var_fn, normalize_obs,
                                          goal, obstacles, 0.006, gs)
    blocked = blocked_by_either(xs, goal, obstacles)
    out = args.out or 'results/obstacle_env/uncertainty_two_gate.png'
    ext = (-1, 1, -1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    vmax_ep = float(np.percentile(epistemic, 99.5))
    vmax_al = float(np.percentile(aleatoric, 99.5))
    im0 = axes[0].imshow(epistemic, origin='lower', extent=ext, cmap='viridis',
                         vmin=0, vmax=vmax_ep)
    axes[0].set_title('Epistemic uncertainty (모델 간 불일치)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label='steps')
    deco2(axes[0], obstacles, goal)
    im1 = axes[1].imshow(aleatoric, origin='lower', extent=ext, cmap='magma',
                         vmin=0, vmax=vmax_al)
    axes[1].contour(xs, xs, blocked.astype(float), levels=[0.5], colors='cyan',
                    linewidths=1.8, linestyles='--')
    axes[1].set_title('Aleatoric uncertainty + 차단 영역(둘 중 하나)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label='steps')
    deco2(axes[1], obstacles, goal)
    fig.suptitle(f'2-장애물 게이트 (통과 폭 {gap:.2f}) 불확실성 분해', fontsize=13)
    fig.tight_layout()
    out_dir = os.path.dirname(out)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    inside = aleatoric[blocked].mean()
    outside = aleatoric[~blocked].mean()
    # 게이트 통로(두 원 사이, y in [-gap/2,gap/2] 근방, x는 원 중심 부근) 표본
    yy, xx = np.meshgrid(xs, xs, indexing='ij')
    corridor = (np.abs(yy) < gap / 2 - 0.02) & (np.abs(xx) < 0.15)
    print(f'차단영역(둘 중 하나) 내부 aleatoric 평균: {inside:.2f}')
    print(f'차단영역 외부 aleatoric 평균: {outside:.2f}')
    print(f'비율: {inside/outside:.2f}x')
    if corridor.sum() > 0:
      print(f'게이트 통로(뚫린 좁은 틈) aleatoric 평균: '
            f'{aleatoric[corridor].mean():.2f}  ({corridor.sum()} 격자점)')
    print(f'-> {out}')
    return

  # ---- ring: 게이트를 가로지르는 x=0 라인을 따라 y를 스캔하는 단면
  ys = np.linspace(-0.9, 0.9, 181, dtype=np.float32)
  concats = []
  for y_ in ys:
    obs = make_obs(np.array([0.0, y_], dtype=np.float32), goal, obstacles,
                   0.006)
    norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
    concats.append(np.asarray(concat_obs2(norm))[0])
  C = jnp.asarray(np.stack(concats))
  mus, vars_ = [], []
  for p in models:
    mu, var = mu_var_fn(p, C)
    mus.append(np.asarray(mu)); vars_.append(np.asarray(var))
  mus = np.stack(mus); vars_ = np.stack(vars_)
  aleatoric = np.sqrt(vars_.mean(0))
  epistemic = mus.std(0)

  out = args.out or 'results/obstacle_env/uncertainty_two_gate_crosssection.png'
  fig, ax = plt.subplots(figsize=(9, 4.3))
  ax2 = ax.twinx()
  l1, = ax.plot(ys, aleatoric, color='C3', lw=1.6, label='Aleatoric')
  l2, = ax2.plot(ys, epistemic, color='C0', lw=1.6, ls='--', label='Epistemic')
  for c, r in obstacles:
    ax.axvspan(c[1] - r, c[1] + r, color='gray', alpha=0.3)
  ax.set_xlabel('y (x=0 라인을 따라, 회색=장애물 점유 구간)')
  ax.set_ylabel('Aleatoric (steps)', color='C3')
  ax2.set_ylabel('Epistemic (steps)', color='C0')
  ax.set_title(f'게이트를 가로지르는 단면 (통과 폭 {gap:.2f}, x=0)')
  ax.legend(handles=[l1, l2], fontsize=9, loc='upper right')
  fig.tight_layout()
  fig.savefig(out, dpi=130)
  plt.close(fig)
  print(f'-> {out}')


if __name__ == '__main__':
  main()
