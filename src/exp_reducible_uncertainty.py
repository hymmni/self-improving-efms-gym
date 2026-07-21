"""고칠 수 있는 헷갈림 vs 타고난 헷갈림 — 모델 간 불일치로 가르기.

"데이터를 늘리면 σ²가 줄어드나"는 틀린 접근이다: 노이즈 있는 데이터에선 잘
배운 모델일수록 σ²가 오히려 커진다(진짜 spread를 정직하게 표현). 그래서 σ²
'크기'가 아니라 모델 간 '불일치'를 본다 (전분산 분해 = 앙상블 표준 도구):

  서로 다른 데이터 조각으로 학습한 모델 K개를 만들고, 각 상태에서
    - epistemic(고칠 수 있는 헷갈림) = 모델들의 기댓값 μ_k가 서로 갈리는 정도
      = Var_k(μ_k). 데이터 더 주면 사라짐.
    - aleatoric(타고난 헷갈림)       = 모델들이 공통으로 갖는 분포 spread
      = mean_k(σ²_k). 데이터로 안 줄어듦.

모델 4개는 랜덤 장애물/골 위치로 학습됐으므로(ObstacleAvoidPoint2D), 학습은
한 번만 하고 캐시해 재사용한다 -- 여러 '장면'(goal/obstacle 배치)에 대한
평가는 순전히 순전파(forward pass)라 저렴하다.

모드:
  --mode normal   : 기본 장면 (원본 reducible_map.png 재현)
  --mode ood      : 속도만 OOD로 바꾼 통제 비교
  --mode verify   : aleatoric 핫존이 '기하학적으로 진짜 막힌 영역'과 겹치는지
                    검증 -- 각 격자점에서 (그 점 -> 목표) 직선이 장애물에
                    막히는지 정확히 계산해 등고선으로 겹쳐 그린다
  --mode scenes   : 장애물이 막는 경우 / 안 막는 경우(구석으로 이동) / 없는
                    경우(반경 0) / 골이 반대편인 경우를 나란히 비교
  --mode datasweep: epistemic 정의(=데이터로 줄어드는 불확실성) 자체를 검증.
                    데이터량 5%/20%/50%/100%에서 각각 앙상블 4개를 새로 학습해
                    같은 장면을 평가 -- epistemic은 데이터가 늘수록 줄어들어야
                    하고 aleatoric은 거의 그대로여야 한다.
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
from src.train_obstacle_predictor import (
    DATA_PATH, OBS_FIELDS, MAX_DISTANCE, NUM_BINS,
    compute_stats, make_normalizers_obstacle, concat_obs)

BIN_VALS = np.linspace(0, MAX_DISTANCE, NUM_BINS + 1, dtype=np.float32)[:-1]
CACHE_PATH = 'results/obstacle_env/ensemble_cache.pkl'


# --------------------------------------------------------------- ensemble
def train_subset(data, stats, frac, steps, batch, lr, seed):
  """에피소드의 frac 비율(seed로 무작위 선택)로 학습. 정규화는 전체 통계 공유."""
  normalize_obs, normalize_action, _ = make_normalizers_obstacle(stats)
  n_ep = int(data['episode_id'].max()) + 1
  rng = np.random.default_rng(seed)
  keep = set(rng.choice(n_ep, size=max(int(n_ep * frac), 1),
                        replace=False).tolist())
  mask = np.isin(data['episode_id'], list(keep))
  obs_c = jnp.asarray(np.asarray(concat_obs(normalize_obs(
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


def get_ensemble(n_models, frac, steps, batch, lr, cache_path, refresh=False,
                 data_path=DATA_PATH):
  """앙상블을 학습하거나 캐시에서 로드. 장면 평가는 순전파뿐이라 재사용이 싸다."""
  if os.path.exists(cache_path) and not refresh:
    with open(cache_path, 'rb') as fp:
      cached = pickle.load(fp)
    if cached['n_models'] == n_models and cached['frac'] == frac:
      print(f'앙상블 캐시 사용: {cache_path}')
      with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
      _, nets = train_subset(data, compute_stats(data), frac, 0, batch, lr, 0)
      return [jax.tree.map(jnp.asarray, p) for p in cached['params']], nets

  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  stats = compute_stats(data)
  print(f'{n_models}개 모델 학습 (각 데이터 {int(frac*100)}%)...')
  models, nets = [], None
  for k in range(n_models):
    p, nets = train_subset(data, stats, frac, steps, batch, lr, seed=100 + k)
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
def eval_scene(models, nets, normalize_obs, mu_var_fn, goal, obst_c, obst_r,
               speed, grid):
  """격자 전 위치에서 '거기 있고 목표를 향해 속력 speed로 움직인다면'을 물어,
  모델별 (mu, var) 지도를 쌓아 반환한다."""
  xs = np.linspace(-1, 1, grid, dtype=np.float32)
  concats = []
  for y_ in xs:
    for x_ in xs:
      pos = np.array([x_, y_], dtype=np.float32)
      d = goal - pos
      nrm = np.linalg.norm(d)
      vel = (speed * d / nrm if nrm > 1e-6 else np.zeros(2)).astype(np.float32)
      obs = {'cur_pos': pos, 'cur_vel': vel, 'goal_pos': goal,
             'obstacle_rel_pos': (obst_c - pos).astype(np.float32),
             'obstacle_radius': np.array([obst_r], dtype=np.float32)}
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
      concats.append(np.asarray(concat_obs(norm))[0])
  C = jnp.asarray(np.stack(concats))

  mus, vars_ = [], []
  for p in models:
    mu, var = mu_var_fn(p, C)
    mus.append(np.asarray(mu).reshape(grid, grid))
    vars_.append(np.asarray(var).reshape(grid, grid))
  mus = np.stack(mus); vars_ = np.stack(vars_)
  epistemic = mus.std(0)
  aleatoric = np.sqrt(vars_.mean(0))
  return epistemic, aleatoric, xs


def is_blocked_point(p, goal, obst_c, obst_r):
  """점 p에서 선분 p->goal 이 원(obst_c, obst_r)과 실제로 교차하는지."""
  seg = goal - p
  L = np.linalg.norm(seg)
  if L < 1e-6:
    return False
  u = seg / L
  t = np.clip(np.dot(obst_c - p, u), 0.0, L)
  closest = p + t * u
  return bool(np.linalg.norm(closest - obst_c) < obst_r)


def blocked_mask(xs, goal, obst_c, obst_r):
  """각 격자점 p에 대해 선분 p->goal 이 원(obst_c, obst_r)과 실제로
  교차하는지(=직선 경로가 장애물에 막혀 진짜로 돌아가야 하는지) 계산."""
  gs = len(xs)
  mask = np.zeros((gs, gs), dtype=bool)
  for i, y_ in enumerate(xs):
    for j, x_ in enumerate(xs):
      mask[i, j] = is_blocked_point(
          np.array([x_, y_], dtype=np.float32), goal, obst_c, obst_r)
  return mask


def deco(ax, obst_c, obst_r, goal, radius_eps=1e-6):
  if obst_r > radius_eps:
    ax.add_patch(patches.Circle(obst_c, obst_r, facecolor='none',
                                edgecolor='white', lw=1.6))
  ax.scatter(*goal, marker='*', s=150, color='lime', edgecolor='k', zorder=5,
             linewidths=0.6)


# --------------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--mode', choices=['normal', 'ood', 'verify', 'scenes',
                                     'datasweep', 'oblique'],
                  default='normal')
  ap.add_argument('--controller', choices=['pf', 'tangent'], default='pf',
                  help='pf=기존 노이즈 데모, tangent=노이즈 없는(혹은 '
                       '--tangent-noise-std로 소량 노이즈만 더한) 접선점 조준'
                       ' 데모(data/cache/출력 모두 접미사로 분리)')
  ap.add_argument('--tangent-noise-std', type=float, default=0.0,
                  help='controller=tangent일 때, train_obstacle_predictor.py'
                       ' --tangent-noise-std로 만든 소량-노이즈 데이터셋을'
                       ' 가리키려면 그때 쓴 값과 동일하게 지정'
                       ' (data/obstacle_demos_clean_n{값}.pkl)')
  ap.add_argument('--fracs', type=float, nargs='+',
                  default=[0.05, 0.2, 0.5, 1.0],
                  help='datasweep 모드에서 비교할 데이터 비율들')
  ap.add_argument('--models', type=int, default=4)
  ap.add_argument('--frac', type=float, default=0.2,
                  help='모델 1개당 학습에 쓰는 에피소드 비율')
  ap.add_argument('--steps', type=int, default=32768)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--grid', type=int, default=61)
  ap.add_argument('--refresh-cache', action='store_true')
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  if args.controller == 'pf':
    suffix = ''
  elif args.tangent_noise_std > 0:
    suffix = f'_clean_n{args.tangent_noise_std:g}'
  else:
    suffix = '_clean'
  data_path = DATA_PATH if suffix == '' else DATA_PATH.replace(
      '.pkl', f'{suffix}.pkl')

  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  normalize_obs, _, _ = make_normalizers_obstacle(compute_stats(data))
  gs = args.grid

  if args.mode == 'datasweep':
    goal = np.array([0.6, 0.6], dtype=np.float32)
    obst_c = np.array([0.0, 0.0], dtype=np.float32)
    obst_r = 0.2
    ep_maps, al_maps = [], []
    for frac in args.fracs:
      cache = f'results/obstacle_env/ensemble_cache_frac{frac:g}{suffix}.pkl'
      print(f'\n--- frac={frac:g} ---')
      models, nets = get_ensemble(args.models, frac, args.steps, args.batch,
                                  args.lr, cache, args.refresh_cache,
                                  data_path=data_path)
      mu_var_fn = make_mu_var_fn(nets)
      epistemic, aleatoric, xs = eval_scene(models, nets, normalize_obs,
                                            mu_var_fn, goal, obst_c, obst_r,
                                            0.006, gs)
      ep_maps.append(epistemic); al_maps.append(aleatoric)
      print(f'  epistemic 평균={epistemic.mean():.2f}  '
            f'aleatoric 평균={aleatoric.mean():.2f}')

    out = args.out or f'results/obstacle_env/uncertainty_datasweep{suffix}.png'
    ext = (-1, 1, -1, 1)
    vmax_ep = float(np.percentile(np.stack(ep_maps), 99.5))
    vmax_al = float(np.percentile(np.stack(al_maps), 99.5))
    n = len(args.fracs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.4))
    for j, frac in enumerate(args.fracs):
      im0 = axes[0, j].imshow(ep_maps[j], origin='lower', extent=ext,
                              cmap='viridis', vmin=0, vmax=vmax_ep)
      axes[0, j].set_title(f'data {int(frac*100)}%\nepistemic 평균='
                           f'{ep_maps[j].mean():.1f}', fontsize=10)
      deco(axes[0, j], obst_c, obst_r, goal)
      im1 = axes[1, j].imshow(al_maps[j], origin='lower', extent=ext,
                              cmap='magma', vmin=0, vmax=vmax_al)
      axes[1, j].set_title(f'aleatoric 평균={al_maps[j].mean():.1f}',
                           fontsize=10)
      deco(axes[1, j], obst_c, obst_r, goal)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                label='epistemic (steps)')
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                label='aleatoric (steps)')
    fig.suptitle('데이터량을 늘리면 epistemic은 줄고 aleatoric은 그대로인가 '
                 '(위: epistemic, 아래: aleatoric, 모델 4개씩)', fontsize=13)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print('\n[요약] frac별 평균')
    for frac, ep, al in zip(args.fracs, ep_maps, al_maps):
      print(f'  {frac:>5.2f}: epistemic={ep.mean():6.2f}  '
            f'aleatoric={al.mean():6.2f}')
    ep_ratio = ep_maps[0].mean() / ep_maps[-1].mean()
    al_ratio = al_maps[0].mean() / al_maps[-1].mean()
    print(f'\nepistemic 감소 배율(최소데이터/최대데이터) = {ep_ratio:.2f}x')
    print(f'aleatoric 감소 배율(최소데이터/최대데이터)  = {al_ratio:.2f}x')
    print(f'-> {out}')
    return

  # frac별로 캐시 파일을 분리 -- datasweep이 만든 frac=1.0 등의 캐시를 그대로
  # 재사용하고, 서로 다른 frac 실행이 같은 캐시 파일을 덮어쓰지 않게 한다.
  cache_path = (CACHE_PATH if (args.frac == 0.2 and suffix == '') else
               f'results/obstacle_env/ensemble_cache_frac{args.frac:g}{suffix}.pkl')
  models, nets = get_ensemble(args.models, args.frac, args.steps, args.batch,
                              args.lr, cache_path, args.refresh_cache,
                              data_path=data_path)
  mu_var_fn = make_mu_var_fn(nets)

  if args.mode in ('normal', 'ood'):
    goal = np.array([0.6, 0.6], dtype=np.float32)
    obst_c = np.array([0.0, 0.0], dtype=np.float32)
    obst_r = 0.2
    speed = 0.05 if args.mode == 'ood' else 0.006
    epistemic, aleatoric, xs = eval_scene(models, nets, normalize_obs,
                                          mu_var_fn, goal, obst_c, obst_r,
                                          speed, gs)
    out = args.out or f'results/obstacle_env/uncertainty_{args.mode}{suffix}.png'
    ext = (-1, 1, -1, 1)
    vmax_ep = float(np.percentile(epistemic, 99.5))
    vmax_al = float(np.percentile(aleatoric, 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    im0 = axes[0].imshow(epistemic, origin='lower', extent=ext, cmap='viridis',
                         vmin=0, vmax=vmax_ep)
    axes[0].set_title('Epistemic uncertainty (모델 간 불일치)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046,
                 label='모델 간 E[STG] 표준편차 (steps)')
    deco(axes[0], obst_c, obst_r, goal)
    im1 = axes[1].imshow(aleatoric, origin='lower', extent=ext, cmap='magma',
                         vmin=0, vmax=vmax_al)
    axes[1].set_title('Aleatoric uncertainty (모델 내 분포 폭 평균)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046,
                 label='STG 분포 표준편차 √mean(σ²) (steps)')
    deco(axes[1], obst_c, obst_r, goal)
    scene_label = ('OOD 상태 (속도=0.05, 학습 최대치의 ~3배)' if args.mode == 'ood'
                   else '정상 상태 (속도=0.006)')
    fig.suptitle(f'Steps-to-go 불확실성 분해 — {scene_label}', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'[전체 평균] epistemic={epistemic.mean():.2f}  '
          f'aleatoric={aleatoric.mean():.2f}')
    print(f'-> {out}')
    return

  if args.mode == 'verify':
    goal = np.array([0.6, 0.6], dtype=np.float32)
    obst_c = np.array([0.0, 0.0], dtype=np.float32)
    obst_r = 0.2
    epistemic, aleatoric, xs = eval_scene(models, nets, normalize_obs,
                                          mu_var_fn, goal, obst_c, obst_r,
                                          0.006, gs)
    blocked = blocked_mask(xs, goal, obst_c, obst_r)
    out = args.out or f'results/obstacle_env/uncertainty_verify{suffix}.png'
    ext = (-1, 1, -1, 1)

    fig, ax = plt.subplots(figsize=(7, 6.4))
    vmax_al = float(np.percentile(aleatoric, 99.5))
    im = ax.imshow(aleatoric, origin='lower', extent=ext, cmap='magma',
                   vmin=0, vmax=vmax_al)
    ax.contour(xs, xs, blocked.astype(float), levels=[0.5], colors='cyan',
              linewidths=2.2, linestyles='--')
    deco(ax, obst_c, obst_r, goal)
    fig.colorbar(im, ax=ax, fraction=0.046,
                label='Aleatoric uncertainty √mean(σ²) (steps)')
    ax.set_title('Aleatoric uncertainty vs 기하학적으로 계산한 차단 영역\n'
                 '(청록 점선 = 그 지점에서 목표로 가는 직선이 장애물에 '
                 '실제로 막히는 영역)', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    inside = aleatoric[blocked].mean()
    outside = aleatoric[~blocked].mean()
    print(f'차단 영역 내부 aleatoric 평균: {inside:.2f}')
    print(f'차단 영역 외부 aleatoric 평균: {outside:.2f}')
    print(f'비율(내부/외부): {inside / outside:.2f}x')
    print(f'-> {out}')
    return

  if args.mode == 'oblique':
    # 장애물을 직선상 중앙이 아니라 비스듬히 배치 -- '차단 쐐기'가 좌우
    # 대칭이 아니라 한쪽으로 기울어지게 만들어, aleatoric 핫존이 장애물
    # 주변 전체가 아니라 실제로 막힌 방향에만 붙는지 확인한다.
    goal = np.array([0.6, 0.6], dtype=np.float32)
    obst_c = np.array([0.25, -0.15], dtype=np.float32)
    obst_r = 0.18
    epistemic, aleatoric, xs = eval_scene(models, nets, normalize_obs,
                                          mu_var_fn, goal, obst_c, obst_r,
                                          0.006, gs)
    blocked = blocked_mask(xs, goal, obst_c, obst_r)
    ext = (-1, 1, -1, 1)

    # ---- 패널 1: 2D 지도 + 차단 경계 등고선 (verify와 동일 방식, 비스듬 배치)
    out2d = (args.out or f'results/obstacle_env/uncertainty_oblique_map{suffix}.png')
    fig, ax = plt.subplots(figsize=(7, 6.4))
    vmax_al = float(np.percentile(aleatoric, 99.5))
    im = ax.imshow(aleatoric, origin='lower', extent=ext, cmap='magma',
                   vmin=0, vmax=vmax_al)
    ax.contour(xs, xs, blocked.astype(float), levels=[0.5], colors='cyan',
              linewidths=2.2, linestyles='--')
    deco(ax, obst_c, obst_r, goal)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Aleatoric (steps)')
    ax.set_title('비스듬한 장애물 배치 — Aleatoric vs 기하학적 차단 영역',
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out2d), exist_ok=True)
    fig.savefig(out2d, dpi=130)
    plt.close(fig)

    # ---- 패널 2: 장애물 주위를 각도별로 도는 링 분석 (핵심 검증)
    # 표면에서 일정 거리 떨어진 원 위의 점들을 각도별로 훑어, 그 지점에서
    # 목표로 가는 직선이 막히는 각도 구간에서만 aleatoric이 높아지는지 본다.
    n_ang = 144
    ring_r = obst_r + 0.10
    angles = np.linspace(0, 360, n_ang, endpoint=False)
    ring_pts = np.stack([
        obst_c + ring_r * np.array([np.cos(np.radians(a)),
                                    np.sin(np.radians(a))], dtype=np.float32)
        for a in angles])
    ring_concats = []
    for p in ring_pts:
      d = goal - p
      nrm = np.linalg.norm(d)
      vel = (0.006 * d / nrm if nrm > 1e-6 else np.zeros(2)).astype(np.float32)
      obs = {'cur_pos': p.astype(np.float32), 'cur_vel': vel, 'goal_pos': goal,
             'obstacle_rel_pos': (obst_c - p).astype(np.float32),
             'obstacle_radius': np.array([obst_r], dtype=np.float32)}
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
      ring_concats.append(np.asarray(concat_obs(norm))[0])
    C = jnp.asarray(np.stack(ring_concats))
    ring_mus, ring_vars = [], []
    for p_ in models:
      mu, var = mu_var_fn(p_, C)
      ring_mus.append(np.asarray(mu)); ring_vars.append(np.asarray(var))
    ring_mus = np.stack(ring_mus); ring_vars = np.stack(ring_vars)
    ring_aleatoric = np.sqrt(ring_vars.mean(0))
    ring_blocked = np.array([is_blocked_point(p, goal, obst_c, obst_r)
                             for p in ring_pts])

    outring = 'results/obstacle_env/uncertainty_oblique_ring.png'
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(angles, ring_aleatoric, color='black', lw=1.3, zorder=3)
    ax.fill_between(angles, 0, ring_aleatoric.max() * 1.15,
                    where=ring_blocked, color='red', alpha=0.15,
                    label='이 각도는 목표로 가는 직선이 장애물에 막힘',
                    zorder=1, step='mid')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, ring_aleatoric.max() * 1.15)
    ax.set_xlabel('장애물 중심 기준 각도 (도)')
    ax.set_ylabel('Aleatoric uncertainty (steps)')
    ax.set_title(f'장애물 표면에서 {ring_r - obst_r:.2f} 떨어진 링을 따라 측정 '
                 f'— 막힌 각도(빨강)에서만 솟는가?', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(outring, dpi=130)
    plt.close(fig)

    blocked_mean = ring_aleatoric[ring_blocked].mean()
    clear_mean = ring_aleatoric[~ring_blocked].mean()
    print(f'\n[링 분석] 막힌 각도 aleatoric 평균: {blocked_mean:.2f} '
          f'({ring_blocked.sum()}/{n_ang} 각도)')
    print(f'[링 분석] 뚫린 각도 aleatoric 평균: {clear_mean:.2f} '
          f'({(~ring_blocked).sum()}/{n_ang} 각도)')
    print(f'비율(막힘/뚫림): {blocked_mean / clear_mean:.2f}x')
    print(f'-> {out2d}\n-> {outring}')
    return

  # ---- scenes: 여러 배치 비교
  scenes = [
      ('차단 (원본)', np.array([0.6, 0.6], np.float32),
       np.array([0.0, 0.0], np.float32), 0.2),
      ('비차단 (장애물을 구석으로)', np.array([0.6, 0.6], np.float32),
       np.array([-0.85, -0.85], np.float32), 0.2),
      ('장애물 없음 (반경≈0)', np.array([0.6, 0.6], np.float32),
       np.array([0.0, 0.0], np.float32), 0.01),
      ('차단 (반대편 골)', np.array([-0.6, -0.6], np.float32),
       np.array([0.0, 0.0], np.float32), 0.2),
      ('차단 (반경 큼)', np.array([0.6, 0.6], np.float32),
       np.array([0.0, 0.0], np.float32), 0.35),
      ('차단 (반경 작음)', np.array([0.6, 0.6], np.float32),
       np.array([0.0, 0.0], np.float32), 0.1),
  ]
  out = args.out or 'results/obstacle_env/uncertainty_scenes.png'
  ext = (-1, 1, -1, 1)
  results = []
  for name, goal, obst_c, obst_r in scenes:
    epistemic, aleatoric, xs = eval_scene(models, nets, normalize_obs,
                                          mu_var_fn, goal, obst_c, obst_r,
                                          0.006, gs)
    results.append((name, goal, obst_c, obst_r, aleatoric))
    print(f'{name}: aleatoric 평균={aleatoric.mean():.2f} '
          f'최댓값={aleatoric.max():.2f}')

  vmax_al = float(np.percentile(np.stack([r[4] for r in results]), 99.5))
  fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
  for ax, (name, goal, obst_c, obst_r, aleatoric) in zip(axes.flat, results):
    im = ax.imshow(aleatoric, origin='lower', extent=ext, cmap='magma',
                   vmin=0, vmax=vmax_al)
    deco(ax, obst_c, obst_r, goal)
    ax.set_title(name, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
  fig.suptitle('Aleatoric uncertainty — 장애물/목표 배치별 비교 '
               '(모든 패널 동일 색상 스케일)', fontsize=13)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out, dpi=130)
  plt.close(fig)
  print(f'-> {out}')


if __name__ == '__main__':
  main()
