"""방향 전환(웨이포인트 스위치) 순간 분산이 오르는가 내리는가 — 무엇이 부호를 가르나.

예측기는 (cur_pos, cur_vel, goal=최종목표)만 관측하고 웨이포인트는 못 본다. 따라서
스위치는 컨트롤러의 추종 목표를 바꿔 '움직임'을 바꾸고, 그 움직임 변화가 분포를
바꾼다(간접 경로). 각 스위치에서 Δσ²의 부호를 아래 후보들로 회귀/분리해 원인 규명:

  remaining_wp   : 남은 웨이포인트 수 (다봉=남은구간수 모호성 가설의 직접 예측변수)
  d_final        : 최종 목표까지 거리
  progress       : 에피소드 진행 비율 s/len
  d_new_wp       : 새로 뽑힌 웨이포인트까지 거리 (그쪽으로 얼마나 멀리 재유도되나)
  turn_angle     : 스위치 전후 속도 방향 전환각
  speed_after    : 스위치 직후 속력

Δσ² = σ²(s+K) − σ²(s)  (K step 뒤, 재유도된 움직임의 효과를 포착; 양수=분산 증가)
"""

import argparse
import json
import os

import numpy as np
import jax
from scipy import stats

from pointmass_core import Point2D, pd_controller
from stg_probe import STGProbe

WINDOW = 4  # 스위치 효과를 읽는 look-ahead (leg~14보다 짧게)


def demo_episode_switches(probe, seed, num_waypoints=5, max_steps=400):
  """generate_dataset 로직 재현 + 각 스위치에서 Δσ²와 후보 예측변수 기록."""
  np.random.seed(seed)
  env = Point2D()
  ts = env.reset()
  cur_obs = ts.observation
  key = jax.random.PRNGKey(seed)

  waypoint_idx = 0
  cur_waypoint = env.sample_goal()
  recs, vels, positions, wp_targets = [], [], [], []
  switch_meta = []  # (step, remaining_wp, d_new_wp)
  step = 0
  while not env.success() and step < max_steps:
    if env.success(waypoint=cur_waypoint):
      waypoint_idx = min(waypoint_idx + 1, num_waypoints)
      cur_waypoint = (cur_obs['goal_pos'] if waypoint_idx == num_waypoints
                      else env.sample_goal())
      remaining = num_waypoints - waypoint_idx
      d_new = float(np.linalg.norm(cur_obs['cur_pos'] - cur_waypoint))
      switch_meta.append((step, remaining, d_new))
    key, sub = jax.random.split(key)
    _, logits = probe._logits_and_act(cur_obs, sub)
    recs.append(probe._record(step, cur_obs, logits))
    vels.append(cur_obs['cur_vel'].copy())
    positions.append(cur_obs['cur_pos'].copy())
    wp_targets.append(cur_waypoint.copy())
    act = pd_controller(cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
    ts = env.step(act)
    cur_obs = ts.observation
    step += 1

  n = len(recs)
  if n < WINDOW + 2:
    return []
  var = np.array([r.variance for r in recs])
  goal = recs[-1].obs['goal_pos']
  events = []
  for (s, remaining, d_new) in switch_meta:
    if s == 0 or s + WINDOW >= n:
      continue
    # 전후 속도 방향 전환각 (스위치 전 s-1 -> 스위치 후 s+1)
    v0, v1 = vels[max(0, s - 1)], vels[min(n - 1, s + 1)]
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    turn = float(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1 + 1e-9), -1, 1))) \
        if n0 > 1e-6 and n1 > 1e-6 else 0.0
    events.append(dict(
        dvar=float(var[s + WINDOW] - var[s]),
        remaining_wp=int(remaining),
        d_final=float(np.linalg.norm(positions[s] - goal)),
        progress=float(s / n),
        d_new_wp=float(d_new),
        turn_angle=float(np.degrees(turn)),
        speed_after=float(np.linalg.norm(vels[min(n - 1, s + 1)])),
    ))
  return events


def ols(events, cols):
  y = np.array([e['dvar'] for e in events])
  X = np.column_stack([np.ones(len(events))] +
                       [[e[c] for e in events] for c in cols])
  coef, *_ = np.linalg.lstsq(X, y, rcond=None)
  resid = y - X @ coef
  nobs, k = X.shape
  s2 = (resid @ resid) / (nobs - k)
  se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
  tv = coef / se
  pv = 2 * (1 - stats.t.cdf(np.abs(tv), nobs - k))
  r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
  return ['intercept'] + cols, coef, pv, r2


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--episodes', type=int, default=120)
  ap.add_argument('--out', default='results/diag_multimodal/switch_sign.json')
  args = ap.parse_args()

  probe = STGProbe(args.checkpoint)
  events = []
  for seed in range(args.episodes):
    events += demo_episode_switches(probe, seed)
  print(f'스위치 이벤트 {len(events)}개')

  dvar = np.array([e['dvar'] for e in events])
  up = dvar > 0
  print(f'  분산 증가 {up.sum()} / 감소 {(~up).sum()} '
        f'(증가 비율 {up.mean():.2f})')

  cols = ['remaining_wp', 'd_final', 'progress', 'd_new_wp', 'turn_angle',
          'speed_after']
  print('\n[증가군 vs 감소군 평균]')
  summary = {'n': len(events), 'frac_up': float(up.mean()), 'groups': {}}
  for c in cols:
    v = np.array([e[c] for e in events])
    mu_up, mu_dn = v[up].mean(), v[~up].mean()
    _, p = stats.mannwhitneyu(v[up], v[~up])
    print(f'  {c:12s}  증가={mu_up:8.3f}  감소={mu_dn:8.3f}  MWU p={p:.4f}')
    summary['groups'][c] = dict(up=float(mu_up), down=float(mu_dn), p=float(p))

  names, coef, pv, r2 = ols(events, cols)
  print(f'\n[다중회귀 Δσ² ~ {"+".join(cols)}]  R²={r2:.3f}')
  summary['ols'] = {'r2': float(r2), 'terms': {}}
  for nm, c, p in zip(names, coef, pv):
    print(f'  {nm:12s}  coef={c:9.3f}  p={p:.4f}')
    summary['ols']['terms'][nm] = dict(coef=float(c), p=float(p))

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)

  # --- 시각화: 무엇이 부호를 가르나 (지배 인자 speed_after, 보조 remaining_wp)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False

  def binned(xkey, edges, xlabel, ax):
    x = np.array([e[xkey] for e in events])
    y = dvar
    centers, means, fracup = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
      m = (x >= lo) & (x < hi)
      if m.sum() >= 8:
        centers.append(0.5 * (lo + hi))
        means.append(y[m].mean())
        fracup.append((y[m] > 0).mean())
    ax.axhline(0, color='gray', lw=0.8)
    ax.bar(centers, means, width=(edges[1] - edges[0]) * 0.8,
           color=['#d1495b' if v > 0 else '#3a7ca5' for v in means])
    ax.set_xlabel(xlabel); ax.set_ylabel('평균 Δσ² (양수=분산 증가)')
    ax2 = ax.twinx()
    ax2.plot(centers, fracup, 'o--', color='black', lw=1.2, ms=5)
    ax2.axhline(0.5, color='black', lw=0.6, ls=':')
    ax2.set_ylabel('분산 증가 비율', color='black'); ax2.set_ylim(0, 1)

  fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  sp = np.array([e['speed_after'] for e in events])
  binned('speed_after', np.quantile(sp, np.linspace(0, 1, 8)),
         '전환 직후 속력 (느림 → 빠름)', axes[0])
  axes[0].set_title('전환 직후 속력 vs 분산 변화 (지배 인자)')
  binned('remaining_wp', np.arange(-0.5, 5.5, 1),
         '남은 웨이포인트 수', axes[1])
  axes[1].set_title('남은 웨이포인트 수 vs 분산 변화')
  fig.suptitle('방향 전환 순간 분산이 오르내리는 이유 (막대=평균Δσ², 점=증가비율)',
               fontsize=13)
  fig.tight_layout()
  fig.savefig(os.path.join(os.path.dirname(args.out), 'switch_sign.png'), dpi=130)
  plt.close(fig)


if __name__ == '__main__':
  main()
