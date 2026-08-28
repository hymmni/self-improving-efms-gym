"""부분관측 환경 녹화 — 숨은 장애물이 센싱되는 순간 STG 이봉이 무너지는 걸 본다.

왼쪽: 궤적. 장애물은 아직 못 봤으면 점선(숨은 진실), 센싱하면 실선으로 바뀐다.
파란 원은 에이전트의 센싱 반경. 가운데: STG 카테고리컬 분포 — 미관측 구간에선
두 봉우리(직진 ~14 vs 우회 ~48), 센싱 순간 한 봉우리로 붕괴. 오른쪽: σ² 추이와
센싱 시점(빨간 세로선).

  python record_partial_obs.py --episodes 6
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.obstacle_env import PartialObsObstacleAvoidPoint2D, demo_action_partial
from src.probe_generic import GenericSTGProbe
from interactive_control import _write_video

CKPT = 'checkpoints/obstacle_po_s0.25_b0.5/predictor.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
      int(h), int(w), 3)


def rollout(probe, seed, sensing, block_prob, max_steps=400):
  np.random.seed(seed)
  env = PartialObsObstacleAvoidPoint2D(sensing_radius=sensing,
                                       block_prob=block_prob)
  ts = env.reset()
  bins = probe.bin_vals
  positions = [env._cur_pos.copy()]
  probs, varis, mus, vis = [], [], [], []
  step = 0
  while (not env.success()) and step < max_steps:
    o = ts.observation
    norm = probe.normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], o))
    preds = probe.nets.network.apply(
        probe.params,
        jnp.concatenate([norm[f] for f in probe.obs_fields], axis=-1))
    p = np.asarray(jax.nn.softmax(
        preds.dist_to_succ_dist_params.logits, axis=-1))[0]
    mu = float((bins * p).sum())
    probs.append(p)
    mus.append(mu)
    varis.append(max(float((bins ** 2 * p).sum() - mu ** 2), 0.0))
    vis.append(float(o['obstacle_visible'][0]))
    ts = env.step(np.asarray(demo_action_partial(o), np.float32))
    positions.append(env._cur_pos.copy())
    step += 1
  return (np.array(positions), np.array(probs), np.array(varis),
          np.array(mus), np.array(vis), env._cur_obstacle,
          env._goal_pos.copy(), env.success(), env._is_blocking_episode)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT)
  ap.add_argument('--episodes', type=int, default=6)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--sensing-radius', type=float, default=0.25)
  ap.add_argument('--block-prob', type=float, default=0.5)
  ap.add_argument('--dist-xmax', type=int, default=90)
  ap.add_argument('--fps', type=int, default=12)
  ap.add_argument('--out', default='results/videos/partial_obs.mp4')
  args = ap.parse_args()

  probe = GenericSTGProbe(args.checkpoint)
  fig = plt.figure(figsize=(15.5, 5.4), dpi=100)
  gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05])
  ax, ax_d, ax_v = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[0, 2]))

  frames = []
  # 차단/비차단이 섞이도록 seed를 훑어 고른다
  picked, seed = [], args.seed0
  while len(picked) < args.episodes and seed < args.seed0 + 400:
    r = rollout(probe, seed, args.sensing_radius, args.block_prob)
    if r[7] and r[4][0] < 0.5:      # 성공 + 처음엔 안 보임
      picked.append((seed, r))
    seed += 1

  for k, (sd, r) in enumerate(picked):
    pos, probs, varis, mus, vis, obst, goal, succ, blk = r
    T = len(varis)
    sense = int(np.argmax(vis > 0.5)) if (vis > 0.5).any() else -1
    oc, orr = obst

    for t in range(1, T + 1):
      seen = vis[t - 1] > 0.5
      ax.clear(); ax_d.clear(); ax_v.clear()

      # ---- 궤적
      ax.add_patch(patches.Circle(
          oc, orr, facecolor=('gray' if seen else 'none'),
          edgecolor=('black' if seen else 'gray'),
          ls=('-' if seen else '--'), alpha=(0.45 if seen else 0.9), lw=1.6))
      ax.add_patch(patches.Circle(
          pos[t - 1], args.sensing_radius + orr, facecolor='none',
          edgecolor='tab:blue', ls=':', lw=1.2, alpha=0.7))
      ax.scatter(*goal, marker='*', s=190, color='orange', zorder=5)
      ax.plot(pos[:t, 0], pos[:t, 1], '-', color='steelblue', lw=1.7)
      ax.plot([pos[t - 1, 0]], [pos[t - 1, 1]], 'o', color='red', ms=8, zorder=6)
      ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
      ax.set_title(f'seed {sd} · step {t}/{T} · '
                   + ('장애물 관측됨' if seen else '장애물 미관측(점선=숨은 진실)')
                   + f'\n실제: {"경로 차단" if blk else "경로 뚫림"}',
                   fontsize=10,
                   color=('black' if seen else 'tab:red'))

      # ---- STG 분포
      p = probs[t - 1][:args.dist_xmax]
      ax_d.bar(probe.bin_vals[:args.dist_xmax], p, width=1.0,
               color=('tab:green' if seen else 'tab:red'))
      ax_d.set_xlim(0, args.dist_xmax)
      ax_d.set_ylim(0, max(float(probs[:, :args.dist_xmax].max()) * 1.1, 1e-6))
      ax_d.set_xlabel('steps-to-go'); ax_d.set_ylabel('확률')
      ax_d.set_title(f'STG 분포  E={mus[t-1]:.1f}  σ²={varis[t-1]:.0f}\n'
                     + ('한 봉우리(불확실성 해소)' if seen
                        else '두 봉우리(직진? 우회?)'), fontsize=10)

      # ---- σ² 추이
      ax_v.plot(range(t), varis[:t], color='C3', lw=1.6)
      if sense >= 0 and t > sense:
        ax_v.axvline(sense, color='red', ls='--', lw=1.3, label='센싱 순간')
        ax_v.legend(fontsize=8, loc='upper right')
      ax_v.set_xlim(0, T); ax_v.set_ylim(0, max(varis.max() * 1.1, 1e-6))
      ax_v.set_xlabel('step'); ax_v.set_ylabel('σ²')
      ax_v.set_title('σ² 추이 — 센싱 순간 급감', fontsize=10)

      fig.tight_layout()
      frames.append(fig_to_frame(fig))

    frames += [fig_to_frame(fig)] * args.fps   # 에피소드 끝 정지
    print(f'ep{k} (seed {sd}): {T} steps, 센싱 step={sense}, '
          f'{"차단" if blk else "뚫림"}  (누적 {len(frames)} 프레임)', flush=True)

  plt.close(fig)
  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out}  ({len(frames)} frames, 약 {len(frames)/args.fps:.0f}초)')


if __name__ == '__main__':
  main()
