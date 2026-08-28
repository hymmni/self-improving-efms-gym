"""구맵(웨이포인트 환경)에서 구/신 예측기 분포를 겹쳐 mp4로 녹화.

watch_stdmap.py --probe both 와 동일한 화면(파랑=구맵 예측기, 빨강=신맵 예측기
교차 적용)을 디스플레이 없이(Agg) 렌더링해 저장한다. 일시정지/되감기 검토용.

  python record_stdmap.py                                   # learned 주행 10 eps
  python record_stdmap.py --controller straight --episodes 5
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import jax

from pointmass_core import Point2D, pd_controller
from src.probe_generic import GenericSTGProbe
from interactive_control import _write_video

STD_CKPT = 'checkpoints/std/predictor_f100.pkl'
OBS_CKPT = 'checkpoints/obstacle/predictor.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
      int(h), int(w), 3)


def far_corner(start, goal):
  corners = np.array([[0.9, 0.9], [0.9, -0.9], [-0.9, 0.9], [-0.9, -0.9]],
                     dtype=np.float32)
  d = np.minimum(np.linalg.norm(corners - start, axis=1),
                 np.linalg.norm(corners - goal, axis=1))
  return corners[int(np.argmax(d))]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--controller', choices=['straight', 'learned'],
                  default='learned')
  ap.add_argument('--episodes', type=int, default=10)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-steps', type=int, default=300)
  ap.add_argument('--fps', type=int, default=20)
  ap.add_argument('--dist-xmax', type=int, default=200)
  ap.add_argument('--out', default='results/videos/stdmap_both_predictors.mp4')
  args = ap.parse_args()

  drive_probe = GenericSTGProbe(STD_CKPT)
  probes = [('구맵 예측기', drive_probe, 'C0', False),
            ('신맵 예측기', GenericSTGProbe(OBS_CKPT), 'C3', True)]
  ctrl_label = ('직선 주행 (PD→골)' if args.controller == 'straight'
                else '학습된 BC 정책(구맵)')

  fig = plt.figure(figsize=(13, 7.2), dpi=100)
  gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.55)
  ax = fig.add_subplot(gs[:, 0])
  ax_dist = fig.add_subplot(gs[0, 1])
  ax_e = fig.add_subplot(gs[1, 1])
  ax_e2 = ax_e.twinx()    # 골까지 거리 축 (1회만 생성)
  ax_var = fig.add_subplot(gs[2, 1])

  frames = []
  key = jax.random.PRNGKey(42)
  n_success = 0
  corr_E = [[] for _ in probes]   # (E, dist) 쌍 축적 -> 최종 상관 출력
  corr_D = []

  for episode in range(args.episodes):
    seed = args.seed + episode
    np.random.seed(seed)
    env = Point2D()
    ts = env.reset()
    traj = [env._cur_pos.copy()]
    virt = (far_corner(env._cur_pos, env._goal_pos), 0.15)

    ax.clear()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.add_patch(patches.Circle(virt[0], virt[1], facecolor='none',
                                edgecolor='gray', ls=':', lw=1.5))
    ax.text(*virt[0], '가상', fontsize=7, ha='center', va='center',
            color='gray')
    ax.add_patch(patches.Circle(env._goal_pos, env._success_radius,
                                edgecolor='green', ls='--', fill=False))
    ax.scatter(*env._goal_pos, marker='*', s=180, color='orange', zorder=5)
    ax.scatter(*traj[0], marker='s', s=70, color='green', zorder=5)
    line, = ax.plot([], [], '-', color='blue', lw=1.2, alpha=0.8)
    dot, = ax.plot([], [], 'o', color='red', ms=9, zorder=6)

    ax_dist.clear()
    ax_dist.set_xlim(0, args.dist_xmax)
    ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')
    ax_dist.set_title('STG 분포 — 회색선=14 step 눈금', fontsize=10)
    for x in range(0, args.dist_xmax, 14):
      ax_dist.axvline(x, color='gray', lw=0.5, alpha=0.4)
    dist_lines = []
    for name, probe, color, _ in probes:
      dl, = ax_dist.plot([], [], color=color, lw=1.2, label=name)
      dist_lines.append(dl)
    ax_dist.legend(fontsize=8, loc='upper right')

    ax_e.clear(); ax_e2.clear(); ax_var.clear()
    ax_e.set_ylabel('E[STG]')
    ax_e2.set_ylabel('골까지 거리', color='green')
    ax_var.set_xlabel('step'); ax_var.set_ylabel('σ²')
    e_lines, v_lines = [], []
    for name, probe, color, _ in probes:
      el, = ax_e.plot([], [], color=color, lw=1.3, label=name)
      vl, = ax_var.plot([], [], color=color, lw=1.1, ls=':')
      e_lines.append(el); v_lines.append(vl)
    d_line, = ax_e2.plot([], [], color='green', lw=1.4, ls='--',
                         label='골까지 거리')
    ax_e.legend(fontsize=7, loc='upper right')
    ax_e.set_title('E[STG] vs 골까지 거리(초록 파선) — 동조 여부 관찰',
                   fontsize=9)
    hist = [([], []) for _ in probes]
    dists_ep = []

    step = 0
    while (not env.success()) and step < args.max_steps:
      obs = ts.observation
      recs = []
      for (name, probe, color, cross) in probes:
        pobs = dict(obs)
        if cross:
          c, r = virt
          pobs['obstacle_rel_pos'] = (c - obs['cur_pos']).astype(np.float32)
          pobs['obstacle_radius'] = np.array([r], dtype=np.float32)
        key, sub = jax.random.split(key)
        _, logits = probe._logits_and_act(pobs, sub)
        recs.append(probe._record(step, pobs, logits))

      if args.controller == 'straight':
        action = pd_controller(obs['cur_pos'], obs['cur_vel'],
                               obs['goal_pos'])
      else:
        key, sub = jax.random.split(key)
        act_norm, _ = drive_probe._logits_and_act(obs, sub)
        action = drive_probe.unnormalize_action(act_norm)[0]

      ts = env.step(np.asarray(action, dtype=np.float32))
      traj.append(env._cur_pos.copy())
      step += 1

      t = np.array(traj)
      line.set_data(t[:, 0], t[:, 1])
      dot.set_data([t[-1, 0]], [t[-1, 1]])
      ymax = 0.02
      dist_to_goal = float(np.linalg.norm(env._cur_pos - env._goal_pos))
      dists_ep.append(dist_to_goal)
      info_parts = [f'거리={dist_to_goal:.2f}']
      for i, ((name, probe, color, _), rec) in enumerate(zip(probes, recs)):
        dist_lines[i].set_data(probe.bin_vals, rec.probs)
        ymax = max(ymax, float(rec.probs.max()))
        hist[i][0].append(rec.expectation)
        hist[i][1].append(rec.variance)
        e_lines[i].set_data(range(len(hist[i][0])), hist[i][0])
        v_lines[i].set_data(range(len(hist[i][1])), hist[i][1])
        info_parts.append(f'{name}: E={rec.expectation:.1f} '
                          f'σ²={rec.variance:.0f}')
      ax_dist.set_ylim(0, ymax * 1.2)
      d_line.set_data(range(len(dists_ep)), dists_ep)
      for a in (ax_e, ax_e2, ax_var):
        a.relim(); a.autoscale_view()
      ax.set_title(f'[{ctrl_label}] ep{episode} (seed {seed})  step {step}  '
                   f'성공 {n_success}/{episode}\n'
                   + '   |   '.join(info_parts), fontsize=9)
      frames.append(fig_to_frame(fig))

    n_success += int(env.success())
    for i in range(len(probes)):
      corr_E[i] += hist[i][0]
    corr_D += dists_ep
    ax.set_title(f'[{ctrl_label}] ep{episode} 종료: '
                 f'{"성공" if env.success() else "실패"} ({step} steps)',
                 fontsize=11, fontweight='bold')
    frames += [fig_to_frame(fig)] * args.fps  # 에피소드 사이 1초 정지
    print(f'ep{episode} (seed {seed}): {"성공" if env.success() else "실패"} '
          f'{step} steps  (누적 프레임 {len(frames)})', flush=True)

  plt.close(fig)
  from scipy import stats as _st
  print('\n[E–거리 상관 (전체 스텝 풀링)]')
  for i, (name, *_rest) in enumerate(probes):
    r_s, _ = _st.spearmanr(corr_E[i], corr_D)
    r_p, _ = _st.pearsonr(corr_E[i], corr_D)
    print(f'  {name}: Spearman={r_s:.3f}  Pearson={r_p:.3f}')
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out}  ({len(frames)} frames, 약 {len(frames) / args.fps:.0f}초)')


if __name__ == '__main__':
  main()
