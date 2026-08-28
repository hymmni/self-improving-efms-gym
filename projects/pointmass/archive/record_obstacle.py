"""장애물 회피 환경 에피소드를 mp4로 녹화 — 일시정지/되감기 하며 보기 위한 저장본.

watch_obstacle.py와 동일한 화면 구성(왼쪽 환경, 오른쪽 STG 분포 + E/σ² 추이)을
디스플레이 없이(Agg) 렌더링해 동영상으로 만든다.

  python record_obstacle.py                          # 학습된 정책 10 에피소드
  python record_obstacle.py --controller pf --episodes 5
  python record_obstacle.py --out results/videos/my.mp4 --fps 15
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

from src.obstacle_env import ObstacleAvoidPoint2D, demo_action, demo_action_pf
from interactive_control import _write_video

CKPT_DEFAULT = 'checkpoints/obstacle/predictor.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
      int(h), int(w), 3)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--controller', choices=['learned', 'pf', 'tangent'],
                  default='learned')
  ap.add_argument('--checkpoint', default=CKPT_DEFAULT)
  ap.add_argument('--episodes', type=int, default=10)
  ap.add_argument('--noise', type=float, default=1.5e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-steps', type=int, default=500)
  ap.add_argument('--fps', type=int, default=20)
  ap.add_argument('--dist-xmax', type=int, default=300)
  ap.add_argument('--out', default='results/videos/obstacle_episodes.mp4')
  args = ap.parse_args()

  probe = None
  if os.path.exists(args.checkpoint):
    from src.probe_generic import GenericSTGProbe
    probe = GenericSTGProbe(args.checkpoint)
  elif args.controller == 'learned':
    raise SystemExit(f'--controller learned 는 체크포인트 필요: {args.checkpoint}')

  labels = {'learned': '학습된 BC 정책',
            'pf': f'포텐셜필드+노이즈 std={args.noise:g}',
            'tangent': '접선점 조준'}
  label = labels[args.controller]

  fig = plt.figure(figsize=(13, 6.5), dpi=100)
  gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.35)
  ax = fig.add_subplot(gs[:, 0])
  ax_dist = fig.add_subplot(gs[0, 1])
  ax_ts = fig.add_subplot(gs[1, 1])
  ax_ts2 = ax_ts.twinx()  # 1회만 생성

  frames = []
  key = jax.random.PRNGKey(42)
  n_success = 0

  for episode in range(args.episodes):
    seed = args.seed + episode
    np.random.seed(seed)
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    traj = [env._cur_pos.copy()]
    center, radius = env._cur_obstacle

    ax.clear()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.add_patch(patches.Circle(center, radius, facecolor='gray',
                                edgecolor='black', alpha=0.5))
    ax.add_patch(patches.Circle(env._goal_pos, env._success_radius,
                                edgecolor='green', ls='--', fill=False))
    ax.scatter(*env._goal_pos, marker='*', s=180, color='orange', zorder=5)
    ax.scatter(*traj[0], marker='s', s=70, color='green', zorder=5)
    line, = ax.plot([], [], '-', color='blue', lw=1.2, alpha=0.8)
    dot, = ax.plot([], [], 'o', color='red', ms=9, zorder=6)

    if probe is not None:
      ax_dist.clear()
      ax_dist.set_xlim(0, args.dist_xmax)
      ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')
      ax_dist.set_title('STG 분포 (실시간)', fontsize=10)
      dist_line, = ax_dist.plot([], [], color='C0', lw=1.0)
      ax_ts.clear(); ax_ts2.clear()
      ax_ts.set_xlabel('step'); ax_ts.set_ylabel('E[STG]', color='C0')
      ax_ts2.set_ylabel('σ²', color='C3')
      e_line, = ax_ts.plot([], [], color='C0', lw=1.2)
      v_line, = ax_ts2.plot([], [], color='C3', lw=1.2, ls='--')
      exps, varis = [], []

    step = 0
    while (not env.success()) and step < args.max_steps:
      obs = ts.observation
      rec = None
      act_norm = None
      if probe is not None:
        key, sub = jax.random.split(key)
        act_norm, logits = probe._logits_and_act(obs, sub)
        rec = probe._record(step, obs, logits)

      if args.controller == 'learned':
        action = probe.unnormalize_action(act_norm)[0]
      elif args.controller == 'pf':
        action = demo_action_pf(obs, noise_std=args.noise)
      else:
        action = demo_action(obs)

      ts = env.step(np.asarray(action, dtype=np.float32))
      traj.append(env._cur_pos.copy())
      step += 1

      t = np.array(traj)
      line.set_data(t[:, 0], t[:, 1])
      dot.set_data([t[-1, 0]], [t[-1, 1]])
      info = (f'[{label}] ep{episode} (seed {seed})  step {step}  '
              f'속력 {np.linalg.norm(env._cur_vel):.4f}  '
              f'성공 {n_success}/{episode}')
      if rec is not None:
        info += f'\nE[STG]={rec.expectation:.1f}  σ²={rec.variance:.0f}'
        dist_line.set_data(probe.bin_vals, rec.probs)
        ax_dist.set_ylim(0, max(0.05, float(rec.probs.max()) * 1.2))
        exps.append(rec.expectation); varis.append(rec.variance)
        e_line.set_data(range(len(exps)), exps)
        v_line.set_data(range(len(varis)), varis)
        ax_ts.relim(); ax_ts.autoscale_view()
        ax_ts2.relim(); ax_ts2.autoscale_view()
      ax.set_title(info, fontsize=10)
      frames.append(fig_to_frame(fig))

    n_success += int(env.success())
    ax.set_title(f'[{label}] ep{episode} 종료: '
                 f'{"성공" if env.success() else "실패"} ({step} steps)',
                 fontsize=11, fontweight='bold')
    end_frame = fig_to_frame(fig)
    frames += [end_frame] * args.fps  # 에피소드 사이 1초 정지
    print(f'ep{episode} (seed {seed}): {"성공" if env.success() else "실패"} '
          f'{step} steps  (누적 프레임 {len(frames)})', flush=True)

  plt.close(fig)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out}  ({len(frames)} frames, {args.fps} fps, '
        f'약 {len(frames) / args.fps:.0f}초)')


if __name__ == '__main__':
  main()
