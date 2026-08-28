"""장애물 회피 환경 실시간 뷰어 — 에이전트 움직임 + STG 분포 라이브 관찰.

사용법 (X11 디스플레이가 있는 터미널에서; 없으면 DISPLAY=:1 를 앞에 붙임):
  python watch_obstacle.py --controller learned   # 학습된 BC 정책 (기본)
  python watch_obstacle.py --controller pf        # 데모 컨트롤러(PF+노이즈)
  python watch_obstacle.py --controller tangent   # 접선점 조준 비교
  python watch_obstacle.py --fps 40 --seed 5 --noise 2e-4

체크포인트(checkpoints/obstacle/predictor.pkl)가 있으면 어느 컨트롤러든
오른쪽에 steps-to-go 분포(위)와 E/σ² 추이(아래)가 실시간으로 함께 표시된다.
에피소드가 끝나면 자동으로 다음 에피소드 시작. 창을 닫으면 종료.
"""

import argparse
import os
import sys

import numpy as np

CKPT_DEFAULT = 'checkpoints/obstacle/predictor.pkl'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--controller', choices=['learned', 'pf', 'tangent'],
                  default='learned',
                  help='learned=학습된 BC 정책, pf=포텐셜필드+노이즈(데모), '
                       'tangent=접선점 조준')
  ap.add_argument('--checkpoint', default=CKPT_DEFAULT)
  ap.add_argument('--noise', type=float, default=1.5e-4,
                  help='pf 컨트롤러의 노이즈 std (기본 1.5e-4 = 채택값)')
  ap.add_argument('--fps', type=float, default=20)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-steps', type=int, default=500)
  ap.add_argument('--dist-xmax', type=int, default=300,
                  help='분포 패널 x축 상한 (bin 500 중 앞부분만 표시)')
  args = ap.parse_args()

  if not os.environ.get('DISPLAY'):
    sys.exit('DISPLAY가 비어 있습니다. 원격 데스크톱/서버 모니터에서는 '
             'DISPLAY=:1 python watch_obstacle.py 로 실행하세요.')

  import matplotlib
  matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False

  import jax
  from src.obstacle_env import (ObstacleAvoidPoint2D, demo_action,
                                demo_action_pf)

  # ---- 예측기 (있으면 분포 패널 + learned 컨트롤러 지원)
  probe = None
  if os.path.exists(args.checkpoint):
    from src.probe_generic import GenericSTGProbe
    probe = GenericSTGProbe(args.checkpoint)
    print(f'예측기 로드: {args.checkpoint} (meta: {probe.meta})')
  elif args.controller == 'learned':
    sys.exit(f'--controller learned 는 체크포인트가 필요합니다: {args.checkpoint}')

  labels = {'learned': '학습된 BC 정책',
            'pf': f'포텐셜필드+노이즈 std={args.noise:g}',
            'tangent': '접선점 조준'}
  label = labels[args.controller]

  # ---- 레이아웃: 왼쪽 환경(정사각), 오른쪽 위 분포 / 아래 E·σ² 추이
  if probe is not None:
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, 1])
  else:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax_dist = ax_ts = None
  fig.canvas.manager.set_window_title('obstacle-avoid live')
  # twinx 축은 반드시 한 번만 생성 — 에피소드마다 만들면 ax.clear()가 지우지
  # 못해 오른쪽 축/이전 에피소드 선이 겹겹이 누적된다
  ax_ts2 = ax_ts.twinx() if probe is not None else None

  episode, seed, n_success = 0, args.seed, 0
  key = jax.random.PRNGKey(42)

  try:
    while plt.fignum_exists(fig.number):
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

      dist_line = None
      exps, varis = [], []
      if probe is not None:
        ax_dist.clear()
        ax_dist.set_xlim(0, args.dist_xmax); ax_dist.set_ylim(0, 0.12)
        ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')
        ax_dist.set_title('STG 분포 (실시간)', fontsize=10)
        dist_line, = ax_dist.plot([], [], color='C0', lw=1.0)
        ax_ts.clear(); ax_ts2.clear()
        ax_ts.set_xlabel('step'); ax_ts.set_ylabel('E[STG]', color='C0')
        ax_ts2.set_ylabel('σ²', color='C3')
        e_line, = ax_ts.plot([], [], color='C0', lw=1.2)
        v_line, = ax_ts2.plot([], [], color='C3', lw=1.2, ls='--')

      step = 0
      while (not env.success()) and step < args.max_steps \
            and plt.fignum_exists(fig.number):
        obs = ts.observation
        rec = None
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
        speed = float(np.linalg.norm(env._cur_vel))
        info = (f'[{label}] ep{episode} (seed {seed})  step {step}  '
                f'속력 {speed:.4f}  성공 {n_success}/{episode}')
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
        plt.pause(max(1.0 / args.fps, 0.001))

      if not plt.fignum_exists(fig.number):
        break
      n_success += int(env.success())
      episode += 1
      seed += 1
      ax.set_title(f'[{label}] ep{episode - 1} 종료: '
                   f'{"성공" if env.success() else "실패"} ({step} steps) — '
                   f'1.5초 후 다음 에피소드', fontsize=10)
      plt.pause(1.5)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
