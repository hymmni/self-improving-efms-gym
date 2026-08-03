"""구맵(웨이포인트 환경) 관찰 뷰어 — 구/신 예측기의 STG 분포를 겹쳐서 실시간 비교.

  DISPLAY=:1 python watch_stdmap.py                          # 구맵 예측기만
  DISPLAY=:1 python watch_stdmap.py --probe both             # 구+신 겹쳐 보기
  DISPLAY=:1 python watch_stdmap.py --probe both --controller learned

- 조종(--controller): straight=PD가 골 직접 조준 / learned=구맵 BC 정책
- 분포(--probe): std=구맵 예측기 / obstacle=신맵 예측기(교차 적용) / both=겹침
  신맵 예측기는 장애물 관측 필드가 필수라, 경로에서 먼 구석에 '가상 장애물'
  (관측 전용, 물리 없음, 회색 점선)을 놓아 공급한다 — 학습 데이터에서
  "장애물을 지나쳐 멀리 있는" 상태는 흔하므로 가장 정직한 에뮬레이션.
- 분포 패널: 구맵=파랑, 신맵=빨강 원본 그대로(평활 없음). 회색 세로선=14 step
  눈금(웨이포인트 사다리 주기 참조).
"""

import argparse
import os
import sys

import numpy as np

STD_CKPT = 'checkpoints/std/predictor_f100.pkl'
OBS_CKPT = 'checkpoints/obstacle/predictor.pkl'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--controller', choices=['straight', 'learned'],
                  default='straight')
  ap.add_argument('--probe', choices=['std', 'obstacle', 'both'],
                  default='std')
  ap.add_argument('--fps', type=float, default=20)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--max-steps', type=int, default=300)
  ap.add_argument('--dist-xmax', type=int, default=200)
  args = ap.parse_args()

  if not os.environ.get('DISPLAY'):
    sys.exit('DISPLAY가 비어 있습니다. DISPLAY=:1 python watch_stdmap.py 로 실행하세요.')

  import matplotlib
  matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False

  import jax
  from pointmass_core import Point2D, pd_controller
  from src.probe_generic import GenericSTGProbe

  # ---- 프로브 구성: (라벨, 프로브, 색, 가상장애물 필요 여부)
  drive_probe = GenericSTGProbe(STD_CKPT)  # learned 조종용은 항상 구맵 정책
  probes = []
  if args.probe in ('std', 'both'):
    probes.append(('구맵 예측기', drive_probe, 'C0', False))
  if args.probe in ('obstacle', 'both'):
    probes.append(('신맵 예측기', GenericSTGProbe(OBS_CKPT), 'C3', True))
  need_virtual = any(cross for *_, cross in probes)
  print('분포 패널:', ', '.join(p[0] for p in probes))

  def far_corner(start, goal):
    corners = np.array([[0.9, 0.9], [0.9, -0.9], [-0.9, 0.9], [-0.9, -0.9]],
                       dtype=np.float32)
    d = np.minimum(np.linalg.norm(corners - start, axis=1),
                   np.linalg.norm(corners - goal, axis=1))
    return corners[int(np.argmax(d))]

  ctrl_label = ('직선 주행 (PD→골)' if args.controller == 'straight'
                else '학습된 BC 정책(구맵)')

  fig = plt.figure(figsize=(13, 6.5))
  gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.35)
  ax = fig.add_subplot(gs[:, 0])
  ax_dist = fig.add_subplot(gs[0, 1])
  ax_ts = fig.add_subplot(gs[1, 1])
  ax_ts2 = ax_ts.twinx()  # 1회만 생성 (누적 버그 방지)
  fig.canvas.manager.set_window_title('stdmap live')

  episode, seed, n_success = 0, args.seed, 0
  key = jax.random.PRNGKey(42)

  try:
    while plt.fignum_exists(fig.number):
      np.random.seed(seed)
      env = Point2D()
      ts = env.reset()
      traj = [env._cur_pos.copy()]
      virt = (far_corner(env._cur_pos, env._goal_pos), 0.15) \
          if need_virtual else None

      ax.clear()
      ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
      if virt is not None:
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
      ax_dist.set_title('STG 분포 (실시간) — 회색선=14 step 눈금', fontsize=10)
      for x in range(0, args.dist_xmax, 14):
        ax_dist.axvline(x, color='gray', lw=0.5, alpha=0.4)
      dist_lines = []
      for name, probe, color, _ in probes:
        dl, = ax_dist.plot([], [], color=color, lw=1.2, label=name)
        dist_lines.append(dl)
      ax_dist.legend(fontsize=8, loc='upper right')

      ax_ts.clear(); ax_ts2.clear()
      ax_ts.set_xlabel('step'); ax_ts.set_ylabel('E[STG] (실선)')
      ax_ts2.set_ylabel('σ² (점선)')
      e_lines, v_lines = [], []
      for name, probe, color, _ in probes:
        el, = ax_ts.plot([], [], color=color, lw=1.3, label=name)
        vl, = ax_ts2.plot([], [], color=color, lw=1.1, ls=':')
        e_lines.append(el); v_lines.append(vl)
      ax_ts.legend(fontsize=8, loc='upper right')
      hist = [([], []) for _ in probes]  # (exps, varis) per probe

      step = 0
      while (not env.success()) and step < args.max_steps \
            and plt.fignum_exists(fig.number):
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
        info_parts = []
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
        ax_ts.relim(); ax_ts.autoscale_view()
        ax_ts2.relim(); ax_ts2.autoscale_view()
        dist_to_goal = float(np.linalg.norm(env._cur_pos - env._goal_pos))
        ax.set_title(f'[{ctrl_label}] ep{episode} (seed {seed})  step {step}  '
                     f'성공 {n_success}/{episode}  골까지 {dist_to_goal:.2f}\n'
                     + '   |   '.join(info_parts), fontsize=9)
        plt.pause(max(1.0 / args.fps, 0.001))

      if not plt.fignum_exists(fig.number):
        break
      n_success += int(env.success())
      episode += 1
      seed += 1
      ax.set_title(f'[{ctrl_label}] ep{episode - 1} 종료: '
                   f'{"성공" if env.success() else "실패"} ({step} steps) — '
                   f'1.5초 후 다음 에피소드', fontsize=10)
      plt.pause(1.5)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
