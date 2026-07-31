"""웨이포인트 기반(phase 1) 데이터/정책 영상 녹화 — 정리 문서용.

driver='demo': pointmass_core.generate_dataset과 동일한 웨이포인트 순회
  컨트롤러(PD로 임의 웨이포인트 5개 경유 후 최종 목표)로 굴린다.
driver='learned': checkpoints/std/predictor_f100.pkl의 액션 헤드로 굴린다.

  python record_waypoint.py --driver demo --out results/videos/waypoint_demo.mp4
  python record_waypoint.py --driver learned --out results/videos/waypoint_policy.mp4
"""
import argparse, os
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pointmass_core import Point2D, pd_controller
from src.probe_generic import GenericSTGProbe
from interactive_control import _write_video

CKPT = 'checkpoints/std/predictor_f100.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)


def rollout(env, probe, seed, driver, n_waypoints=5, max_steps=300):
  np.random.seed(seed)
  ts = env.reset()
  obs = ts.observation
  succ = env.success()
  positions = [env._cur_pos.copy()]
  waypoints_hit = []
  wp_idx = 0
  cur_wp = env.sample_goal() if n_waypoints > 0 else obs['goal_pos']
  key = jax.random.PRNGKey(seed)
  step = 0
  while not succ and step < max_steps:
    wp_succ = env.success(waypoint=cur_wp)
    if wp_succ:
      waypoints_hit.append(env._cur_pos.copy())
      wp_idx = min(wp_idx + 1, n_waypoints)
      cur_wp = obs['goal_pos'] if wp_idx == n_waypoints else env.sample_goal()
    if driver == 'demo':
      act = pd_controller(obs['cur_pos'], obs['cur_vel'], cur_wp)
    else:
      key, sub = jax.random.split(key)
      act_norm, _ = probe._logits_and_act(obs, sub)
      act = probe.unnormalize_action(act_norm)[0]
    ts = env.step(np.asarray(act, dtype=np.float32))
    obs = ts.observation
    positions.append(env._cur_pos.copy())
    succ = env.success()
    step += 1
  return np.array(positions), np.array(waypoints_hit) if waypoints_hit else np.zeros((0, 2)), succ, obs['goal_pos'].copy()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--driver', choices=['demo', 'learned'], default='demo')
  ap.add_argument('--episodes', type=int, default=5)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--n-waypoints', type=int, default=5)
  ap.add_argument('--fps', type=int, default=15)
  ap.add_argument('--out', default='results/videos/waypoint_demo.mp4')
  args = ap.parse_args()

  probe = GenericSTGProbe(CKPT) if args.driver == 'learned' else None
  env = Point2D()
  label = '데모(웨이포인트 5개 PD 순회)' if args.driver == 'demo' else '학습된 정책(std 예측기 액션헤드)'

  fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
  frames = []
  n_succ = 0
  for k in range(args.episodes):
    seed = args.seed0 + k
    pos, wps, succ, goal = rollout(env, probe, seed, args.driver, args.n_waypoints)
    n_succ += int(succ)
    T = len(pos)
    for t in range(1, T + 1, 2):   # 2프레임씩 건너뛰어 파일 크기 절약
      ax.clear()
      ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
      ax.scatter(*goal, marker='*', s=180, color='orange', zorder=5, label='최종 목표')
      if len(wps):
        ax.scatter(wps[:, 0], wps[:, 1], marker='x', s=60, color='gray', zorder=4)
      ax.plot(pos[:t, 0], pos[:t, 1], '-', color='steelblue', lw=1.6)
      ax.plot([pos[t-1, 0]], [pos[t-1, 1]], 'o', color='red', ms=8, zorder=6)
      ax.set_title(f'[{label}] seed {seed}  step {t}/{T}  성공 {n_succ}/{k+(1 if t==T else 0)}', fontsize=10)
      fig.tight_layout()
      frames.append(fig_to_frame(fig))
    frames += [fig_to_frame(fig)] * args.fps
    print(f'ep{k}(seed{seed}): {"성공" if succ else "실패"} {T-1}steps 웨이포인트{len(wps)}개', flush=True)
  plt.close(fig)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out} ({len(frames)}프레임)')


if __name__ == '__main__':
  main()
