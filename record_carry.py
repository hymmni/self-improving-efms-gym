"""GraspCarry2D 롤아웃 영상 녹화.

같은 시드(= 같은 블록, 같은 은닉 물성)를 **느린 속도 vs 빠른 속도**로 나란히
굴려서 "속도가 곧 위험"과 "은닉 물성이 결과를 가른다"를 한 화면에 보인다.

  python record_carry.py --seeds 3 7 11 --out results/videos/grasp_carry.mp4
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.grasp_carry_env import GraspCarry2D, CarryConfig, ScriptedCarryPolicy


def block_verts(env):
  """블록의 월드 좌표 꼭짓점 (회전 포함)."""
  b = env.block
  out = []
  for v in env.block_shape.get_vertices():
    r = v.rotated(b.angle) + b.position
    out.append([r.x, r.y])
  return np.array(out)


def draw(ax, env, title, color, base_offset=None):
  cfg = env.cfg
  ax.clear()
  # 작업 영역만 잡는다 (월드 상단은 전부 빈 공간)
  ax.set_xlim(20, cfg.world - 20); ax.set_ylim(cfg.floor_y + 25, 250)
  ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
  ax.set_title(title, fontsize=9)

  # 바닥
  ax.plot([0, cfg.world], [cfg.floor_y, cfg.floor_y], color='0.35', lw=4)
  # 박스 (소스=회색, 타겟=초록)
  for cx, bw, c, lab in ((cfg.src_cx, cfg.src_box_w, '0.45', '소스(좁음)'),
                         (cfg.tgt_cx, cfg.tgt_box_w, 'green', '타겟(넓음)')):
    for sx in (-1, 1):
      x = cx + sx * bw / 2
      ax.plot([x, x], [cfg.floor_y, cfg.floor_y - cfg.box_rim_h], color=c, lw=4)
    ax.text(cx, cfg.floor_y + 20, lab, ha='center', fontsize=8, color=c)

  # 그리퍼 (스템 + 손바닥 + 손가락/패드) — 잡으면 손가락이 닫힌다
  ex, ey = env.ee.position.x, env.ee.position.y
  gcol = 'crimson' if env.gripping else '0.30'
  for poly in env.gripper_polys():
    ax.add_patch(Polygon(poly, closed=True, facecolor=gcol,
                         edgecolor='k', alpha=0.95, lw=0.8, zorder=2))
  # 블록 (그리퍼보다 위 레이어 — 패드는 물체 뒤로 가린다)
  ax.add_patch(Polygon(block_verts(env), closed=True, facecolor=color,
                       edgecolor='k', alpha=0.9, lw=1.2, zorder=3))
  # 접촉 구간(손가락과 블록 옆면이 겹친 길이) 표시
  c = env.contact_length()
  if c > 0:
    fy0 = max(ey, env.block.position.y - env.block_h / 2)
    xr = ex + env.cfg.gap_open / 2 + env.cfg.finger_t + 6
    ax.plot([xr, xr], [fy0, fy0 + c], '-', color='lime', lw=4, solid_capstyle='butt')
    ax.text(xr + 6, fy0 + c / 2, f'접촉 {c:.0f}px', fontsize=8, color='green',
            va='center')
  # 그립 중이면 미끄러짐(=경고 신호)을 표시.
  # 파지 순간의 오프셋을 뺀 **증가분**이 실제 미끄러진 양이다.
  if env.gripping:
    bx, by = env.block.position.x, env.block.position.y
    raw = float(np.hypot(bx - ex, by - ey))
    slip = raw - (base_offset if base_offset is not None else raw)
    ax.plot([ex, bx], [ey, by], '-', color='orange', lw=2.0)
    ax.text(35, 268, f'미끄러짐 {slip:+5.1f}px', fontsize=9,
            color=('red' if slip > 8 else 'darkorange'),
            fontweight=('bold' if slip > 8 else 'normal'))


def run_pair(seeds, speeds, cfg, fps, out):
  env_a, env_b = GraspCarry2D(cfg), GraspCarry2D(cfg)
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), dpi=110)
  frames = []
  for seed in seeds:
    pols = [ScriptedCarryPolicy(speed=float(s)) for s in speeds]
    envs = [env_a, env_b]
    for e, p in zip(envs, pols):
      e.reset(seed=seed); p.reset()
    done = [False, False]
    steps = [0, 0]
    base = [None, None]                   # 파지 순간의 EE-블록 오프셋
    for t in range(cfg.max_steps):
      for i, (e, p) in enumerate(zip(envs, pols)):
        if done[i]:
          continue
        _, _, term, trunc, _ = e.step(p(e))
        steps[i] += 1
        if term or trunc:
          done[i] = True
      for i, e in enumerate(envs):        # 파지 시작/해제에 맞춰 기준 오프셋 갱신
        if e.gripping and base[i] is None:
          base[i] = float((e.ee.position - e.block.position).length)
        elif not e.gripping:
          base[i] = None
      for i, (e, sp) in enumerate(zip(envs, speeds)):
        state = '성공' if e.success() else ('물림' if e.is_held() else ('닫힘' if e.gripping else '열림'))
        ac = (f'  접촉={e.contact_length():.0f}px' if e.is_held() else '')
        draw(axes[i], e,
             f'{"느리게" if i == 0 else "빠르게"} (speed={sp})   '
             f'step {steps[i]}   낙하 {e.n_drops}회   [{state}]{ac}',
             color=('tab:blue' if i == 0 else 'tab:red'),
             base_offset=base[i])
      fig.suptitle(
          f'seed {seed}  |  [은닉] mu={env_a.mu:.2f}  fill={env_a.fill:.2f}  '
          f'질량={env_a.mass:.1f}   |   [관측 가능] 블록 '
          f'{env_a.block_w:.0f}x{env_a.block_h:.0f}',
          fontsize=10)
      fig.tight_layout(rect=[0, 0, 1, 0.94])
      c = FigureCanvasAgg(fig); c.draw()
      w, h = fig.get_size_inches() * fig.get_dpi()
      frames.append(np.frombuffer(c.buffer_rgba(), np.uint8)
                    .reshape(int(h), int(w), 4)[..., :3].copy())
      if all(done):
        break
    frames += [frames[-1]] * fps          # 에피소드 끝에서 잠시 정지
    print(f'seed {seed}: mu={env_a.mu:.2f} fill={env_a.fill:.2f}  '
          + '  '.join(f'speed{s}->{st}스텝/{e.n_drops}낙하/'
                      f'{"성공" if e.success() else "실패"}'
                      for s, st, e in zip(speeds, steps, envs)), flush=True)
  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'saved {out} ({len(frames)} frames)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 11])
  ap.add_argument('--speeds', type=float, nargs=2, default=[28.0, 58.0])
  ap.add_argument('--fps', type=int, default=12)
  ap.add_argument('--out', default='results/videos/grasp_carry.mp4')
  args = ap.parse_args()
  cfg = CarryConfig(max_steps=120)
  run_pair(args.seeds, args.speeds, cfg, args.fps, args.out)


if __name__ == '__main__':
  main()
