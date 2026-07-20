"""스킬 체이닝 1단계(σ² 기반 경계 탐지)를 mp4로 녹화 — 일시정지/되감기로 검토.

왼쪽: 궤적이 자라나며 스킬 구간마다 색이 바뀐다(경계 도달 시점에 정확히
전환). 오른쪽: σ² 실시간 추이 + 임계값(점선) + 검출된 경계(빨간 세로선,
도달하는 순간 나타남).

  python record_skill_chaining.py                    # 학습된 정책 8 에피소드
  python record_skill_chaining.py --episodes 5 --seed0 20
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.probe_generic import GenericSTGProbe
from src.skill_chaining_detect import detect_segments, refine_resolve_point
from src.skill_chaining_predict import rollout_full
from interactive_control import _write_video

CKPT = 'checkpoints/obstacle/predictor.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
      int(h), int(w), 3)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT)
  ap.add_argument('--episodes', type=int, default=8)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--factor', type=float, default=1.4)
  ap.add_argument('--hysteresis', type=float, default=0.85)
  ap.add_argument('--smooth', type=int, default=3)
  ap.add_argument('--min-gap', type=int, default=4)
  ap.add_argument('--fps', type=int, default=20)
  ap.add_argument('--out', default='results/videos/skill_chaining.mp4')
  args = ap.parse_args()

  probe = GenericSTGProbe(args.checkpoint)
  cmap = get_cmap('tab10')

  fig = plt.figure(figsize=(13, 5.6), dpi=100)
  gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
  ax = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[0, 1])

  frames = []
  n_success = 0
  for k in range(args.episodes):
    seed = args.seed0 + k
    pos, varis, obs_list, obst, goal, succ = rollout_full(probe, seed)
    boundaries, thresh, entries = detect_segments(
        varis, args.factor, args.smooth, args.hysteresis, args.min_gap)
    boundaries = [max(refine_resolve_point(varis, b), e + 1)
                 for b, e in zip(boundaries, entries)]
    n_success += int(succ)
    T = len(pos)

    ax.clear()
    ax.add_patch(patches.Circle(obst[0], obst[1], facecolor='gray',
                                edgecolor='black', alpha=0.4))
    ax.scatter(*goal, marker='*', s=180, color='orange', zorder=5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')

    ax2.clear()
    ax2.axhline(thresh, color='gray', ls=':', lw=1.2, label='임계값')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, max(varis) * 1.1)
    ax2.set_xlabel('step'); ax2.set_ylabel('σ²')
    ax2.legend(fontsize=8, loc='upper right')

    seg_id = 0
    b_ptr = 0
    seg_start = 0
    line_var, = ax2.plot([], [], color='C3', lw=1.3)
    dot, = ax.plot([], [], 'o', color='red', ms=8, zorder=6)

    for t in range(1, T + 1):
      # 이번 스텝에서 경계에 도달했으면 구간 전환
      if b_ptr < len(boundaries) and t - 1 == boundaries[b_ptr]:
        ax2.axvline(boundaries[b_ptr], color='red', ls='--', lw=1.2)
        seg_start = t - 1
        seg_id += 1
        b_ptr += 1

      seg_pos = pos[seg_start:t]
      ax.plot(seg_pos[:, 0], seg_pos[:, 1], '-', color=cmap(seg_id % 10),
              lw=1.6)
      dot.set_data([pos[t - 1, 0]], [pos[t - 1, 1]])
      line_var.set_data(range(t), varis[:t])

      ax.set_title(f'[학습된 BC 정책] seed {seed}  step {t}/{T}  '
                   f'스킬 구간 {seg_id + 1}  성공 {n_success}/{k + (1 if t==T else 0)}',
                   fontsize=10)
      ax2.set_title('σ² 실시간 추이 (빨강 점선=검출된 스킬 경계)', fontsize=10)
      frames.append(fig_to_frame(fig))

    end_title = (f'[학습된 BC 정책] seed {seed} 종료: '
                f'{"성공" if succ else "실패"} ({T-1} steps, '
                f'{seg_id + 1}개 스킬 구간 검출)')
    ax.set_title(end_title, fontsize=10, fontweight='bold')
    frames += [fig_to_frame(fig)] * args.fps
    print(f'ep{k} (seed {seed}): {"성공" if succ else "실패"} {T-1} steps, '
          f'{seg_id + 1}개 구간  (누적 프레임 {len(frames)})', flush=True)

  plt.close(fig)
  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out}  ({len(frames)} frames, 약 {len(frames)/args.fps:.0f}초)')


if __name__ == '__main__':
  main()
