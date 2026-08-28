"""Visualize the multimodal map: left/right demonstration routes and the
bimodal steps-to-go label histogram at the start region.

Run: python -m src.viz_multimodal   (writes results/e0_multimodal_map.png)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_env import (  # noqa: E402
    generate_multimodal_dataset, OBSTACLE_CENTER, OBSTACLE_RADIUS, GOAL, START)


def main(out='results/e0_multimodal_map.png', num_episodes=60):
  eps, _, sides = generate_multimodal_dataset(num_episodes=num_episodes, seed=0)
  lens = np.array([e.observation['cur_pos'].shape[0] for e in eps])
  left_lens, right_lens = lens[0::2], lens[1::2]

  fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

  # left panel: routes
  ax_map.add_patch(patches.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                                  facecolor='gray', edgecolor='black', alpha=0.5))
  for i, e in enumerate(eps):
    pos = e.observation['cur_pos']
    color = 'C0' if i % 2 == 0 else 'C3'
    ax_map.plot(pos[:, 0], pos[:, 1], color=color, alpha=0.3, linewidth=1)
  ax_map.scatter(*GOAL, marker='*', s=300, color='orange', zorder=5, label='goal')
  ax_map.scatter(*START, marker='o', s=100, color='green', zorder=5, label='start')
  ax_map.plot([], [], color='C0', label='left route')
  ax_map.plot([], [], color='C3', label='right route')
  ax_map.set_xlim(-1.3, 1.3); ax_map.set_ylim(-1, 1); ax_map.set_aspect('equal')
  ax_map.legend(loc='upper right'); ax_map.set_title('Multimodal map: left/right detours')

  # right panel: bimodal steps-to-go at start (episode length = STG from start)
  ax_hist.hist(left_lens, bins=15, alpha=0.6, color='C0', label=f'left (μ={left_lens.mean():.1f})')
  ax_hist.hist(right_lens, bins=15, alpha=0.6, color='C3', label=f'right (μ={right_lens.mean():.1f})')
  ax_hist.set_xlabel('steps-to-go from start (episode length)')
  ax_hist.set_ylabel('count')
  ax_hist.set_title(f'Bimodal STG labels (separation={right_lens.mean()-left_lens.mean():.1f})')
  ax_hist.legend()

  fig.tight_layout()
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.savefig(out, dpi=120); plt.close(fig)
  print(f'sides={sides}  left μ={left_lens.mean():.1f}±{left_lens.std():.1f}  '
        f'right μ={right_lens.mean():.1f}±{right_lens.std():.1f}  -> {out}')
  return out


if __name__ == '__main__':
  main()
