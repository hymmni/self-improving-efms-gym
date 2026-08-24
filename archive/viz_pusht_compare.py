"""인간 데모 vs DP 롤아웃 시각 비교 (STG 소스 구분 선검증용).

저장된 상태(agent_pos + T블록 8키포인트)로 두 데이터셋을 동일 방식으로 렌더한다.
  - 몽타주 PNG: 에피소드별 에이전트 전체 궤적 + 최종 블록 포즈 + 목표 T
  - 나란히 MP4 : 인간 1 에피소드 vs DP 1 에피소드 동시 재생

lerobot 환경에서 실행(imageio 필요). 데이터 pkl은 순수 numpy라 환경 무관.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

# 목표 T 키포인트 — 커버리지가 쓰는 get_goal_pose_body(goal_pose=[256,256,pi/4])
# 경로로 계산(=실제 목표 존과 일치). _set_state+space.step 방식은 물리 스텝이
# 블록을 움직여 틀린 위치가 나왔음.
GOAL_KP = np.array([[213.6, 213.6], [298.4, 298.4], [277.2, 319.6], [192.4, 234.8],
                    [224.2, 266.6], [245.4, 287.8], [181.8, 351.5], [160.5, 330.2]])
# 키포인트 -> 두 사각형(상단 가로바 [0,1,2,3], 세로바 [4,5,6,7])
TEE_POLYS = ([0, 1, 2, 3], [4, 5, 6, 7])
SIZE = 512


def draw_tee(ax, kp8, facecolor, alpha, edgecolor, zorder=2):
  for idx in TEE_POLYS:
    ax.add_patch(Polygon(kp8[idx], closed=True, facecolor=facecolor,
                         edgecolor=edgecolor, alpha=alpha, lw=1.5, zorder=zorder))


def episodes(data):
  eid = data['episode_id']
  ap = data['observation']['agent_pos']
  es = data['observation']['env_state']
  cov = data['coverage']
  out = []
  for e in np.unique(eid):
    m = eid == e
    out.append(dict(agent=ap[m], kp=es[m].reshape(m.sum(), 8, 2), cov=cov[m]))
  return out


def setup_ax(ax, title):
  ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)   # 이미지 좌표(위가 y=0)
  ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
  ax.set_title(title, fontsize=10)
  draw_tee(ax, GOAL_KP, 'green', 0.18, 'green', zorder=1)   # 목표(반투명)


def montage(human, dp, out_path, n=4):
  fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6), dpi=110)
  for row, (data, label, col) in enumerate(
      [(human, '인간 데모', 'tab:blue'), (dp, 'DP 롤아웃', 'tab:red')]):
    for j in range(n):
      ax = axes[row, j]
      ep = data[j]
      setup_ax(ax, f'{label} #{j}  ({len(ep["agent"])}스텝)')
      a = ep['agent']
      ax.plot(a[:, 0], a[:, 1], '-', color=col, lw=1.0, alpha=0.6, zorder=3)
      ax.scatter(a[0, 0], a[0, 1], s=30, c='k', marker='o', zorder=4)   # 시작
      draw_tee(ax, ep['kp'][-1], col, 0.35, col, zorder=2)              # 최종 블록
  fig.suptitle('에이전트 궤적(선) + 최종 블록 포즈(색) + 목표 T(초록)', fontsize=12)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  fig.savefig(out_path); plt.close(fig)
  print('saved', out_path)


def _frame(fig):
  c = FigureCanvasAgg(fig); c.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(c.buffer_rgba(), np.uint8).reshape(int(h), int(w), 4)[..., :3]


def side_by_side_video(human_ep, dp_ep, out_path, fps=12, stride=2):
  T = max(len(human_ep['agent']), len(dp_ep['agent']))
  fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=110)
  frames = []
  for t in range(0, T, stride):
    for ax, ep, label, col in [(axes[0], human_ep, '인간 데모', 'tab:blue'),
                               (axes[1], dp_ep, 'DP 롤아웃', 'tab:red')]:
      ax.clear()
      ti = min(t, len(ep['agent']) - 1)
      setup_ax(ax, f'{label}  step {ti}/{len(ep["agent"])-1}  cov {ep["cov"][ti]:.2f}')
      a = ep['agent'][:ti + 1]
      ax.plot(a[:, 0], a[:, 1], '-', color=col, lw=1.2, alpha=0.7, zorder=3)
      draw_tee(ax, ep['kp'][ti], col, 0.4, col, zorder=2)
      ax.scatter(a[-1, 0], a[-1, 1], s=45, c=col, edgecolors='k', zorder=5)
    fig.tight_layout()
    frames.append(_frame(fig))
  for ep_last in range(fps):     # 마지막 프레임 정지
    frames.append(frames[-1])
  plt.close(fig)
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  imageio.mimsave(out_path, frames, fps=fps)
  print('saved', out_path, f'({len(frames)} frames)')


def length_hist(human, dp, out_path):
  hl = np.array([len(e['agent']) for e in human])
  dl = np.array([len(e['agent']) for e in dp])
  fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
  bins = np.linspace(0, max(hl.max(), dl.max()) + 10, 40)
  ax.hist(hl, bins=bins, alpha=0.55, color='tab:blue',
          label=f'인간 데모 (중앙 {np.median(hl):.0f})')
  ax.hist(dl, bins=bins, alpha=0.55, color='tab:red',
          label=f'DP 롤아웃 (중앙 {np.median(dl):.0f})')
  ax.axvline(np.median(hl), color='tab:blue', ls='--', lw=1)
  ax.axvline(np.median(dl), color='tab:red', ls='--', lw=1)
  ax.set_xlabel('성공까지 스텝 수 (에피소드 길이 = STG[0])')
  ax.set_ylabel('에피소드 수')
  ax.set_title('성공까지 걸린 스텝 분포: 인간 vs DP 정책')
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_path); plt.close(fig)
  print('saved', out_path, f'| human med {np.median(hl):.0f} dp med {np.median(dl):.0f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--human', default='data/pusht_demos.pkl')
  ap.add_argument('--dp', default='data/pusht_dp_rollouts.pkl')
  ap.add_argument('--outdir', default='results/pusht_compare')
  ap.add_argument('--human-ep', type=int, default=0)
  ap.add_argument('--dp-ep', type=int, default=0)
  args = ap.parse_args()

  human = episodes(pickle.load(open(args.human, 'rb')))
  dp = episodes(pickle.load(open(args.dp, 'rb')))
  print(f'human {len(human)}ep, dp {len(dp)}ep')

  montage(human, dp, os.path.join(args.outdir, 'montage.png'), n=4)
  length_hist(human, dp, os.path.join(args.outdir, 'length_hist.png'))
  side_by_side_video(human[args.human_ep], dp[args.dp_ep],
                     os.path.join(args.outdir, 'human_vs_dp.mp4'))


if __name__ == '__main__':
  main()
