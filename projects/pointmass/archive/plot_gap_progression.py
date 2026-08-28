"""학습 정도별 STG gap 구조 비교 (덜 학습 -> 수렴).

gap의 크기가 아니라 **상태 종속성의 변화**를 본다:
  - 덜 학습된 정책: gap이 크지만 모든 상태에서 골고루(전반적 미숙)
  - 수렴한 정책    : gap이 작지만 특정 상태(목표 근처)에 몰림(국소 병목)
후자가 self-improvement에서 "어디를 고칠지" 짚어주는 신호다.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from analyze_stg_gap import load, ep_lengths, matched_gap

HUMAN = 'data/pusht_demos.pkl'
# (경로, 라벨, 성공률, 색)
RUNS = [
    ('data/pusht_dp10k_rollouts.pkl', '10k (30%)', 0.30, 'tab:red'),
    ('data/pusht_dp20k_rollouts.pkl', '20k (87%)', 0.87, 'tab:orange'),
    ('data/pusht_dp_rollouts.pkl', '수렴 (93%)', 0.93, 'tab:green'),
]
EDGES = [0, 10, 20, 40, 60, 90]


def main():
  h = load(HUMAN)
  hl = ep_lengths(h)
  runs = []
  for path, label, sr, color in RUNS:
    if not os.path.exists(path):
      print(f'skip (없음): {path}')
      continue
    p = load(path)
    g, h_med, q, dist = matched_gap(h, p, k=5, n_query=2000)
    runs.append(dict(label=label, sr=sr, color=color, p=p, g=g, h_med=h_med))

  fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=110)

  # (a) 길이 분포 생존곡선
  ax = axes[0]
  xs = np.sort(hl)
  ax.plot(xs, 1 - np.arange(len(xs)) / len(xs), lw=2.2, color='tab:blue',
          label='인간 데모')
  for r in runs:
    x = np.sort(ep_lengths(r['p']))
    ax.plot(x, 1 - np.arange(len(x)) / len(x), lw=1.8, color=r['color'],
            label=r['label'])
  ax.set_yscale('log'); ax.grid(alpha=.3)
  ax.set_xlabel('성공까지 스텝'); ax.set_ylabel('P(길이 > x)')
  ax.set_title('(a) 성공까지 스텝 분포')
  ax.legend(fontsize=8)

  # (b) gap 크기 요약 (중앙 + IQR)
  ax = axes[1]
  labels = [r['label'] for r in runs]
  med = [np.median(r['g']) for r in runs]
  lo = [np.percentile(r['g'], 25) for r in runs]
  hi = [np.percentile(r['g'], 75) for r in runs]
  cols = [r['color'] for r in runs]
  ax.bar(labels, med, color=cols, alpha=.85,
         yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
         capsize=5)
  for i, r in enumerate(runs):
    ax.text(i, med[i], f'  gap>0 {float((r["g"]>0).mean()):.0%}',
            ha='center', va='bottom', fontsize=8)
  ax.axhline(0, color='k', lw=1)
  ax.set_ylabel('gap 중앙 (막대: IQR)')
  ax.set_title('(b) gap 크기 — 학습될수록 줄어든다')

  # (c) gap의 상태 종속성 (정규화: 각 런의 평균 gap으로 나눠 '모양'만 비교)
  ax = axes[2]
  centers = [f'{lo_}~{hi_}' for lo_, hi_ in zip(EDGES[:-1], EDGES[1:])]
  for r in runs:
    ys = []
    for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
      m = (r['h_med'] >= lo_) & (r['h_med'] < hi_)
      ys.append(r['g'][m].mean() if m.sum() >= 20 else np.nan)
    ys = np.array(ys, float)
    ax.plot(centers, ys / np.nanmean(ys), 'o-', color=r['color'],
            label=r['label'], lw=1.8)
  ax.axhline(1.0, color='k', ls='--', lw=1, alpha=.6)
  ax.set_xlabel('인간 기준 남은 스텝(진행도)')
  ax.set_ylabel('구간 gap / 전체 평균 gap')
  ax.set_title('(c) gap의 상태 종속성 (모양만 비교)')
  ax.legend(fontsize=8); ax.grid(alpha=.3)

  fig.suptitle('학습 정도별 STG 격차: 크기는 줄지만 구조는 오히려 선명해진다',
               fontsize=12)
  fig.tight_layout(rect=[0, 0, 1, 0.93])
  out = 'results/pusht_compare/gap_progression.png'
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out); plt.close(fig)
  print('saved', out)

  # 표로도 출력
  print(f'\n{"정책":>12} {"성공률":>7} {"gap중앙":>8} {"gap>0":>7} '
        f'{"구간변동(최대/최소)":>18}')
  for r in runs:
    ys = []
    for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
      m = (r['h_med'] >= lo_) & (r['h_med'] < hi_)
      if m.sum() >= 20:
        ys.append(r['g'][m].mean())
    ratio = (max(ys) / min(ys)) if ys and min(ys) > 0 else float('nan')
    print(f'{r["label"]:>12} {r["sr"]:>7.0%} {np.median(r["g"]):>+8.1f} '
          f'{float((r["g"]>0).mean()):>7.0%} {ratio:>18.2f}')


if __name__ == '__main__':
  main()
