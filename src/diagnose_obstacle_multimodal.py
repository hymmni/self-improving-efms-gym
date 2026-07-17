"""phase-3 핵심 검증: 웨이포인트 제거로 인위적 다봉 사다리(~14 step)가 사라졌나.

비교 대상 (둘 다 데모 데이터만으로 학습된 SFT 단계 예측기 — 동일 단계 비교):
  구맵: checkpoints/std/predictor_f100.pkl (웨이포인트 5개 데모, 140 bins)
  신맵: checkpoints/obstacle/predictor.pkl (장애물+노이즈 데모, 500 bins)
둘 다 bin 폭 = 1 step이라 봉우리 간격을 스텝 단위로 직접 비교 가능.

각 예측기를 자기 환경의 두 기질(데모 컨트롤러 / 학습 정책)에서 굴리며
diagnose_multimodal과 동일한 피크 검출로 다봉 비율/봉우리 간격을 측정한다.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.diagnose_multimodal import (demo_episode, learned_episode, dist_peaks,
                                     peak_spacings)
from src.obstacle_env import ObstacleAvoidPoint2D, demo_action_pf
from src.probe_generic import GenericSTGProbe
from stg_probe import STGProbe


def collect_old(episodes):
  """구맵(웨이포인트) 기질 2종 — diagnose_multimodal의 수집 로직 재사용."""
  probe = STGProbe('checkpoints/std/predictor_f100.pkl')
  all_records = []
  for seed in range(episodes):
    recs, _, _, _ = demo_episode(probe, seed)
    if len(recs) >= 15:
      all_records.append(recs)
  for seed in range(episodes):
    recs, _, _ = learned_episode(probe, seed)
    if len(recs) >= 15:
      all_records.append(recs)
  return all_records


def collect_new(episodes):
  """신맵(장애물+노이즈) 기질 2종."""
  probe = GenericSTGProbe('checkpoints/obstacle/predictor.pkl')
  all_records = []
  for seed in range(episodes):
    np.random.seed(seed)
    env = ObstacleAvoidPoint2D()
    recs = probe.rollout(env, policy='external',
                         action_source=lambda o: demo_action_pf(o), seed=seed)
    if len(recs) >= 15:
      all_records.append(recs)
  for seed in range(episodes):
    np.random.seed(seed)
    env = ObstacleAvoidPoint2D()
    recs = probe.rollout(env, policy='learned', seed=seed)
    if len(recs) >= 15:
      all_records.append(recs)
  return all_records, probe


def spacing_stats(all_records, rel_prominence=0.2):
  n_steps = n_mm = 0
  spacings = []
  for recs in all_records:
    for r in recs:
      pk = dist_peaks(r.probs, rel_prominence)
      n_steps += 1
      if len(pk) >= 2:
        n_mm += 1
        spacings += np.diff(pk).tolist()
  return dict(mm_fraction=n_mm / max(n_steps, 1),
              spacings=spacings, n_steps=n_steps)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=40)
  ap.add_argument('--out', default='results/obstacle_env')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  print('구맵(웨이포인트) 수집...', flush=True)
  old_records = collect_old(args.episodes)
  print('신맵(장애물+노이즈) 수집...', flush=True)
  new_records, new_probe = collect_new(args.episodes)

  summary = {}
  for prom in (0.1, 0.2, 0.3):
    so = spacing_stats(old_records, prom)
    sn = spacing_stats(new_records, prom)
    summary[str(prom)] = dict(
        old=dict(mm_fraction=so['mm_fraction'],
                 spacing_median=(float(np.median(so['spacings']))
                                 if so['spacings'] else None),
                 n_spacings=len(so['spacings'])),
        new=dict(mm_fraction=sn['mm_fraction'],
                 spacing_median=(float(np.median(sn['spacings']))
                                 if sn['spacings'] else None),
                 n_spacings=len(sn['spacings'])))
    print(f"prom={prom}: 다봉 비율 구맵 {so['mm_fraction']:.2f} -> "
          f"신맵 {sn['mm_fraction']:.2f} | 간격 중앙값 구맵 "
          f"{summary[str(prom)]['old']['spacing_median']} -> 신맵 "
          f"{summary[str(prom)]['new']['spacing_median']}")

  # ---- 그림: (a,b) 간격 히스토그램, (c) 신맵 분포 스냅샷 예시
  so = spacing_stats(old_records, 0.2)
  sn = spacing_stats(new_records, 0.2)
  fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

  axes[0].hist(so['spacings'], bins=np.arange(0, 80, 2), color='C0', alpha=0.8)
  axes[0].axvline(14, color='green', lw=2, label='웨이포인트 구간 14 step')
  axes[0].set_title(f"구맵(웨이포인트 데모)\n다봉 스텝 비율 {so['mm_fraction']:.0%}")
  axes[0].set_xlabel('인접 봉우리 간격 (step)'); axes[0].set_ylabel('빈도')
  axes[0].legend(fontsize=9)

  axes[1].hist(sn['spacings'], bins=np.arange(0, 80, 2), color='C3', alpha=0.8)
  axes[1].axvline(14, color='green', lw=2, ls='--', label='(구맵의 14 step 위치)')
  axes[1].set_title(f"신맵(장애물+노이즈 데모)\n다봉 스텝 비율 {sn['mm_fraction']:.0%}")
  axes[1].set_xlabel('인접 봉우리 간격 (step)')
  axes[1].legend(fontsize=9)

  # 신맵 예시 분포 3개 (에피소드 초/중/후반)
  recs = new_records[0]
  for frac, color in [(0.15, 'C0'), (0.5, 'C1'), (0.85, 'C3')]:
    r = recs[int(len(recs) * frac)]
    axes[2].plot(new_probe.bin_vals, r.probs, color=color, lw=1.4,
                 label=f'step {r.step_idx} (E={r.expectation:.0f})')
  axes[2].set_xlim(0, 300)
  axes[2].set_title('신맵 분포 스냅샷 (한 에피소드 초/중/후반)')
  axes[2].set_xlabel('steps-to-go'); axes[2].set_ylabel('prob')
  axes[2].legend(fontsize=9)

  fig.suptitle('인위적 다봉 사다리 검증 — 구맵 vs 신맵 (rel prominence 0.2, bin 폭 1 step 동일)',
               fontsize=13)
  fig.tight_layout()
  out_png = os.path.join(args.out, 'ladder_comparison.png')
  fig.savefig(out_png, dpi=130)
  plt.close(fig)

  with open(os.path.join(args.out, 'ladder_comparison.json'), 'w') as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
  print(f'-> {out_png}')


if __name__ == '__main__':
  main()
