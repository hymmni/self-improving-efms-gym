"""장애물 회피 환경(phase 3)에서 baseline vs ours(분산 보상) REINFORCE 비교.

phase 2 src/run_e2.py과 같은 구조(다중 시드, 수렴 곡선)를 이 환경에
적용한다. 이 환경 고유의 차이: BC 정책 성공률이 이미 0.96~1.0로 천장이라
성공률보다 **에피소드 길이(효율)**가 개선을 볼 더 정보량 있는 지표 —
데모가 노이즈로 의도적으로 비효율적이므로 self-improvement가 실제로 할
일이 있는지(길이 단축)를 이걸로 본다.

eps는 안전값(기본 1.0)을 쓴다 — eps=1e-6은 이 잘 보정된 예측기에서 보상
폭주 버그가 있다고 확인됐음(2026-07-13, 그때는 수정을 보류했었음).

Run: python -m src.run_e2_obstacle --seeds 0,1,2,3,4 --iters 30
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.reinforce_obstacle import train_reinforce, CKPT_DEFAULT


def run_condition(ckpt, reward_cfg, seeds, iters, env_steps, lr, max_steps):
  sr_curves, len_curves = [], []
  for s in seeds:
    c = train_reinforce(ckpt, reward_cfg, num_iters=iters,
                        env_steps_per_iter=env_steps, seed=s, lr=lr,
                        max_steps=max_steps)
    sr_curves.append(c['success_rate'])
    len_curves.append(c['ep_len'])
    print(f'    seed {s}: SR {c["success_rate"][0]:.2f}->{c["success_rate"][-1]:.2f}  '
          f'len {c["ep_len"][0]:.0f}->{c["ep_len"][-1]:.0f}')
  return np.array(sr_curves), np.array(len_curves)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT_DEFAULT)
  ap.add_argument('--seeds', default='0,1,2,3,4')
  ap.add_argument('--iters', type=int, default=30)
  ap.add_argument('--env_steps', type=int, default=2048)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--max_steps', type=int, default=300)
  ap.add_argument('--beta', type=float, default=0.5, help='beta for the ours arm')
  ap.add_argument('--eps', type=float, default=1.0,
                  help='분산항 분모 eps (안전값; 1e-6은 폭주 버그 있음)')
  ap.add_argument('--out', default='results/e2_obstacle')
  args = ap.parse_args()

  seeds = [int(s) for s in args.seeds.split(',')]
  os.makedirs(args.out, exist_ok=True)
  base_cfg = {'alpha': 1.0, 'beta': 0.0, 'variant': 'change_rate',
              'eps': args.eps, 'gamma': 0.9}
  ours_cfg = dict(base_cfg, beta=args.beta)

  print('baseline (beta=0)...')
  base_sr, base_len = run_condition(args.checkpoint, base_cfg, seeds, args.iters,
                                    args.env_steps, args.lr, args.max_steps)
  print(f'ours (beta={args.beta})...')
  ours_sr, ours_len = run_condition(args.checkpoint, ours_cfg, seeds, args.iters,
                                    args.env_steps, args.lr, args.max_steps)

  it = np.arange(args.iters)
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))
  for arr, name, color in [(base_sr, 'baseline (β=0)', 'C0'),
                           (ours_sr, f'ours (β={args.beta})', 'C3')]:
    m, sd = arr.mean(0), arr.std(0)
    axes[0].plot(it, m, color=color, label=name)
    axes[0].fill_between(it, m - sd, m + sd, color=color, alpha=0.2)
  axes[0].set_xlabel('REINFORCE iteration'); axes[0].set_ylabel('success rate')
  axes[0].set_title('성공률 (이미 천장 근처 — 참고용)'); axes[0].legend()

  for arr, name, color in [(base_len, 'baseline (β=0)', 'C0'),
                           (ours_len, f'ours (β={args.beta})', 'C3')]:
    m, sd = arr.mean(0), arr.std(0)
    axes[1].plot(it, m, color=color, label=name)
    axes[1].fill_between(it, m - sd, m + sd, color=color, alpha=0.2)
  axes[1].set_xlabel('REINFORCE iteration'); axes[1].set_ylabel('평균 에피소드 길이')
  axes[1].set_title('효율 (짧을수록 좋음 — 핵심 지표)'); axes[1].legend()
  fig.suptitle(f'장애물 환경 E2: baseline vs ours ({len(seeds)} seeds, mean±std)')
  fig.tight_layout()
  p = os.path.join(args.out, 'e2_obstacle_convergence.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  def final(arr):
    return float(arr[:, -3:].mean())
  summary = {
      'seeds': seeds, 'iters': args.iters, 'beta': args.beta, 'eps': args.eps,
      'baseline_start_SR': float(base_sr[:, 0].mean()),
      'baseline_final_SR': final(base_sr),
      'ours_start_SR': float(ours_sr[:, 0].mean()),
      'ours_final_SR': final(ours_sr),
      'baseline_start_len': float(base_len[:, 0].mean()),
      'baseline_final_len': final(base_len),
      'ours_start_len': float(ours_len[:, 0].mean()),
      'ours_final_len': final(ours_len),
  }
  with open(os.path.join(args.out, 'e2_obstacle_summary.json'), 'w') as fp:
    json.dump(summary, fp, indent=2)
  with open(os.path.join(args.out, 'e2_obstacle_curves.csv'), 'w', newline='') as fp:
    w = csv.writer(fp)
    w.writerow(['arm', 'metric', 'seed'] + [f'it{i}' for i in it])
    for name, sr, ln in [('baseline', base_sr, base_len), ('ours', ours_sr, ours_len)]:
      for si, s in enumerate(seeds):
        w.writerow([name, 'success_rate', s] + sr[si].tolist())
        w.writerow([name, 'ep_len', s] + ln[si].tolist())
  print('summary:', json.dumps(summary, indent=2))
  print('plot:', p)


if __name__ == '__main__':
  main()
