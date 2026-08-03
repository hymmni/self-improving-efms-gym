"""E2 — main comparison: baseline (eq 1.1) vs ours (eq 1.2) reward in REINFORCE.

Same steps-to-go predictor, same everything except the reward. Runs multiple
seeds and reports the success-rate convergence curve (mean ± std). Also supports
the alpha:beta sweep (decision item 2.3) and the variance-term variant sweep
(decision item 2.1).

Run: python -m src.run_e2 --checkpoint checkpoints/std/predictor_f100.pkl \
       --map standard --seeds 0,1,2,3,4 --iters 30
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforce import train_reinforce  # noqa: E402


def run_condition(ckpt, reward_cfg, seeds, iters, env_steps, lr, map_name,
                  init_noise):
  curves = []
  for s in seeds:
    c = train_reinforce(ckpt, reward_cfg, num_iters=iters,
                        env_steps_per_iter=env_steps, seed=s, lr=lr,
                        map_name=map_name, init_noise=init_noise)
    curves.append(c['success_rate'])
    print(f'    seed {s}: SR {c["success_rate"][0]:.2f} -> {c["success_rate"][-1]:.2f}')
  return np.array(curves)  # (n_seeds, iters)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--map', default='standard')
  ap.add_argument('--seeds', default='0,1,2,3,4')
  ap.add_argument('--iters', type=int, default=30)
  ap.add_argument('--env_steps', type=int, default=2048)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--init_noise', type=float, default=0.0)
  ap.add_argument('--beta', type=float, default=0.5, help='beta for the ours arm')
  ap.add_argument('--out', default='results/e2')
  args = ap.parse_args()

  seeds = [int(s) for s in args.seeds.split(',')]
  os.makedirs(args.out, exist_ok=True)
  base_cfg = {'alpha': 1.0, 'beta': 0.0, 'variant': 'change_rate',
              'eps': 1e-6, 'gamma': 0.9}
  ours_cfg = dict(base_cfg, beta=args.beta)

  print('baseline (beta=0)...')
  base = run_condition(args.checkpoint, base_cfg, seeds, args.iters,
                       args.env_steps, args.lr, args.map, args.init_noise)
  print(f'ours (beta={args.beta})...')
  ours = run_condition(args.checkpoint, ours_cfg, seeds, args.iters,
                       args.env_steps, args.lr, args.map, args.init_noise)

  it = np.arange(args.iters)
  fig, ax = plt.subplots(figsize=(8, 5))
  for arr, name, color in [(base, 'baseline (β=0)', 'C0'),
                           (ours, f'ours (β={args.beta})', 'C3')]:
    m, sd = arr.mean(0), arr.std(0)
    ax.plot(it, m, color=color, label=name)
    ax.fill_between(it, m - sd, m + sd, color=color, alpha=0.2)
  ax.set_xlabel('REINFORCE iteration'); ax.set_ylabel('success rate')
  ax.set_title(f'E2: baseline vs ours reward ({len(seeds)} seeds, mean±std)')
  ax.legend(); fig.tight_layout()
  p = os.path.join(args.out, 'e2_convergence.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  # summary metrics
  def final(arr):  # mean of last 3 iters
    return float(arr[:, -3:].mean())
  def auc(arr):    # area under mean curve (convergence speed proxy)
    return float(arr.mean(0).mean())
  summary = {
      'seeds': seeds, 'iters': args.iters, 'beta': args.beta,
      'baseline_final_SR': final(base), 'ours_final_SR': final(ours),
      'baseline_AUC': auc(base), 'ours_AUC': auc(ours),
      'baseline_start_SR': float(base[:, 0].mean()),
      'ours_start_SR': float(ours[:, 0].mean()),
  }
  with open(os.path.join(args.out, 'e2_summary.json'), 'w') as fp:
    json.dump(summary, fp, indent=2)
  with open(os.path.join(args.out, 'e2_curves.csv'), 'w', newline='') as fp:
    w = csv.writer(fp); w.writerow(['arm', 'seed'] + [f'it{i}' for i in it])
    for name, arr in [('baseline', base), ('ours', ours)]:
      for si, s in enumerate(seeds):
        w.writerow([name, s] + arr[si].tolist())
  print('E2 summary:', json.dumps(summary, indent=2))
  print('plot:', p)


if __name__ == '__main__':
  main()
