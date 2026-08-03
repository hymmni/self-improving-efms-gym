"""E3 — policy improvement vs predictor quality (spec section 4).

For each predictor checkpoint (trained to a different fraction, hence different
MAE/NLL), run REINFORCE with baseline vs ours and record final success rate.
Question: does the variance-augmented reward help (relative to baseline) even
when the predictor is weak? Plots final SR vs predictor MAE for both arms.

Run: python -m src.run_e3 --ckpt_dir checkpoints/std --seeds 0,1,2 --iters 12
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforce import train_reinforce  # noqa: E402


def _final_sr(ckpt, cfg, seeds, iters, env_steps, lr, map_name, init_noise):
  fins = []
  for s in seeds:
    c = train_reinforce(ckpt, cfg, num_iters=iters, env_steps_per_iter=env_steps,
                        seed=s, lr=lr, map_name=map_name, init_noise=init_noise)
    fins.append(float(np.array(c['success_rate'])[-3:].mean()))
  return float(np.mean(fins)), float(np.std(fins))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt_dir', default='checkpoints/std')
  ap.add_argument('--map', default='standard')
  ap.add_argument('--seeds', default='0,1,2')
  ap.add_argument('--iters', type=int, default=12)
  ap.add_argument('--env_steps', type=int, default=1200)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--init_noise', type=float, default=0.02)
  ap.add_argument('--beta', type=float, default=1.0)
  ap.add_argument('--out', default='results/e3')
  args = ap.parse_args()
  seeds = [int(s) for s in args.seeds.split(',')]
  os.makedirs(args.out, exist_ok=True)

  ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'predictor_f*.pkl')))
  base_cfg = {'alpha': 1.0, 'beta': 0.0, 'variant': 'change_rate',
              'eps': 1e-6, 'gamma': 0.9}
  ours_cfg = dict(base_cfg, beta=args.beta)

  rows = []
  for ck in ckpts:
    meta = pickle.load(open(ck, 'rb'))['meta']
    mae = meta['predictor_mae']; frac = meta['fraction']
    bm, bs = _final_sr(ck, base_cfg, seeds, args.iters, args.env_steps, args.lr,
                       args.map, args.init_noise)
    om, os_ = _final_sr(ck, ours_cfg, seeds, args.iters, args.env_steps, args.lr,
                        args.map, args.init_noise)
    rows.append({'fraction': frac, 'mae': mae, 'baseline_SR': bm,
                 'baseline_std': bs, 'ours_SR': om, 'ours_std': os_})
    print(f'  f={frac:.2f} MAE={mae:.2f}: baseline {bm:.3f}±{bs:.3f}  '
          f'ours {om:.3f}±{os_:.3f}  (ours-base {om-bm:+.3f})')

  rows.sort(key=lambda r: r['mae'])
  maes = [r['mae'] for r in rows]
  fig, ax = plt.subplots(figsize=(8, 5))
  ax.errorbar(maes, [r['baseline_SR'] for r in rows],
              yerr=[r['baseline_std'] for r in rows], marker='o', capsize=4,
              color='C0', label='baseline (β=0)')
  ax.errorbar(maes, [r['ours_SR'] for r in rows],
              yerr=[r['ours_std'] for r in rows], marker='s', capsize=4,
              color='C3', label=f'ours (β={args.beta})')
  ax.set_xlabel('predictor MAE (higher = weaker predictor)')
  ax.set_ylabel('final success rate')
  ax.set_title('E3: policy improvement vs predictor quality')
  ax.legend(); ax.invert_xaxis()  # stronger predictor on the right
  fig.tight_layout()
  p = os.path.join(args.out, 'e3_predictor_quality.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  with open(os.path.join(args.out, 'e3_summary.json'), 'w') as fp:
    json.dump(rows, fp, indent=2)
  print('E3 rows:', json.dumps(rows, indent=2))
  print('plot:', p)


if __name__ == '__main__':
  main()
