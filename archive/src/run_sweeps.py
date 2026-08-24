"""Decision-item sweeps (spec section 2) + the E2 main comparison at a degraded
start (more headroom than the near-ceiling SFT policy).

  2.3 alpha:beta  — beta in {0, 0.25, 0.5, 1, 2}; beta=0 is the baseline. Shows
                    whether weighting the variance term helps and how much.
  2.1 variant     — change_rate vs ratio variance term (at a fixed beta).
  2.2 eps         — eps in {1e-6, 1e-3, 1e-1}; checks numerical stability (NaN /
                    divergence) of the change_rate term near sigma^2 -> 0.

Each condition runs several seeds; we report final success rate (mean of last 3
iters) ± std. The policy is degraded from the predictor checkpoint's SFT policy
so there is clear room to improve (ceiling effect otherwise hides the difference).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforce import train_reinforce  # noqa: E402


def _run(ckpt, cfg, seeds, iters, env_steps, lr, map_name, init_noise):
  finals, aucs, curves = [], [], []
  nan = False
  for s in seeds:
    c = train_reinforce(ckpt, cfg, num_iters=iters, env_steps_per_iter=env_steps,
                        seed=s, lr=lr, map_name=map_name, init_noise=init_noise)
    sr = np.array(c['success_rate'])
    if not np.all(np.isfinite(sr)):
      nan = True
    finals.append(float(sr[-3:].mean()))
    aucs.append(float(sr.mean()))
    curves.append(sr.tolist())
  return {'final_mean': float(np.mean(finals)), 'final_std': float(np.std(finals)),
          'finals': finals, 'auc_mean': float(np.mean(aucs)), 'nan': nan,
          'curves': curves}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--map', default='standard')
  ap.add_argument('--seeds', default='0,1,2,3,4')
  ap.add_argument('--iters', type=int, default=15)
  ap.add_argument('--env_steps', type=int, default=1200)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--init_noise', type=float, default=0.02)
  ap.add_argument('--out', default='results/sweeps')
  args = ap.parse_args()
  seeds = [int(s) for s in args.seeds.split(',')]
  os.makedirs(args.out, exist_ok=True)
  common = dict(ckpt=args.checkpoint, seeds=seeds, iters=args.iters,
                env_steps=args.env_steps, lr=args.lr, map_name=args.map,
                init_noise=args.init_noise)
  results = {}

  # --- 2.3 beta sweep (contains baseline beta=0) ---
  print('=== 2.3 beta sweep ===')
  betas = [0.0, 0.25, 0.5, 1.0, 2.0]
  beta_res = {}
  for b in betas:
    cfg = {'alpha': 1.0, 'beta': b, 'variant': 'change_rate', 'eps': 1e-6,
           'gamma': 0.9}
    r = _run(cfg=cfg, **common)
    beta_res[b] = r
    print(f'  beta={b}: final SR {r["final_mean"]:.3f}±{r["final_std"]:.3f}')
  results['beta_sweep'] = beta_res

  # --- 2.1 variant (change_rate vs ratio) at beta=1 ---
  print('=== 2.1 variant ===')
  var_res = {}
  for v in ['change_rate', 'ratio']:
    cfg = {'alpha': 1.0, 'beta': 1.0, 'variant': v, 'eps': 1e-6, 'gamma': 0.9}
    r = _run(cfg=cfg, **common)
    var_res[v] = r
    print(f'  variant={v}: final SR {r["final_mean"]:.3f}±{r["final_std"]:.3f}')
  results['variant'] = var_res

  # --- 2.2 eps stability at beta=1 ---
  print('=== 2.2 eps ===')
  eps_res = {}
  for e in [1e-6, 1e-3, 1e-1]:
    cfg = {'alpha': 1.0, 'beta': 1.0, 'variant': 'change_rate', 'eps': e,
           'gamma': 0.9}
    r = _run(cfg=cfg, **common)
    eps_res[str(e)] = r
    print(f'  eps={e}: final SR {r["final_mean"]:.3f}±{r["final_std"]:.3f} nan={r["nan"]}')
  results['eps'] = eps_res

  # --- plots ---
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
  bx = list(beta_res)
  bm = [beta_res[b]['final_mean'] for b in bx]
  bs = [beta_res[b]['final_std'] for b in bx]
  axes[0].errorbar(bx, bm, yerr=bs, marker='o', capsize=4, color='C3')
  axes[0].axhline(beta_res[0.0]['final_mean'], color='C0', ls='--',
                  label='baseline (β=0)')
  axes[0].set_xlabel('β (variance weight)'); axes[0].set_ylabel('final success rate')
  axes[0].set_title('2.3 α:β sweep'); axes[0].legend()

  vx = list(var_res)
  axes[1].bar(vx, [var_res[v]['final_mean'] for v in vx],
              yerr=[var_res[v]['final_std'] for v in vx], capsize=4,
              color=['C3', 'C2'])
  axes[1].set_title('2.1 variance-term variant (β=1)')
  axes[1].set_ylabel('final success rate')

  ex = list(eps_res)
  axes[2].bar(ex, [eps_res[e]['final_mean'] for e in ex],
              yerr=[eps_res[e]['final_std'] for e in ex], capsize=4, color='C1')
  axes[2].set_title('2.2 eps stability (β=1)')
  axes[2].set_ylabel('final success rate')

  fig.suptitle(f'Decision-item sweeps ({len(seeds)} seeds, degraded start)',
               fontweight='bold')
  fig.tight_layout()
  p = os.path.join(args.out, 'sweeps.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  # strip heavy curves before json dump
  slim = json.loads(json.dumps(results))
  for grp in slim.values():
    for cond in grp.values():
      cond.pop('curves', None)
  with open(os.path.join(args.out, 'sweeps_summary.json'), 'w') as fp:
    json.dump(slim, fp, indent=2)
  print('SWEEP SUMMARY:', json.dumps(slim, indent=2))
  print('plot:', p)


if __name__ == '__main__':
  main()
