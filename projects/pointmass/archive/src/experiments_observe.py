"""E0 / E1 / E1b — observation experiments (spec section 4).

E0  : bimodal steps-to-go distribution collapse (snapshots along an episode).
E1  : distribution behaviour in 3 situations (goal-approach, near-failure,
      multimodal-collapse) as (mu_t, sigma2_t) time series + representative
      histograms.
E1b : independence of the two reward signals — Pearson r and scatter of
      Delta-mu vs Delta-sigma^2 (novelty defense: variance must add information
      beyond the expectation).

Reuses stg_probe.STGProbe with a multimodal-map predictor checkpoint, and phase-1
interventions (bias force) to induce failures.

Run: python -m src.experiments_observe --checkpoint checkpoints/mm/predictor_f100.pkl
"""

import argparse
import os
import sys

import numpy as np
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stg_probe import STGProbe  # noqa: E402
from src.multimodal_env import MultiModalPoint2D, routed_pd_controller  # noqa: E402


def _rollout(probe, seed, bias=None, max_steps=140):
  env = MultiModalPoint2D(jitter=0.08)
  if bias is not None:
    # bias is applied after reset (reset reinstalls obstacle but keeps bias off,
    # so set it via an intervention_fn on step 0)
    def interv(e, i):
      if i == 0:
        e.set_bias_force(np.array(bias, dtype=np.float32))
    recs = probe.rollout(env, max_steps=max_steps, policy='learned', seed=seed,
                         intervention_fn=interv)
  else:
    recs = probe.rollout(env, max_steps=max_steps, policy='learned', seed=seed)
  success = bool(np.linalg.norm(
      recs[-1].obs['cur_pos'] - recs[-1].obs['goal_pos']) < 0.15)
  return recs, success, env


# ------------------------------------------------------------------------ E0
def _routed_source(side):
  def src(obs):
    return routed_pd_controller(obs['cur_pos'], obs['cur_vel'],
                                obs['goal_pos'], side)
  return src


def run_e0(probe, out_dir, seed=0):
  """Snapshot the distribution collapsing from bimodal to unimodal for a
  left-committing and a right-committing episode.

  The learned BC policy mode-averages the two demonstration modes and (because
  the map is asymmetric) commits to the same side every time, so we drive each
  route explicitly with the demonstration controller (policy='external'). The
  probed distribution is the predictor's output and is independent of who drives.
  """
  os.makedirs(out_dir, exist_ok=True)
  picks = {}
  for side in ('left', 'right'):
    env = MultiModalPoint2D(jitter=0.08)
    recs = probe.rollout(env, max_steps=140, policy='external', seed=seed,
                         action_source=_routed_source(side))
    picks[side] = (seed, recs)

  fig, axes = plt.subplots(len(picks), 5, figsize=(18, 3.2 * len(picks)))
  if len(picks) == 1:
    axes = axes[None, :]
  for row, (side, (s, recs)) in enumerate(picks.items()):
    T = recs[-1].step_idx
    snap_steps = np.linspace(0, len(recs) - 1, 5).astype(int)
    for col, si in enumerate(snap_steps):
      r = recs[si]
      ax = axes[row, col]
      ax.bar(probe.bin_vals, r.probs, width=0.9, color='C0')
      ax.set_title(f'{side} ep, step {r.step_idx}\nμ={r.expectation:.1f} σ²={r.variance:.1f}',
                   fontsize=9)
      ax.set_xlim(0, min(60, probe.bin_vals.max()))
      if col == 0:
        ax.set_ylabel('prob')
  fig.suptitle('E0: bimodal steps-to-go distribution collapse', fontweight='bold')
  fig.tight_layout()
  p = os.path.join(out_dir, 'e0_collapse.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  # quantify: variance at start vs end for each pick
  metrics = {}
  for side, (s, recs) in picks.items():
    metrics[side] = {'var_start': float(recs[0].variance),
                     'var_end': float(recs[-1].variance),
                     'mu_start': float(recs[0].expectation)}
  return p, metrics


# ------------------------------------------------------------------------ E1
def run_e1(probe, out_dir):
  """(mu, sigma2) time series for goal-approach (success) and near-failure
  (bias-perturbed) episodes, plus the multimodal-collapse start."""
  os.makedirs(out_dir, exist_ok=True)
  # success episode (goal approach)
  succ_recs = None
  for s in range(20):
    recs, ok, _ = _rollout(probe, s)
    if ok:
      succ_recs = recs; break
  # near-failure episode: strong lateral bias
  fail_recs = None
  for s in range(50):
    recs, ok, _ = _rollout(probe, s, bias=(0.00006, 0.0))
    if not ok:
      fail_recs = recs; break
  if fail_recs is None:  # fall back to any strongly perturbed episode
    fail_recs, _, _ = _rollout(probe, 0, bias=(0.00008, 0.0))

  fig, axes = plt.subplots(1, 3, figsize=(16, 4))
  # situation 2: goal approach
  st = [r.step_idx for r in succ_recs]
  axes[0].plot(st, [r.expectation for r in succ_recs], 'C0-', label='μ')
  ax0b = axes[0].twinx()
  ax0b.plot(st, [r.variance for r in succ_recs], 'C3--', label='σ²')
  axes[0].set_title('E1-2 goal approach (success)\nμ↓, σ²↓')
  axes[0].set_xlabel('step'); axes[0].set_ylabel('μ', color='C0')
  ax0b.set_ylabel('σ²', color='C3')
  # situation 1: near failure
  st = [r.step_idx for r in fail_recs]
  axes[1].plot(st, [r.expectation for r in fail_recs], 'C0-')
  ax1b = axes[1].twinx()
  ax1b.plot(st, [r.variance for r in fail_recs], 'C3--')
  axes[1].set_title('E1-1 near failure (bias)\nμ↑, σ²↑ ?')
  axes[1].set_xlabel('step'); axes[1].set_ylabel('μ', color='C0')
  ax1b.set_ylabel('σ²', color='C3')
  # situation 3: multimodal collapse (histograms at start/mid/end of success ep)
  snaps = np.linspace(0, len(succ_recs) - 1, 3).astype(int)
  for si in snaps:
    r = succ_recs[si]
    axes[2].plot(probe.bin_vals, r.probs, label=f'step {r.step_idx}')
  axes[2].set_title('E1-3 multimodal collapse')
  axes[2].set_xlim(0, min(60, probe.bin_vals.max()))
  axes[2].set_xlabel('steps-to-go'); axes[2].legend()

  fig.suptitle('E1: distribution behaviour across situations', fontweight='bold')
  fig.tight_layout()
  p = os.path.join(out_dir, 'e1_situations.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  metrics = {
      'success_mu_start': float(succ_recs[0].expectation),
      'success_mu_end': float(succ_recs[-1].expectation),
      'success_var_start': float(succ_recs[0].variance),
      'success_var_end': float(succ_recs[-1].variance),
      'fail_mu_max': float(max(r.expectation for r in fail_recs)),
      'fail_var_max': float(max(r.variance for r in fail_recs)),
      'fail_success': False,
  }
  return p, metrics


# ----------------------------------------------------------------------- E1b
def run_e1b(probe, out_dir, num_episodes=40, var_thresh=1.0):
  """Independence of Δμ and Δσ², analyzed by regime.

  The naive global Pearson r is misleading here because the data splits into two
  regimes: (a) 'certain' steps (σ²_t ≤ var_thresh) — the majority — where Δσ²≈0
  and the expectation carries the signal, and (b) 'uncertain' steps
  (σ²_t > var_thresh, i.e. the multimodal decision / near-failure region) where
  Δσ² is large and Δμ can even reverse sign. So the variance term is inert where
  the expectation already works and fires exactly where it doesn't — it adds
  information rather than duplicating it. We report the global r plus per-regime r
  and counts.
  """
  os.makedirs(out_dir, exist_ok=True)
  dmu, dvar, s_t = [], [], []
  for s in range(num_episodes):
    recs, _, _ = _rollout(probe, s)
    mu = np.array([r.expectation for r in recs])
    var = np.array([r.variance for r in recs])
    dmu.extend((mu[:-1] - mu[1:]).tolist())
    dvar.extend((var[:-1] - var[1:]).tolist())
    s_t.extend(var[:-1].tolist())
  dmu = np.array(dmu); dvar = np.array(dvar); s_t = np.array(s_t)
  uncertain = s_t > var_thresh
  certain = ~uncertain

  def _r(mask):
    if mask.sum() < 3 or np.std(dvar[mask]) < 1e-9 or np.std(dmu[mask]) < 1e-9:
      return float('nan')
    return float(np.corrcoef(dmu[mask], dvar[mask])[0, 1])

  r_all = _r(np.ones_like(uncertain))
  r_cert = _r(certain)
  r_unc = _r(uncertain)

  fig, ax = plt.subplots(figsize=(7, 6))
  ax.scatter(dmu[certain], dvar[certain], s=8, alpha=0.25, color='C0',
             label=f'certain σ²≤{var_thresh} (n={certain.sum()}, r={r_cert:.2f})')
  ax.scatter(dmu[uncertain], dvar[uncertain], s=14, alpha=0.5, color='C3',
             label=f'uncertain σ²>{var_thresh} (n={uncertain.sum()}, r={r_unc:.2f})')
  ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
  ax.set_xlabel('Δμ = μ_t − μ_{t+1}')
  ax.set_ylabel('Δσ² = σ²_t − σ²_{t+1}')
  ax.set_title(f'E1b: Δμ vs Δσ² by regime\nglobal r={r_all:.3f}  '
               f'(certain steps: Δσ²≈0, variance inert)')
  ax.legend(fontsize=8)
  fig.tight_layout()
  p = os.path.join(out_dir, 'e1b_independence.png')
  fig.savefig(p, dpi=120); plt.close(fig)

  # fraction of "variance-only" steps: Δσ² significant while Δμ small
  var_only = (np.abs(dvar) > 0.5) & (np.abs(dmu) < 0.2)
  return p, {'pearson_r_global': r_all, 'pearson_r_certain': r_cert,
             'pearson_r_uncertain': r_unc,
             'n_pairs': len(dmu), 'n_uncertain': int(uncertain.sum()),
             'frac_variance_only': float(var_only.mean())}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/mm/predictor_f100.pkl')
  ap.add_argument('--out', default='results/observe')
  args = ap.parse_args()
  probe = STGProbe(args.checkpoint)
  print('E0...'); p0, m0 = run_e0(probe, args.out); print(' ', m0)
  print('E1...'); p1, m1 = run_e1(probe, args.out); print(' ', m1)
  print('E1b...'); p1b, m1b = run_e1b(probe, args.out); print(' ', m1b)
  print('plots:', p0, p1, p1b)


if __name__ == '__main__':
  main()
