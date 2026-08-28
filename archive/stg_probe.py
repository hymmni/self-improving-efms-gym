"""Steps-to-go (STG) distribution probe.

Records and visualizes the FULL categorical steps-to-go prediction distribution
(not just its expectation) along a rollout. This is the core observation tool for
the research question in references/context_2.md: how does the predicted
distribution's shape (flat / peaked / multimodal) and its variance behave as the
agent approaches, is perturbed away from, or fails to reach the goal.

Reused by step 4 (interactive control) and step 5 (scenario battery).

The expectation/variance are computed on exactly the same bin values the training
converter uses (pointmass_core.build_discrete_distance_converter:
`dist_vals = linspace(min, max, num_bins+1)[:-1]`), so the numbers are directly
comparable to the Stage-2 REINFORCE reward signal.
"""

import argparse
import glob
import os
import pickle
from typing import Callable, NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

from pointmass_core import (
    Point2D,
    build_continuous_act_discrete_dist_v0,
    build_discrete_distance_converter,
    make_normalizers,
)


class STGRecord(NamedTuple):
  step_idx: int
  obs: dict            # un-normalized observation (cur_pos, cur_vel, goal_pos)
  probs: np.ndarray    # (num_bins,) softmax probabilities
  expectation: float   # sum(bin_vals * probs)
  variance: float      # sum(bin_vals^2 * probs) - expectation^2
  entropy: float       # -sum(probs * log probs)


class STGProbe:
  """Loads an SFT checkpoint and queries the steps-to-go distribution."""

  def __init__(self, checkpoint_path: str):
    (self.params, self.normalize_obs, self.unnormalize_action,
     self.dc, self.nets, self.bin_vals, self.meta) = load_checkpoint(
         checkpoint_path)

    def _infer(params, norm_obs_concat, rng):
      preds = self.nets.network.apply(params, norm_obs_concat)
      act = self.nets.sample_act(preds.act_dist_params, rng)
      return act, preds.dist_to_succ_dist_params.logits

    self._infer = jax.jit(_infer, backend='cpu')

  # ------------------------------------------------------------------ helpers
  def _logits_and_act(self, obs, rng):
    norm = self.normalize_obs(jax.tree.map(lambda x: x[None], obs))
    concat = jnp.concatenate(
        [norm['cur_pos'], norm['cur_vel'], norm['goal_pos']], axis=-1)
    act, logits = self._infer(self.params, concat, rng)
    return np.asarray(act), np.asarray(logits)[0]

  def _record(self, step_idx, obs, logits):
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    exp = float(np.sum(self.bin_vals * probs))
    var = float(np.sum(self.bin_vals ** 2 * probs) - exp ** 2)
    ent = float(-np.sum(probs * np.log(probs + 1e-12)))
    return STGRecord(step_idx, {k: np.asarray(v).copy() for k, v in obs.items()},
                     probs, exp, var, ent)

  # ------------------------------------------------------------------ queries
  def query(self, obs: dict, rng=None) -> STGRecord:
    """Distribution for a single observation, without stepping the env."""
    if rng is None:
      rng = jax.random.PRNGKey(0)
    _, logits = self._logits_and_act(obs, rng)
    return self._record(-1, obs, logits)

  def rollout(self, env, max_steps: int = 140, policy: str = 'learned',
              seed: int = 0, action_source: Optional[Callable] = None,
              intervention_fn: Optional[Callable] = None) -> list:
    """Runs one episode, recording an STGRecord per step.

    policy='learned': action = sampled TIMER policy action (un-normalized).
    policy='external': action = action_source(obs) each step.
    intervention_fn(env, step_idx): optional callback to trigger interventions
    (used by the scenario battery); called before each env.step.
    """
    np.random.seed(seed)
    key = jax.random.PRNGKey(42)
    ts = env.reset()
    records = []
    step_idx = 0
    while (not env.success()) and step_idx < max_steps:
      obs = ts.observation
      key, sub = jax.random.split(key)
      act, logits = self._logits_and_act(obs, sub)
      records.append(self._record(step_idx, obs, logits))

      if policy == 'learned':
        action = self.unnormalize_action(act)[0]
      elif policy == 'external':
        if action_source is None:
          raise ValueError("policy='external' requires action_source")
        action = np.asarray(action_source(obs), dtype=np.float32)
      else:
        raise ValueError(f'unknown policy {policy!r}')

      if intervention_fn is not None:
        intervention_fn(env, step_idx)
      ts = env.step(action)
      step_idx += 1

    # final record at the terminal observation
    obs = ts.observation
    key, sub = jax.random.split(key)
    _, logits = self._logits_and_act(obs, sub)
    records.append(self._record(step_idx, obs, logits))
    return records


def load_checkpoint(path: str):
  """Rebuilds everything needed to probe from a checkpoint dict."""
  with open(path, 'rb') as fp:
    ck = pickle.load(fp)
  dc_cfg = ck['dc_config']
  dc = build_discrete_distance_converter(
      dc_cfg['min_distance'], dc_cfg['max_distance'], dc_cfg['num_bins'])
  bin_vals = np.linspace(
      dc_cfg['min_distance'], dc_cfg['max_distance'], dc_cfg['num_bins'] + 1,
      endpoint=True, dtype=np.float32)[:-1]
  nets = build_continuous_act_discrete_dist_v0(
      (256, 256, 256), 2, dc_cfg['num_bins'], np.ones((4, 6), dtype=np.float32))
  normalize_obs, _, unnormalize_action = make_normalizers(ck['norm_stats'])
  return (ck['params'], normalize_obs, unnormalize_action, dc, nets, bin_vals,
          ck.get('meta', {}))


# --------------------------------------------------------------------- plots
def plot_episode(records: list, bin_vals: np.ndarray,
                 env_traj_img: Optional[np.ndarray] = None,
                 out_path: str = 'episode.png', title: str = ''):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  steps = np.array([r.step_idx for r in records])
  exps = np.array([r.expectation for r in records])
  varis = np.array([r.variance for r in records])
  ents = np.array([r.entropy for r in records])
  stds = np.sqrt(np.clip(varis, 0, None))
  prob_mat = np.stack([r.probs for r in records], axis=1)  # (num_bins, T)

  fig = plt.figure(figsize=(16, 4))
  gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1.2, 1.2, 1.0])

  # (a) distribution evolution heatmap
  ax0 = fig.add_subplot(gs[0, 0])
  im = ax0.imshow(prob_mat, aspect='auto', origin='lower', cmap='viridis',
                  extent=[steps.min(), steps.max(), bin_vals.min(),
                          bin_vals.max()])
  ax0.set_title('STG distribution over steps')
  ax0.set_xlabel('step'); ax0.set_ylabel('predicted steps-to-go (bin value)')
  fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label='prob')

  # (b) expectation ± std
  ax1 = fig.add_subplot(gs[0, 1])
  ax1.plot(steps, exps, color='C0', label='E[STG]')
  ax1.fill_between(steps, exps - stds, exps + stds, color='C0', alpha=0.2,
                   label='±std')
  ax1.set_title('Expectation ± std'); ax1.set_xlabel('step'); ax1.legend()

  # (c) variance & entropy
  ax2 = fig.add_subplot(gs[0, 2])
  ax2.plot(steps, varis, color='C3', label='variance')
  ax2b = ax2.twinx()
  ax2b.plot(steps, ents, color='C2', linestyle='--', label='entropy')
  ax2.set_title('Variance / Entropy'); ax2.set_xlabel('step')
  ax2.set_ylabel('variance', color='C3'); ax2b.set_ylabel('entropy', color='C2')

  # (d) env trajectory
  ax3 = fig.add_subplot(gs[0, 3])
  if env_traj_img is not None:
    ax3.imshow(env_traj_img); ax3.axis('off'); ax3.set_title('trajectory')
  else:
    pos = np.array([r.obs['cur_pos'] for r in records])
    goal = records[-1].obs['goal_pos']
    ax3.plot(pos[:, 0], pos[:, 1], '.-', color='blue')
    ax3.scatter(goal[0], goal[1], marker='*', s=200, color='orange')
    ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); ax3.set_aspect('equal')
    ax3.set_title('trajectory')

  if title:
    fig.suptitle(title, fontsize=14, fontweight='bold')
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  return out_path


def plot_distribution_frames(records: list, bin_vals: np.ndarray, out_dir: str):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  os.makedirs(out_dir, exist_ok=True)
  paths = []
  for r in records:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(bin_vals, r.probs, width=(bin_vals[1] - bin_vals[0]) * 0.9,
           color='C0')
    ax.set_title(f'step {r.step_idx}  E={r.expectation:.1f}  var={r.variance:.1f}')
    ax.set_xlabel('steps-to-go'); ax.set_ylabel('prob')
    fig.tight_layout()
    p = os.path.join(out_dir, f'frame_{r.step_idx:04d}.png')
    fig.savefig(p, dpi=90); plt.close(fig); paths.append(p)
  return paths


def main():
  parser = argparse.ArgumentParser(description='Steps-to-go distribution probe.')
  parser.add_argument('--checkpoint', default='checkpoints/sft_state.pkl')
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--max_steps', type=int, default=140)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', default='outputs/probe/')
  args = parser.parse_args()

  probe = STGProbe(args.checkpoint)
  print(f'Loaded checkpoint (meta: {probe.meta})')
  os.makedirs(args.out, exist_ok=True)
  for ep in range(args.episodes):
    env = Point2D()
    records = probe.rollout(env, max_steps=args.max_steps, policy='learned',
                            seed=args.seed + ep)
    traj_img = env.render(title=f'episode {ep}')
    out_path = os.path.join(args.out, f'episode_{ep:02d}.png')
    plot_episode(records, probe.bin_vals, env_traj_img=traj_img,
                 out_path=out_path, title=f'episode {ep}')
    success = np.linalg.norm(
        records[-1].obs['cur_pos'] - records[-1].obs['goal_pos']) < 0.15
    print(f'  episode {ep}: steps={records[-1].step_idx} '
          f'success={success} final_E={records[-1].expectation:.1f} '
          f'-> {out_path}')

  print('Probe plots:', sorted(glob.glob(os.path.join(args.out, '*.png'))))


if __name__ == '__main__':
  main()
