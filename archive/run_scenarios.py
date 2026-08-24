"""Scenario battery: probe the steps-to-go distribution under systematic
interventions and dump trend plots + a raw-metrics CSV.

This is the observational evidence for the "variance is informative and unused"
argument in references/context_2.md, and the input to the later reward-design
phase. Seven scenarios (baseline + teleport away/near + small/large bias +
obstacle + random action) are each rolled out `--episodes` times with the learned
Stage-1 policy; per step we record the full STG distribution via stg_probe.

Outputs (all under --out, gitignored):
  <scenario>/episode_*.png   three example episodes (distribution + curves)
  <scenario>/summary.png     E and variance curves overlaid (success blue / fail red)
  summary.png                success rate / mean length / intervention Δvariance
  summary.csv                per-episode raw metrics for re-analysis
"""

import argparse
import csv
import os

import numpy as np
import jax

from envs_enhanced import EnhancedPoint2D
from stg_probe import STGProbe, plot_episode


# --------------------------------------------------------------- scenario defs
def _far_corner(goal):
  corners = np.array([[-0.9, -0.9], [-0.9, 0.9], [0.9, -0.9], [0.9, 0.9]],
                     dtype=np.float32)
  d = np.linalg.norm(corners - goal[None], axis=1)
  return corners[int(np.argmax(d))]


def _cur_E(probe, env):
  obs = {'cur_pos': env._cur_pos.copy(), 'cur_vel': env._cur_vel.copy(),
         'goal_pos': env._goal_pos.copy()}
  return probe.query(obs).expectation


def make_scenarios(bias_small, bias_large, obstacle_radius, random_prob,
                   random_scale):
  def setup_none(probe, env, st):
    pass

  def setup_bias_small(probe, env, st):
    env.set_bias_force(np.array(bias_small, dtype=np.float32))

  def setup_bias_large(probe, env, st):
    env.set_bias_force(np.array(bias_large, dtype=np.float32))

  def setup_obstacle(probe, env, st):
    mid = 0.5 * (env._cur_pos + env._goal_pos)
    env.add_obstacle(mid.astype(np.float32), obstacle_radius)
    st['intervention_step'] = 0

  def setup_random(probe, env, st):
    env.set_random_action(random_prob, random_scale, seed=st['seed'])

  def setup_teleport_near(probe, env, st):
    off = np.random.uniform(-0.3, 0.3, size=2).astype(np.float32)
    env.teleport((env._goal_pos + off).astype(np.float32), zero_velocity=True)
    st['intervention_step'] = 0

  def on_teleport_away(probe, env, step_idx, st):
    if st.get('fired'):
      return
    E0 = st.setdefault('E0', _cur_E(probe, env))
    if step_idx >= 1 and _cur_E(probe, env) < 0.5 * E0:
      env.teleport(_far_corner(env._goal_pos), zero_velocity=True)
      st['fired'] = True
      st['intervention_step'] = step_idx

  def noop(probe, env, step_idx, st):
    pass

  return {
      'baseline':      dict(setup=setup_none,        on_step=noop),
      'teleport_away': dict(setup=setup_none,        on_step=on_teleport_away),
      'teleport_near': dict(setup=setup_teleport_near, on_step=noop),
      'bias_small':    dict(setup=setup_bias_small,  on_step=noop),
      'bias_large':    dict(setup=setup_bias_large,  on_step=noop),
      'obstacle_path': dict(setup=setup_obstacle,    on_step=noop),
      'random_act':    dict(setup=setup_random,      on_step=noop),
  }


# ------------------------------------------------------------- episode rollout
def run_scenario_episode(probe, scenario, seed, max_steps=140):
  """Rollout mirroring STGProbe.rollout but with a post-reset setup hook and a
  per-step intervention hook, so scenarios that depend on the sampled start/goal
  (obstacle placement, teleport-near) can be configured."""
  np.random.seed(seed)
  key = jax.random.PRNGKey(42)
  env = EnhancedPoint2D()
  ts = env.reset()
  st = {'seed': seed, 'intervention_step': -1}
  scenario['setup'](probe, env, st)

  records = []
  step_idx = 0
  while (not env.success()) and step_idx < max_steps:
    obs = ts.observation
    key, sub = jax.random.split(key)
    act, logits = probe._logits_and_act(obs, sub)
    records.append(probe._record(step_idx, obs, logits))
    scenario['on_step'](probe, env, step_idx, st)
    action = probe.unnormalize_action(act)[0]
    ts = env.step(action)
    step_idx += 1

  obs = ts.observation
  key, sub = jax.random.split(key)
  _, logits = probe._logits_and_act(obs, sub)
  records.append(probe._record(step_idx, obs, logits))
  success = bool(np.linalg.norm(
      records[-1].obs['cur_pos'] - records[-1].obs['goal_pos']) < 0.15)
  return records, st, success, env


# ------------------------------------------------------------------- metrics
def _around(records, istep):
  """(E_before, var_before, E_after, var_after) around an intervention step."""
  if istep is None or istep < 0:
    return (np.nan, np.nan, np.nan, np.nan)
  idx = min(istep, len(records) - 1)
  before = records[max(0, idx - 1)]
  after = records[min(len(records) - 1, idx + 1)]
  return (before.expectation, before.variance, after.expectation,
          after.variance)


def _scenario_summary_plot(name, episodes, out_path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, (axE, axV) = plt.subplots(1, 2, figsize=(12, 4))
  for records, st, success in episodes:
    steps = [r.step_idx for r in records]
    color = 'C0' if success else 'C3'
    axE.plot(steps, [r.expectation for r in records], color=color, alpha=0.5)
    axV.plot(steps, [r.variance for r in records], color=color, alpha=0.5)
    istep = st.get('intervention_step', -1)
    if istep is not None and istep >= 0:
      axE.axvline(istep, color='k', linestyle=':', alpha=0.3)
  axE.set_title(f'{name}: E[STG] (blue=success, red=fail)')
  axE.set_xlabel('step'); axE.set_ylabel('E[STG]')
  axV.set_title(f'{name}: variance')
  axV.set_xlabel('step'); axV.set_ylabel('variance')
  fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def _overall_summary_plot(rows, scenarios, out_path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  names = list(scenarios)
  succ, length, dvar = [], [], []
  for n in names:
    rs = [r for r in rows if r['scenario'] == n]
    succ.append(np.mean([r['success'] for r in rs]))
    length.append(np.mean([r['len'] for r in rs]))
    dv = [r['var_after'] - r['var_before'] for r in rs
          if not np.isnan(r['var_before'])]
    dvar.append(np.mean(dv) if dv else 0.0)
  fig, axs = plt.subplots(1, 3, figsize=(16, 4))
  axs[0].bar(names, succ, color='C0'); axs[0].set_title('success rate')
  axs[1].bar(names, length, color='C1'); axs[1].set_title('mean episode length')
  axs[2].bar(names, dvar, color='C3')
  axs[2].set_title('mean Δvariance around intervention')
  for ax in axs:
    ax.tick_params(axis='x', rotation=45)
  fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
  parser = argparse.ArgumentParser(description='STG scenario battery.')
  parser.add_argument('--checkpoint', default='checkpoints/sft_state.pkl')
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--max_steps', type=int, default=140)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', default='outputs/scenarios/')
  parser.add_argument('--only', default=None,
                      help='comma-separated scenario subset')
  # Bias is added every physics substep (10x/step) so it accumulates fast;
  # these magnitudes were calibrated so small keeps ~0.87 success while large
  # drops to ~0.40 (many failures) — see experiments/2026-07-13_scenario-battery.md.
  parser.add_argument('--bias_small', type=float, nargs=2, default=(0.00002, 0.0))
  parser.add_argument('--bias_large', type=float, nargs=2, default=(0.00005, 0.0))
  parser.add_argument('--obstacle_radius', type=float, default=0.2)
  parser.add_argument('--random_prob', type=float, default=0.1)
  parser.add_argument('--random_scale', type=float, default=0.0015)
  args = parser.parse_args()

  probe = STGProbe(args.checkpoint)
  scenarios = make_scenarios(args.bias_small, args.bias_large,
                             args.obstacle_radius, args.random_prob,
                             args.random_scale)
  if args.only:
    want = set(args.only.split(','))
    scenarios = {k: v for k, v in scenarios.items() if k in want}

  os.makedirs(args.out, exist_ok=True)
  rows = []
  for name, scen in scenarios.items():
    print(f'=== scenario: {name} ===')
    sdir = os.path.join(args.out, name)
    os.makedirs(sdir, exist_ok=True)
    episodes = []
    for ep in range(args.episodes):
      seed = args.seed + ep
      records, st, success, env = run_scenario_episode(
          probe, scen, seed, args.max_steps)
      episodes.append((records, st, success))
      Eb, Vb, Ea, Va = _around(records, st.get('intervention_step', -1))
      rows.append({
          'scenario': name, 'seed': seed, 'success': int(success),
          'len': records[-1].step_idx,
          'intervention_step': st.get('intervention_step', -1),
          'E_before': Eb, 'var_before': Vb, 'E_after': Ea, 'var_after': Va,
          'final_E': records[-1].expectation,
          'final_var': records[-1].variance,
      })
      if ep < 3:
        traj = env.render(title=f'{name} ep{ep}')
        plot_episode(records, probe.bin_vals, env_traj_img=traj,
                     out_path=os.path.join(sdir, f'episode_{ep:02d}.png'),
                     title=f'{name} ep{ep} success={success}')
    _scenario_summary_plot(name, episodes, os.path.join(sdir, 'summary.png'))
    sr = np.mean([e[2] for e in episodes])
    print(f'  success rate={sr:.2f}  mean len='
          f'{np.mean([r["len"] for r in rows if r["scenario"]==name]):.1f}')

  # write CSV + overall summary
  csv_path = os.path.join(args.out, 'summary.csv')
  with open(csv_path, 'w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  _overall_summary_plot(rows, scenarios, os.path.join(args.out, 'summary.png'))
  print(f'Wrote {csv_path} ({len(rows)} rows) and summary.png')


if __name__ == '__main__':
  main()
