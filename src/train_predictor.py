"""Train steps-to-go predictor(s) on the multimodal map, snapshotting at several
training fractions for the E3 predictor-quality ablation.

Reuses the phase-1 SFT machinery (train_sft helpers + pointmass_core) but swaps in
the multimodal demonstration dataset. Each snapshot is a self-contained checkpoint
(same schema as train_sft) plus predictor-quality metrics (MAE, NLL) measured on a
held-out split, so E3 can plot policy improvement vs predictor accuracy.

Run: python -m src.train_predictor --fractions 0.1,0.3,0.5,1.0 --out_dir checkpoints/mm
"""

import argparse
import datetime
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmass_core import (  # noqa: E402
    build_continuous_act_discrete_dist_v0, build_discrete_distance_converter,
    PretrainLearner, compute_normalization_stats, make_normalizers,
    make_timer_rollout_policy)
from train_sft import make_dataset_from_tuples, evaluate_policy  # noqa: E402
from src.multimodal_env import MultiModalPoint2D, generate_multimodal_dataset  # noqa: E402

tf.config.set_visible_devices([], 'GPU')


def predictor_metrics(nets, params, dc, val_tuples, num_bins):
  """MAE (|E[STG]-true|) and NLL (-log p(true_bin)) on a held-out split."""
  obs = val_tuples.observation
  x = jnp.concatenate([obs['cur_pos'], obs['cur_vel'], obs['goal_pos']], axis=-1)
  preds = nets.network.apply(params, x)
  logits = preds.dist_to_succ_dist_params.logits
  mu = dc.network_format_to_distance(logits)  # (N,)
  ttg = val_tuples.time_to_success
  mae = float(np.mean(np.abs(np.asarray(mu) - np.asarray(ttg))))
  true_bin = np.asarray(dc.distance_to_network_format(ttg)).astype(int)
  logp = jax.nn.log_softmax(logits, axis=-1)
  nll = float(-np.mean(np.asarray(logp)[np.arange(len(true_bin)), true_bin]))
  return mae, nll


def main():
  ap = argparse.ArgumentParser(description='Train STG predictors on multimodal map.')
  ap.add_argument('--fractions', default='0.1,0.3,0.5,1.0')
  ap.add_argument('--num_steps', type=int, default=20000)
  ap.add_argument('--dataset_episodes', type=int, default=4000)
  ap.add_argument('--jitter', type=float, default=0.08)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval_episodes', type=int, default=50)
  ap.add_argument('--out_dir', default='checkpoints/mm')
  args = ap.parse_args()

  fractions = sorted(float(f) for f in args.fractions.split(','))
  np.random.seed(args.seed); tf.random.set_seed(args.seed)

  gmb, nmb = 256, 128
  # bin_size = 1 (max_distance == num_bins). This is deliberate: the notebook's
  # distance loss passes raw time_to_success as the Categorical class label (the
  # computed bin index is unused — a latent bug in the original). With bin_size=1,
  # class == floor(ttg/1) == ttg, so that convention becomes correct and the
  # predicted mean E[STG] is calibrated in real step units (see step-2 notes).
  # num_bins=64 safely covers the longest multimodal detour (~47 steps).
  min_d, max_d, nbins = 0, 64, 64
  layers = (256, 256, 256)
  print(f'JAX devices: {jax.devices()}')

  # dataset (multimodal)
  print(f'Generating multimodal dataset ({args.dataset_episodes} eps)...')
  _, all_tuples, sides = generate_multimodal_dataset(
      num_episodes=args.dataset_episodes, jitter=args.jitter, seed=args.seed)
  print(f'  sides={sides}, transitions={all_tuples.observation["cur_pos"].shape[0]}')

  norm_stats = compute_normalization_stats(all_tuples)
  normalize_obs, normalize_action, unnormalize_action = make_normalizers(norm_stats)
  norm_tuples = all_tuples._replace(
      observation=normalize_obs(all_tuples.observation),
      action=normalize_action(all_tuples.action))

  n = norm_tuples.observation['cur_pos'].shape[0]
  n_train = int(n * 0.9)
  train_tuples = jax.tree.map(lambda x: np.array(x[:n_train]), norm_tuples)
  val_tuples = jax.tree.map(lambda x: np.array(x[n_train:]), norm_tuples)
  # keep un-normalized ttg for MAE (norm doesn't touch ttg, but be explicit)
  val_for_metrics = all_tuples._replace(
      observation=normalize_obs(all_tuples.observation))
  val_for_metrics = jax.tree.map(lambda x: np.array(x[n_train:]), val_for_metrics)

  train_dataset = make_dataset_from_tuples(train_tuples, gmb, nmb)

  dc = build_discrete_distance_converter(min_d, max_d, nbins)
  nets = build_continuous_act_discrete_dist_v0(
      layers, 2, nbins, np.ones((4, 6), dtype=np.float32))
  optimizer = optax.chain(optax.scale_by_adam(eps=1e-7), optax.scale(-3e-4))
  learner = PretrainLearner(nets, dc, optimizer, jax.random.PRNGKey(42), gmb, nmb)
  rollout = make_timer_rollout_policy(nets, dc)

  os.makedirs(args.out_dir, exist_ok=True)
  num_iters = args.num_steps // nmb
  milestones = {int(round(f * num_iters)): f for f in fractions}
  print(f'Training {args.num_steps} steps; snapshots at fractions {fractions}')

  saved = []
  for i in range(1, num_iters + 1):
    learner.step(next(train_dataset))
    if i in milestones:
      frac = milestones[i]
      params = learner.get_state().params
      mae, nll = predictor_metrics(nets, params, dc, val_for_metrics, nbins)
      env = MultiModalPoint2D(jitter=args.jitter)
      stats = evaluate_policy(env, params, args.eval_episodes, rollout,
                              normalize_obs, unnormalize_action, max_d)
      sr = float(np.mean(stats['success']))
      out = os.path.join(args.out_dir, f'predictor_f{int(frac*100):03d}.pkl')
      ckpt = {
          'params': params,
          'norm_stats': {k: np.asarray(v) for k, v in norm_stats.items()},
          'dc_config': {'min_distance': min_d, 'max_distance': max_d,
                        'num_bins': nbins},
          'meta': {'fraction': frac, 'steps': i * nmb, 'seed': args.seed,
                   'predictor_mae': mae, 'predictor_nll': nll,
                   'policy_success_rate': sr,
                   'created_at': datetime.datetime.now().astimezone().isoformat()},
      }
      with open(out, 'wb') as fp:
        pickle.dump(ckpt, fp)
      print(f'  [f={frac:.2f}] step {i*nmb}: MAE={mae:.2f} NLL={nll:.3f} '
            f'policy_SR={sr:.2f} -> {out}')
      saved.append(out)

  print('Saved predictors:', saved)


if __name__ == '__main__':
  main()
