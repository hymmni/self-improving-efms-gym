"""Stage 1 (SFT) training script for the SI-EFM pointmass demo.

Reconstructs the notebook's Stage 1 pipeline (cells 28~34) on top of
`pointmass_core`, then saves a checkpoint so later tools (distribution
probes, interactive control, scenario batteries) can reuse the trained
steps-to-go head without re-running the notebook.

All hyperparameter defaults match the notebook:
  dataset:   num_episodes=10000, num_waypoints_per_episode=5,
             episode_len_discard_thresh=10 (cell 10)
  loader:    global_minibatch_size=256, num_minibatches=128, train ratio 0.9 (cell 28)
  converter: min_distance=0, max_distance=140, num_bins=50 (cell 29)
  network:   layer_sizes=(256, 256, 256) (cell 30)
  optimizer: Adam(eps=1e-7), learning_rate=3e-4 (cell 31)
  learner:   PRNGKey(42) (cell 32); --seed controls only env/dataset sampling
  training:  num_steps=32768 SGD steps (cell 33)

Checkpoint format (pickle dict):
  params:     TIMER network params (numpy pytree from the first device)
  norm_stats: output of pointmass_core.compute_normalization_stats — feeding
              it to pointmass_core.make_normalizers fully reconstructs
              normalize_obs / normalize_action / unnormalize_action
  dc_config:  {min_distance, max_distance, num_bins} for
              pointmass_core.build_discrete_distance_converter
  meta:       {num_steps, seed, final_success_rate, created_at}

Usage:
  python train_sft.py [--num_steps 32768] [--dataset_steps N] [--seed 0]
                      [--eval_episodes 50] [--out checkpoints/sft_state.pkl]

By default the demonstration dataset is 10000 episodes (~780k transitions),
exactly as in notebook cell 10. --dataset_steps switches to a transition
budget instead, for quick runs only — 100k steps was measured to drop the
SFT success rate to 0.76 vs 0.92 at notebook scale.
"""

import argparse
import datetime
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from pointmass_core import (
    Point2D,
    pd_controller,
    generate_dataset,
    build_continuous_act_discrete_dist_v0,
    build_discrete_distance_converter,
    PretrainLearner,
    compute_normalization_stats,
    make_normalizers,
    make_timer_rollout_policy,
)

# The dataset pipeline is tf.data-only; keep TF off the GPU so it does not
# grab memory that JAX needs.
tf.config.set_visible_devices([], 'GPU')


def collect_demonstrations(env, num_episodes=None, num_steps=None,
                           chunk_episodes=500):
  """Collects PD-controller demonstration episodes.

  Either exactly `num_episodes` episodes (notebook cell 10 behavior) or, if
  `num_steps` is given instead, episodes until at least that many
  transitions. Calls pointmass_core.generate_dataset (cell 10 logic, with
  its default num_waypoints_per_episode=5 / episode_len_discard_thresh=10)
  in chunks; only the concatenation across chunks happens here.
  """
  episodes = []
  total_steps = 0
  def _done():
    if num_steps is not None:
      return total_steps >= num_steps
    return len(episodes) >= num_episodes
  while not _done():
    chunk = chunk_episodes
    if num_steps is None:
      chunk = min(chunk, num_episodes - len(episodes))
    chunk_eps, chunk_tuples = generate_dataset(
        env, pd_controller, num_episodes=chunk)
    episodes.extend(chunk_eps)
    total_steps += chunk_tuples.observation['cur_pos'].shape[0]
    print(f'  dataset: {len(episodes)} episodes, {total_steps} steps')
  all_tuples = jax.tree.map(
      lambda *xs: np.concatenate(xs, dtype=np.float32), *episodes)
  return episodes, all_tuples


# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 28)
# MODIFIED: notebook globals (global_minibatch_size, num_minibatches, ...)
# are taken as arguments; the shuffle buffer uses the passed tuples' own size
# (the notebook uses the global all_tuples size for both splits).
def make_dataset_from_tuples(
    data_tuples, global_minibatch_size, num_minibatches):
  batch_size = global_minibatch_size * num_minibatches
  num_learners = jax.device_count()

  def reshape_for_pmap(x):
    new_shape = [
        num_learners,
        (global_minibatch_size * num_minibatches) // num_learners,
    ] + list(x.shape[1:])
    return tf.reshape(x, new_shape)

  dataset = tf.data.Dataset.from_tensor_slices(data_tuples).cache()
  dataset = dataset.shuffle(
      data_tuples.observation['cur_pos'].shape[0],
      reshuffle_each_iteration=True)
  dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
  dataset = dataset.map(lambda x: jax.tree.map(reshape_for_pmap, x))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.as_numpy_iterator()
  return dataset


# From: pointmass_core.evaluate_timer_rollout_policy (notebook cell 39)
# MODIFIED: returns the stats arrays instead of only printing, so the success
# rate can be stored in the checkpoint meta. Rollout logic is unchanged.
def evaluate_policy(
    env, params, num_episodes,
    timer_rollout_policy, normalize_obs, unnormalize_action, max_distance):
  stats = []
  for eps_num in range(num_episodes):
    timesteps = []
    ts = env.reset()
    ts = ts._replace(reward=0.)
    timesteps.append(ts)

    key = jax.random.PRNGKey(42)

    while (not env.success()) and len(timesteps) < max_distance:
      key, sub_key = jax.random.split(key)
      cur_obs = ts.observation
      cur_obs = jax.tree.map(lambda x: x[None], cur_obs)
      cur_obs = normalize_obs(cur_obs)  # 1 x dims
      norm_act, _ = timer_rollout_policy(params, cur_obs, sub_key)
      unnorm_act = unnormalize_action(norm_act)
      ts = env.step(unnorm_act[0])
      timesteps.append(ts)

    episode_stats = {}
    episode_stats['success'] = env.success()
    episode_stats['return'] = sum(x.reward for x in timesteps)
    episode_stats['len'] = len(timesteps)
    stats.append(episode_stats)

  stats = jax.tree.map(lambda *xs: np.stack(xs), *stats)
  return stats


def main():
  parser = argparse.ArgumentParser(
      description='Stage 1 SFT training for the SI-EFM pointmass demo.')
  parser.add_argument('--num_steps', type=int, default=32768,
                      help='Total SGD steps (notebook cell 33 default).')
  parser.add_argument('--dataset_steps', type=int, default=None,
                      help='If set, collect at least this many demonstration '
                           'transitions instead of the notebook default of '
                           '10000 episodes (quick runs only; hurts success '
                           'rate below ~780k).')
  parser.add_argument('--dataset_episodes', type=int, default=10000,
                      help='Demonstration episodes (notebook cell 10 '
                           'default). Ignored when --dataset_steps is set.')
  parser.add_argument('--seed', type=int, default=0,
                      help='Seed for env/dataset sampling (numpy, tf).')
  parser.add_argument('--eval_episodes', type=int, default=50,
                      help='Rollout episodes for the post-training eval.')
  parser.add_argument('--out', type=str, default='checkpoints/sft_state.pkl',
                      help='Checkpoint output path.')
  args = parser.parse_args()

  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)

  # Notebook hyperparameters (cells 28~31)
  global_minibatch_size = 256
  num_minibatches = 128
  min_distance = 0
  max_distance = 140
  num_distance_bins = 50
  layer_sizes = (256, 256, 256)
  learning_rate = 3e-4
  train_set_ratio = 0.9

  assert args.num_steps % num_minibatches == 0, (
      f'--num_steps must be a multiple of num_minibatches={num_minibatches}')

  print(f'JAX devices: {jax.devices()}')

  # 1) Demonstration dataset (notebook cell 10)
  env = Point2D()
  if args.dataset_steps is not None:
    print(f'Generating dataset (>= {args.dataset_steps} steps)...')
    episodes, all_tuples = collect_demonstrations(
        env, num_steps=args.dataset_steps)
  else:
    print(f'Generating dataset ({args.dataset_episodes} episodes)...')
    episodes, all_tuples = collect_demonstrations(
        env, num_episodes=args.dataset_episodes)

  # 2) Normalization stats (notebook cell 25)
  norm_stats = compute_normalization_stats(all_tuples)
  normalize_obs, normalize_action, unnormalize_action = make_normalizers(
      norm_stats)
  normalized_all_tuples = all_tuples._replace(
      observation=normalize_obs(all_tuples.observation),
      action=normalize_action(all_tuples.action))

  # 3) Data loaders (notebook cell 28)
  normalized_all_tuples_size = (
      normalized_all_tuples.observation['cur_pos'].shape[0])
  train_set_size = int(normalized_all_tuples_size * train_set_ratio)
  train_dataset = make_dataset_from_tuples(
      jax.tree.map(lambda x: np.array(x[:train_set_size]),
                   normalized_all_tuples),
      global_minibatch_size, num_minibatches)
  val_dataset = make_dataset_from_tuples(
      jax.tree.map(lambda x: np.array(x[train_set_size:]),
                   normalized_all_tuples),
      global_minibatch_size, num_minibatches)

  # 4) Distance converter, networks, optimizer, learner (cells 29~32)
  distance_converter = build_discrete_distance_converter(
      min_distance, max_distance, num_distance_bins)

  timer_networks = build_continuous_act_discrete_dist_v0(
      layer_sizes,
      env.action_spec().shape[0],
      num_distance_bins,
      np.ones((4, 6), dtype=np.float32))

  optimizer = optax.chain(
      optax.scale_by_adam(eps=1e-7),
      optax.scale(-1. * learning_rate))

  learner = PretrainLearner(
      timer_networks,
      distance_converter,
      optimizer,
      jax.random.PRNGKey(42),  # notebook cell 32 value; not tied to --seed
      global_minibatch_size,
      num_minibatches)

  # 5) Stage 1 SFT train loop (notebook cell 33, without live plotting)
  num_iters = args.num_steps // num_minibatches
  log_every = max(1, num_iters // 16)
  print(f'Training: {args.num_steps} SGD steps '
        f'({num_iters} learner iterations x {num_minibatches} minibatches)')
  for i in range(num_iters):
    batch = next(train_dataset)
    results = learner.step(batch)
    if i % log_every == 0 or i == num_iters - 1:
      val_batch = next(val_dataset)
      val_results = learner.compute_loss(val_batch)
      print(f'  step {(i + 1) * num_minibatches:6d}/{args.num_steps}: '
            f"train_loss={results['pretrain_loss'].item():.4f} "
            f"val_loss={val_results['pretrain_loss'].item():.4f}")

  # 6) Evaluation rollouts (notebook cell 39 logic)
  print(f'Evaluating over {args.eval_episodes} episodes...')
  cpu_state = learner.get_state()
  cpu_params = cpu_state.params
  timer_rollout_policy = make_timer_rollout_policy(
      timer_networks, distance_converter)
  stats = evaluate_policy(
      env, cpu_params, args.eval_episodes,
      timer_rollout_policy, normalize_obs, unnormalize_action, max_distance)
  success_rate = float(np.mean(stats['success']))
  print(f'success rate: {success_rate}')
  print(f"Returns: {np.mean(stats['return']):.2f} "
        f"+/- {np.std(stats['return']):.2f}")
  print(f"Episode Lengths: {np.mean(stats['len']):.2f} "
        f"+/- {np.std(stats['len']):.2f}")

  # 7) Checkpoint
  checkpoint = {
      'params': cpu_params,
      'norm_stats': {k: np.asarray(v) for k, v in norm_stats.items()},
      'dc_config': {
          'min_distance': min_distance,
          'max_distance': max_distance,
          'num_bins': num_distance_bins,
      },
      'meta': {
          'num_steps': args.num_steps,
          'seed': args.seed,
          'final_success_rate': success_rate,
          'created_at': datetime.datetime.now().astimezone().isoformat(),
      },
  }
  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump(checkpoint, fp)
  print(f'Saved checkpoint to {args.out}')


if __name__ == '__main__':
  main()
