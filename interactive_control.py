"""Interactive / replayable pointmass control with live steps-to-go distribution.

Two modes:

  * GUI (default): drive the point mass with the keyboard while a side panel shows
    the TIMER steps-to-go distribution updating in real time. Requires a display.
    Trigger interventions (teleport / obstacle / bias / random) with hotkeys.

  * Replay (--replay JSON): headless (Agg). Runs a scripted event sequence,
    renders each 2-panel frame, and writes a video. Used for automated
    verification since the harness shell has no DISPLAY.

The distribution panel reuses stg_probe.STGProbe so the numbers match step 3 and
the Stage-2 reward signal. All interventions go through EnhancedPoint2D's public
API (so they land in intervention_log); env private fields are never written here.
"""

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np
import jax

from envs_enhanced import EnhancedPoint2D
from stg_probe import STGProbe


# ------------------------------------------------------------------ frame draw
def _compose_frame(env, records, bin_vals, backend_plt):
  """Render a single 2-panel RGB frame (env view | STG distribution + curves)."""
  plt = backend_plt
  fig = plt.figure(figsize=(11, 5))
  gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[2, 1])

  # left: environment
  ax_env = fig.add_subplot(gs[:, 0])
  ax_env.imshow(env.render())
  ax_env.axis('off')
  ax_env.set_title('environment')

  # right-top: current distribution
  ax_dist = fig.add_subplot(gs[0, 1])
  if records:
    r = records[-1]
    ax_dist.bar(bin_vals, r.probs, width=(bin_vals[1] - bin_vals[0]) * 0.9,
                color='C0')
    ax_dist.set_title(f'STG dist  E={r.expectation:.1f}  var={r.variance:.1f}')
  ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')

  # right-bottom: expectation & variance timeseries
  ax_ts = fig.add_subplot(gs[1, 1])
  if records:
    steps = [r.step_idx for r in records]
    ax_ts.plot(steps, [r.expectation for r in records], color='C0',
               label='E[STG]')
    ax_tsb = ax_ts.twinx()
    ax_tsb.plot(steps, [r.variance for r in records], color='C3',
                linestyle='--', label='var')
    ax_ts.set_ylabel('E', color='C0'); ax_tsb.set_ylabel('var', color='C3')
  ax_ts.set_xlabel('step')

  fig.tight_layout()
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
  img = buf.reshape(h, w, 4)[..., :3].copy()
  plt.close(fig)
  return img


def _record(probe, env, ts, key):
  key, sub = jax.random.split(key)
  _, logits = probe._logits_and_act(ts.observation, sub)
  step_idx = env._step_count
  return probe._record(step_idx, ts.observation, logits), key


# ------------------------------------------------------------------- replay
def run_replay(checkpoint: str, replay_path: str, out_path: str,
               max_steps: int = 200):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  with open(replay_path) as fp:
    script = json.load(fp)
  seed = script.get('seed', 0)
  events = script.get('events', [])
  by_step = {}
  for ev in events:
    by_step.setdefault(ev['step'], []).append(ev)

  probe = STGProbe(checkpoint)
  np.random.seed(seed)
  env = EnhancedPoint2D()
  ts = env.reset()
  key = jax.random.PRNGKey(42)

  records = []
  frames = []
  rec, key = _record(probe, env, ts, key)
  records.append(rec)
  frames.append(_compose_frame(env, records, probe.bin_vals, plt))

  last_step = max((ev['step'] for ev in events), default=0)
  n_steps = max(last_step + 5, 1)
  for step in range(min(n_steps, max_steps)):
    action = np.zeros(2, dtype=np.float32)
    use_policy = False
    for ev in by_step.get(step, []):
      kind = ev['kind']
      if kind == 'action':
        action = np.asarray(ev['value'], dtype=np.float32)
      elif kind == 'policy':
        use_policy = True
      elif kind == 'teleport':
        env.teleport(np.asarray(ev['value'], dtype=np.float32),
                     zero_velocity=ev.get('zero_velocity', False))
      elif kind == 'obstacle':
        env.add_obstacle(np.asarray(ev['center'], dtype=np.float32),
                         float(ev['radius']))
      elif kind == 'bias':
        v = ev['value']
        env.set_bias_force(None if v is None
                           else np.asarray(v, dtype=np.float32))
      elif kind == 'random':
        env.set_random_action(float(ev['prob']), float(ev['scale']),
                              ev.get('seed'))

    if use_policy:
      key, sub = jax.random.split(key)
      act, _ = probe._logits_and_act(ts.observation, sub)
      action = probe.unnormalize_action(act)[0]

    ts = env.step(action)
    rec, key = _record(probe, env, ts, key)
    records.append(rec)
    frames.append(_compose_frame(env, records, probe.bin_vals, plt))
    if env.success():
      break

  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  _write_video(frames, out_path)
  print(f'Replay: {len(frames)} frames, {len(records)} records -> {out_path}')
  print(f'  interventions: {env.intervention_log}')
  return out_path


def _write_video(frames: List[np.ndarray], out_path: str):
  """mp4 via imageio-ffmpeg; falls back to gif (PIL) if encoding fails."""
  try:
    import imageio.v2 as imageio
    if out_path.lower().endswith('.mp4'):
      imageio.mimsave(out_path, frames, fps=10, macro_block_size=None)
      return
  except Exception as e:  # pragma: no cover - depends on ffmpeg availability
    print(f'  mp4 encode failed ({e}); falling back to gif')
  from PIL import Image
  gif_path = os.path.splitext(out_path)[0] + '.gif'
  imgs = [Image.fromarray(f) for f in frames]
  imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100,
               loop=0)
  print(f'  wrote {gif_path}')


# --------------------------------------------------------------------- GUI
def run_gui(checkpoint: str, action_scale: float, bias_force, obstacle_radius,
            out_dir: str, max_steps: int = 400):  # pragma: no cover - needs display
  import matplotlib.pyplot as plt

  probe = STGProbe(checkpoint)
  np.random.seed(int(time.time()) % 10000)
  env = EnhancedPoint2D()
  ts = [env.reset()]
  key = [jax.random.PRNGKey(0)]
  records = []
  state = {'bias_on': False, 'random_on': False, 'mode': None}

  fig = plt.figure(figsize=(11, 5))
  gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[2, 1])
  ax_env = fig.add_subplot(gs[:, 0])
  ax_dist = fig.add_subplot(gs[0, 1])
  ax_ts = fig.add_subplot(gs[1, 1])

  def redraw():
    ax_env.clear(); ax_dist.clear(); ax_ts.clear()
    ax_env.imshow(env.render()); ax_env.axis('off')
    if records:
      r = records[-1]
      ax_dist.bar(probe.bin_vals, r.probs,
                  width=(probe.bin_vals[1] - probe.bin_vals[0]) * 0.9)
      ax_dist.set_title(f'E={r.expectation:.1f} var={r.variance:.1f}')
      steps = [x.step_idx for x in records]
      ax_ts.plot(steps, [x.expectation for x in records], color='C0')
    fig.canvas.draw_idle()

  def do_step(action):
    key[0], sub = jax.random.split(key[0])
    rec, key[0] = _record(probe, env, ts[0], key[0])
    records.append(rec)
    ts[0] = env.step(np.asarray(action, dtype=np.float32))
    if env.success():
      ax_env.set_title('SUCCESS')
    redraw()

  def on_key(event):
    k = event.key
    if k in ('up', 'down', 'left', 'right'):
      vec = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0),
             'right': (1, 0)}[k]
      do_step(np.array(vec) * action_scale)
    elif k == ' ':
      do_step(np.zeros(2))
    elif k == 'p':
      key[0], sub = jax.random.split(key[0])
      act, _ = probe._logits_and_act(ts[0].observation, sub)
      do_step(probe.unnormalize_action(act)[0])
    elif k == 'b':
      state['bias_on'] = not state['bias_on']
      env.set_bias_force(np.asarray(bias_force, dtype=np.float32)
                         if state['bias_on'] else None)
    elif k == 'x':
      state['random_on'] = not state['random_on']
      env.set_random_action(0.3 if state['random_on'] else 0.0, action_scale)
    elif k == 'g':
      env.set_goal(env.sample_goal()); redraw()
    elif k == 'n':
      ts[0] = env.reset(); records.clear(); redraw()
    elif k == 't':
      state['mode'] = 'teleport'
    elif k == 'o':
      state['mode'] = 'obstacle'
    elif k == 's':
      _save_session(records, env, out_dir)
    elif k == 'q':
      _save_session(records, env, out_dir); plt.close(fig)

  def on_click(event):
    if event.inaxes is not ax_env or event.xdata is None:
      return
    # map click (image pixels) to world coords [-1,1]
    img_h, img_w = env.render().shape[:2]
    wx = event.xdata / img_w * 2 - 1
    wy = 1 - event.ydata / img_h * 2
    if state['mode'] == 'teleport':
      env.teleport(np.array([wx, wy], dtype=np.float32), zero_velocity=True)
      do_step(np.zeros(2)); state['mode'] = None
    elif state['mode'] == 'obstacle':
      env.add_obstacle(np.array([wx, wy], dtype=np.float32), obstacle_radius)
      redraw(); state['mode'] = None

  fig.canvas.mpl_connect('key_press_event', on_key)
  fig.canvas.mpl_connect('button_press_event', on_click)
  redraw()
  print('GUI controls: arrows=move  space=noop  p=policy  b=bias  x=random  '
        'g=new goal  n=reset  t+click=teleport  o+click=obstacle  s=save  q=quit')
  plt.show()


def _save_session(records, env, out_dir):  # pragma: no cover - GUI helper
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'session_{int(time.time())}.json')
  # Save in the replay schema so a GUI session can be re-played headlessly.
  events = []
  for step_idx, kind, detail in env.intervention_log:
    if kind == 'teleport':
      events.append({'step': step_idx, 'kind': 'teleport',
                     'value': list(detail[0]), 'zero_velocity': detail[1]})
    elif kind == 'add_obstacle':
      events.append({'step': step_idx, 'kind': 'obstacle',
                     'center': list(detail[1]), 'radius': detail[2]})
  with open(path, 'w') as fp:
    json.dump({'seed': 0, 'events': events}, fp, indent=2)
  print(f'  saved session -> {path}')


def main():
  parser = argparse.ArgumentParser(description='Interactive/replay pointmass '
                                   'control with live STG distribution.')
  parser.add_argument('--checkpoint', default='checkpoints/sft_state.pkl')
  parser.add_argument('--replay', default=None,
                      help='Replay JSON script (headless video output).')
  parser.add_argument('--out', default='outputs/interactive/replay.mp4')
  parser.add_argument('--action_scale', type=float, default=0.002)
  parser.add_argument('--bias_force', type=float, nargs=2,
                      default=(0.002, 0.0))
  parser.add_argument('--obstacle_radius', type=float, default=0.15)
  args = parser.parse_args()

  if args.replay:
    run_replay(args.checkpoint, args.replay, args.out)
  else:
    run_gui(args.checkpoint, args.action_scale, args.bias_force,
            args.obstacle_radius,
            out_dir=os.path.dirname(args.out) or 'outputs/interactive')


if __name__ == '__main__':
  main()
