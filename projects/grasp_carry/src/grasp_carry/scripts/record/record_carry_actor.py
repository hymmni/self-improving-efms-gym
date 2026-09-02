r"""AI-E 보상으로 학습한 정책 vs AI-R 보상으로 학습한 정책 — 같은 시드로 나란히 녹화.

`train_carry_actor.py`가 학습한 두 actor를 `ScriptedCarryPolicy`의
`speed_selector`로 꽂아 굴린다(`eval_carry_actor.py`와 같은 조합). 렌더링은
`record_carry.py`의 `draw_env`를 그대로 재사용한다(그리퍼·블록·박스 기하를
렌더러에서 다시 계산하지 않는다는 원칙 유지).

    python -m grasp_carry.scripts.record.record_carry_actor --seeds 3 7 11 \
        --out results/videos/grasp_carry_actor_compare.mp4
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

from grasp_carry.scripts.record.record_carry import draw_env, _ee_block_dist
from grasp_carry.scripts.analyze.eval_carry_actor import ActorSelector, NOMINAL_SPEED
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_pair(seeds, exp_actor, risk_actor, cfg, fps, out):
  env_a, env_b = GraspCarry2D(cfg), GraspCarry2D(cfg)
  sel_e, sel_r = ActorSelector(exp_actor), ActorSelector(risk_actor)
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=110)
  frames = []
  labels = ('AI-E 보상 학습', 'AI-R 보상 학습')
  colors = ('tab:red', 'tab:blue')

  for seed in seeds:
    envs = [env_a, env_b]
    policies = [ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED, speed_selector=sel_e),
               ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED, speed_selector=sel_r)]
    for e, p in zip(envs, policies):
      e.reset(seed=seed); p.reset()
    done = [False, False]
    actions = [None, None]
    base_offset = [None, None]

    for _ in range(cfg.max_steps):
      for i, (e, p) in enumerate(zip(envs, policies)):
        if done[i]:
          continue
        actions[i] = p(e)
        _, _, term, trunc, _ = e.step(actions[i])
        if term or trunc:
          done[i] = True
      for i, e in enumerate(envs):
        if e.is_held() and base_offset[i] is None:
          base_offset[i] = _ee_block_dist(e)
        elif not e.is_held():
          base_offset[i] = None

      for i, (e, lbl) in enumerate(zip(envs, labels)):
        draw_env(axes[i], e, action=actions[i], side_label=f'{lbl}   ',
                 block_color=colors[i], base_offset=base_offset[i])
      info_a = env_a._info()
      fig.suptitle(
          f"[은닉] 질량={info_a['mass']:.2f}kg 마찰={info_a['friction']:.2f}"
          f"   |   [관측 가능] 블록 {env_a.block_w:.0f}x{env_a.block_h:.0f}mm"
          f"  소스 내폭 {env_a.src_box.inner_width:.0f}mm   (seed {seed})",
          fontsize=10)
      fig.tight_layout(rect=[0, 0, 1, 0.90])
      canvas = FigureCanvasAgg(fig)
      canvas.draw()
      w, h = fig.get_size_inches() * fig.get_dpi()
      frames.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                    .reshape(int(h), int(w), 4)[..., :3].copy())
      if all(done):
        break

    frames += [frames[-1]] * fps
    outcomes = [e._info()['outcome'] for e in envs]
    steps = [e._info()['steps'] for e in envs]
    print(f'seed {seed}: ' + '  '.join(
        f'{lbl}->{st}스텝/{oc}' for lbl, st, oc in zip(labels, steps, outcomes)),
        flush=True)

  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'saved {out} ({len(frames)} frames)')


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 11])
  ap.add_argument('--exp-actor', default='checkpoints/grasp_carry_actor_exponly/actor.pkl')
  ap.add_argument('--risk-actor', default='checkpoints/grasp_carry_actor_risk/actor.pkl')
  ap.add_argument('--fps', type=int, default=10)
  ap.add_argument('--out', default='results/videos/grasp_carry_actor_compare.mp4')
  args = ap.parse_args(argv)
  cfg = CarryConfig()
  run_pair(args.seeds, args.exp_actor, args.risk_actor, cfg, args.fps, args.out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
