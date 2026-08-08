r"""DDPO 적용 전(control) vs 후(arm A) 디퓨전 정책 — 같은 시드로 나란히 녹화.

`record_carry_actor.py`와 같은 패턴(같은 `draw_env` 재사용, 같은 시드로 두 env를 동시에
굴려 나란히 그림)이지만, 여기 두 policy는 `ScriptedCarryPolicy`(사람이 짠 상태기계)가
아니라 `BCStgGuided`(디퓨전 정책이 끝단까지 액션을 직접 생성, STG 평가 없이 그대로 실행)다
— `run_bc_stg_guided.py`/`eval_carry_si.py`가 실제로 채점한 그 정책 그대로를 시각화한다.

**주의(중요)**: `BCStgGuided`의 확산 샘플링 노이즈 키(`self._key`)는 매 호출마다
`jax.random.split`으로 진행되고 `reset()`으로 되돌아가지 않는다 — 그래서 "env 시드
900004의 결과"는 그 하나만 따로 부르면 다르게 나오고, **정책을 이 시드까지 오는 동안
몇 번 불렀는지**(=몇 개의 이전 에피소드를 거쳤는지)에 의존한다. `eval_carry_si.py`가
보고한 성공률은 매번 fresh 정책으로 `seed0`부터 순서대로 200개를 부른 결과이므로, 그와
같은 이야기를 하려면 이 스크립트도 `--seed0`부터 순서대로 전부 호출해야 한다(그려서
저장하는 건 `--seeds`로 고른 것만). 이 재현성 함정은 2026-08-07에 실측으로 발견했다 —
`--seeds`만 따로 호출한 첫 시도는 스캔 때와 다른 결과가 나왔다.

    python record_carry_si.py --seeds 900004 900006 900022 900039 \
        --out results/videos/grasp_carry_si_compare.mp4
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

from record_carry import draw_env, _ee_block_dist
from run_bc_stg_guided import BCStgGuided
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_pair(seeds, seed0, control_ckpt, si_ckpt, cfg, fps, out):
  """`seed0`부터 `max(seeds)`까지 **전부** 순서대로 굴린다(RNG 진행용) — `seeds`에
  없는 시드는 그리지 않고 결과만 버린다. 이렇게 해야 `eval_carry_si.py`가 실제로
  채점할 때와 같은 확산 노이즈 궤적이 재현된다(모듈 docstring의 재현성 주의 참고)."""
  env_a, env_b = GraspCarry2D(cfg), GraspCarry2D(cfg)
  pol_control = BCStgGuided(control_ckpt, None, seed=1)
  pol_si = BCStgGuided(si_ckpt, None, seed=1)
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=110)
  frames = []
  labels = ('DDPO 적용 전(control)', 'DDPO 적용 후(arm A)')
  colors = ('tab:red', 'tab:blue')

  want = set(seeds)
  all_seeds = range(seed0, max(seeds) + 1)
  for seed in all_seeds:
    render_this = seed in want
    envs = [env_a, env_b]
    policies = [pol_control, pol_si]
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
      if render_this:
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

    outcomes = [e._info()['outcome'] for e in envs]
    steps = [e._info()['steps'] for e in envs]
    tag = '  [녹화]' if render_this else '  (RNG 진행용, 미녹화)'
    print(f'seed {seed}: ' + '  '.join(
        f'{lbl}->{st}스텝/{oc}' for lbl, st, oc in zip(labels, steps, outcomes)) + tag,
        flush=True)
    if render_this and frames:
      frames += [frames[-1]] * fps

  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'saved {out} ({len(frames)} frames)')


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--seeds', type=int, nargs='+', default=[900004, 900006, 900022, 900039],
                  help='기본값(900000부터 순서대로 굴린 실제 궤적에서 고른 것): '
                       'control 타임아웃->armA 성공(900004), control 전도->armA 성공(900006), '
                       '둘 다 성공이지만 armA가 2배 빠름(900022), 정직하게 armA가 더 '
                       '나쁜 사례 — control 성공(78스텝)->armA 전도(900039).')
  ap.add_argument('--seed0', type=int, default=900000,
                  help='eval_carry_si.py와 같은 held-out 시작점. seeds에 없는 시드도 '
                       '여기서부터 순서대로 전부 호출해 확산 노이즈 궤적을 재현한다.')
  ap.add_argument('--control-ckpt', default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--si-ckpt', default='checkpoints/grasp_carry_si_A_mean_succ/predictor.pkl')
  ap.add_argument('--fps', type=int, default=10)
  ap.add_argument('--out', default='results/videos/grasp_carry_si_compare.mp4')
  args = ap.parse_args(argv)
  cfg = CarryConfig()
  run_pair(args.seeds, args.seed0, args.control_ckpt, args.si_ckpt, cfg, args.fps, args.out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
