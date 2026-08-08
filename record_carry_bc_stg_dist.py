r"""학습된(스크립트 아님) diffusion BC 정책의 **실패** 롤아웃에 STG 카테고리컬
분포를 나란히 붙여서 녹화한다. `record_carry_stg_dist.py`와 같은 화면 구성이지만
액션이 `ScriptedCarryPolicy`가 아니라 실제 학습된 정책(`checkpoints/grasp_carry_diff100_v5`)
에서 나온다 — `data/grasp_carry_bc_v5_rollouts.pkl` 수집 때와 동일 경로를 재현한다.

빨간 세로선은 데드라인 B(=reset_cost+T̂, `grasp_carry_dstg_deadline_v5rollout`이
실제 실패 롤아웃으로 학습한 값)다.

    python record_carry_bc_stg_dist.py --seed 900003 \
        --out results/videos/grasp_carry_bc_fail_seed900003.mp4
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

from record_carry import draw_env, _ee_block_dist
from rollout_carry_diff_stats import load_diff_policy
from src.train_carry_predictor import concat_obs
from src.carry_stg_reward import StgReward
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(seed, diff_ckpt, dstg_ckpt, fps, xmax, out):
  ck, nets, normalize_obs, normalize_action, unnormalize_action = load_diff_policy(diff_ckpt)
  _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
  reward = StgReward(dstg_ckpt, statistic='mean')
  deadline_B = reward.meta.get('deadline_B')
  bin_vals = np.arange(reward.num_bins)

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  obs, info = env.reset(seed=seed)
  base_offset = None
  action = None

  fig, (ax_scene, ax_dist) = plt.subplots(
      1, 2, figsize=(11.5, 4.6), dpi=110, gridspec_kw={'width_ratios': [1, 1.3]})
  frames = []

  kk = jax.random.PRNGKey(1234)
  for t in range(cfg.max_steps):
    o = {'frame': np.asarray(obs, np.float32)[None]}
    c = np.asarray(concat_obs(normalize_obs(o)))
    kk, s2 = jax.random.split(kk)
    a_n = np.asarray(_sample(ck['params'], jnp.asarray(c), s2))[0]
    action = np.asarray(unnormalize_action(a_n), np.float32)
    obs, _, term, trunc, info = env.step(action)

    if env.is_held() and base_offset is None:
      base_offset = _ee_block_dist(env)
    elif not env.is_held():
      base_offset = None

    stacked = env._stacked_obs()[None]
    probs = reward._probs(stacked)[0]
    mu, sigma = reward.mean_std(stacked)
    mu, sigma = float(mu[0]), float(sigma[0])

    draw_env(ax_scene, env, action=action, side_label='', base_offset=base_offset)

    ax_dist.clear()
    xshow = min(xmax, reward.num_bins)
    ax_dist.bar(bin_vals[:xshow], probs[:xshow], width=1.0, color='tab:purple', alpha=0.75)
    if deadline_B is not None and deadline_B < xshow:
      ax_dist.axvline(deadline_B, color='tab:red', ls='--', lw=1.5,
                      label=f'B={deadline_B:.0f}(리셋 데드라인)')
    ax_dist.axvline(mu, color='black', ls='-', lw=1.2, label=f'μ={mu:.1f}')
    ax_dist.set_xlim(0, xshow)
    ax_dist.set_ylim(0, max(0.05, float(probs[:xshow].max()) * 1.25))
    ax_dist.set_xlabel('예측 잔여 스텝 bin')
    ax_dist.set_ylabel('확률')
    ax_dist.set_title(f'STG 분포  μ={mu:.1f}  σ={sigma:.1f}')
    ax_dist.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'seed={seed}  t={t}  outcome={info["outcome"]}  (학습된 BC 정책)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    frames.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                  .reshape(int(h), int(w), 4)[..., :3].copy())

    if term or trunc:
      break

  frames += [frames[-1]] * (fps * 2)
  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'seed={seed}: {info["steps"]}스텝, outcome={info["outcome"]}')
  print(f'saved {out} ({len(frames)} frames)')


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100_v5/predictor.pkl')
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl')
  ap.add_argument('--fps', type=int, default=8)
  ap.add_argument('--xmax', type=int, default=160, help='분포 막대그래프 x축 상한(bin)')
  ap.add_argument('--out', default='results/videos/grasp_carry_bc_fail.mp4')
  args = ap.parse_args(argv)
  run(args.seed, args.diff_ckpt, args.dstg_ckpt, args.fps, args.xmax, args.out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
