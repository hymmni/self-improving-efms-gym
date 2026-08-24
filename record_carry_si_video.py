r"""DDPO-SF로 자기개선된(또는 순수 BC) diffusion 정책을 실제로 굴려서 STG 분포와
함께 녹화한다. `record_carry_stg_dist.py`(스크립트 정책 전용)와 달리 학습된 정책이
직접 행동을 만든다 — `rollout_carry_diff_stats.py::load_diff_policy`로 로드한다
(DDPO-SF 체크포인트도 원본 BC 체크포인트와 같은 payload 구조라 그대로 재사용 가능,
train_carry_si.py::_save_ckpt가 'params'만 바꿔치기하고 나머지는 복사하기 때문).

    python record_carry_si_video.py --seed 3 \
        --diff-ckpt checkpoints/grasp_carry_si_v5n50_successonly/predictor_best.pkl \
        --out results/videos/si_successonly_seed3.mp4
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

  # 2026-08-11: --horizon>1(액션 청킹) 체크포인트 지원. 청킹 안 쓰는 체크포인트는
  # meta에 horizon이 없으므로 기본값 1(기존 동작과 동일)로 되돌아간다.
  m = ck['meta']
  horizon = int(m.get('horizon', 1))
  exec_horizon = int(m.get('exec_horizon', horizon))
  act_dim = int(m.get('act_dim', len(ck['norm_stats']['act_mean'])))

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  obs, info = env.reset(seed=seed)
  base_offset = None
  action = None

  fig, (ax_scene, ax_dist) = plt.subplots(
      1, 2, figsize=(11.5, 4.6), dpi=110, gridspec_kw={'width_ratios': [1, 1.3]})
  frames = []

  kk = jax.random.PRNGKey(1234)
  t = 0
  stop = False
  while t < cfg.max_steps and not stop:
    o = {'frame': np.asarray(obs, np.float32)[None]}
    c = np.asarray(concat_obs(normalize_obs(o)))
    kk, s2 = jax.random.split(kk)
    chunk_n = np.asarray(_sample(ck['params'], jnp.asarray(c), s2))[0].reshape(horizon, act_dim)
    chunk = np.asarray(unnormalize_action(chunk_n), np.float32)

    for h in range(min(exec_horizon, horizon)):
      if t >= cfg.max_steps:
        stop = True
        break
      action = chunk[h]
      obs, _, term, trunc, info = env.step(action)
      t += 1

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

      fig.suptitle(f'seed={seed}  t={t}  outcome={info["outcome"]}  (학습된 정책)', fontsize=11)
      fig.tight_layout(rect=[0, 0, 1, 0.93])

      canvas = FigureCanvasAgg(fig)
      canvas.draw()
      w, h = fig.get_size_inches() * fig.get_dpi()
      frames.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                    .reshape(int(h), int(w), 4)[..., :3].copy())

      if term or trunc:
        stop = True
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
  ap.add_argument('--diff-ckpt', required=True)
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl')
  ap.add_argument('--fps', type=int, default=8)
  ap.add_argument('--xmax', type=int, default=160)
  ap.add_argument('--out', default='results/videos/si_rollout.mp4')
  args = ap.parse_args(argv)
  run(args.seed, args.diff_ckpt, args.dstg_ckpt, args.fps, args.xmax, args.out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
