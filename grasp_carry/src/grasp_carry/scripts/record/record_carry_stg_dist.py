r"""실패하는 롤아웃에 STG 카테고리컬 분포를 나란히 붙여서 녹화.

왼쪽은 `record_carry.py`의 `draw_env`를 그대로 재사용한 장면, 오른쪽은 그 순간
관측에 대한 `checkpoints/grasp_carry_dstg_deadline` 예측기의 카테고리컬 분포
막대그래프다. `analyze_mu_sigma_highrisk.py`(2026-08-07)에서 "고위험 구간에선
평균이 비슷해도 분포 모양(분산)이 크게 다르다"를 정적 산점도로 확인했는데,
이 영상은 그 분포가 **한 에피소드 안에서 시간에 따라 실제로 어떻게 변해가는지**를
보여준다 — 특히 실패(전도)로 흘러가는 동안 분포가 퍼지는지/한쪽으로 쏠리는지.

빨간 세로선은 데드라인 B(=reset_cost+T̂, 이 예측기가 실패 판정 순간에 학습한 값)다.

    python record_carry_stg_dist.py --seed 52 \
        --out results/videos/grasp_carry_stg_dist_seed52.mp4
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
from grasp_carry.carry_stg_reward import StgReward
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(seed, dstg_ckpt, speed, torque_safety, fps, xmax, out):
  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  pol = ScriptedCarryPolicy(config=cfg, speed=speed, explore_range=None,
                             torque_safety=torque_safety)
  reward = StgReward(dstg_ckpt, statistic='mean')
  deadline_B = reward.meta.get('deadline_B')
  bin_vals = np.arange(reward.num_bins)

  env.reset(seed=seed); pol.reset()
  base_offset = None
  action = None

  fig, (ax_scene, ax_dist) = plt.subplots(
      1, 2, figsize=(11.5, 4.6), dpi=110, gridspec_kw={'width_ratios': [1, 1.3]})
  frames = []

  for t in range(cfg.max_steps):
    action = pol(env)
    _, _, term, trunc, info = env.step(action)

    if env.is_held() and base_offset is None:
      base_offset = _ee_block_dist(env)
    elif not env.is_held():
      base_offset = None

    obs = env._stacked_obs()[None]
    probs = reward._probs(obs)[0]
    mu, sigma = reward.mean_std(obs)
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

    fig.suptitle(f'seed={seed}  t={t}  outcome={info["outcome"]}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    frames.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                  .reshape(int(h), int(w), 4)[..., :3].copy())

    if term or trunc:
      break

  frames += [frames[-1]] * (fps * 2)   # 마지막 프레임(실패 순간) 2초 정지
  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'seed={seed}: {info["steps"]}스텝, outcome={info["outcome"]}')
  print(f'saved {out} ({len(frames)} frames)')


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--seed', type=int, default=1, help='v4 검증: 재파지 없이 한 번에 성공하는 시드')
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_succ_v4/predictor.pkl')
  ap.add_argument('--speed', type=float, default=None,
                   help='None이면 CarryConfig 기본(max_accel/k_p) 사용 — v4 수집 설정과 동일')
  ap.add_argument('--torque-safety', type=float, default=0.05,
                   help='v4 데이터 수집 때 쓴 값(실측 보정)')
  ap.add_argument('--fps', type=int, default=8)
  ap.add_argument('--xmax', type=int, default=120, help='분포 막대그래프 x축 상한(bin)')
  ap.add_argument('--out', default='results/videos/grasp_carry_stg_dist.mp4')
  args = ap.parse_args(argv)
  run(args.seed, args.dstg_ckpt, args.speed, args.torque_safety, args.fps, args.xmax, args.out)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
