r"""파지직후 mu 급등이 "쌍봉분포 붕괴 + B(=reset_cost+T_hat) 아래에서의 상승"으로
설명되는지 검증한다 (사용자 가설, 2026-08-09).

`rollout_carry_diff_stats.py`가 찾은 mu 급등(파지 후 6스텝 안에서 어떤 한 스텝의
증가폭이 5 초과)을 실제로 다시 골라내, 그 급등 전/후 스텝의 카테고리컬 분포
모양(단봉/쌍봉)과, 급등 후 mu가 B를 넘는지를 직접 확인한다.

B = reset_cost + T_hat. T_hat은 이 롤아웃 자체의 성공 에피소드 길이 분포에서
(IQR 이상치 제외 후) 낸 평균 — "포기 판정"은 지금 굴리고 있는 정책 기준으로
해야 의미가 있어서 v3 demo가 아니라 이 정책의 롤아웃 길이를 쓴다.

실행:
    python -m grasp_carry.scripts.analyze.analyze_mu_jump_bimodal --diff-ckpt checkpoints/grasp_carry_diff100_v5/predictor.pkl
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from grasp_carry.scripts.analyze.rollout_carry_diff_stats import load_diff_policy
from grasp_carry.train_carry_predictor import concat_obs
from grasp_carry.carry_stg_reward import StgReward
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D


def is_bimodal(probs, min_prominence=0.02, min_gap_bins=3):
  """카테고리컬 분포에서 국소 최대값(피크)이 2개 이상이고, 그 사이 골이
  충분히 파여 있으면(각 피크 대비 절반 이하) 쌍봉으로 본다."""
  peaks = []
  for i in range(1, len(probs) - 1):
    if probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1] and probs[i] > min_prominence:
      peaks.append(i)
  if len(peaks) < 2:
    return False, peaks
  peaks.sort(key=lambda i: -probs[i])
  p0, p1 = peaks[0], peaks[1]
  if abs(p0 - p1) < min_gap_bins:
    return False, peaks
  lo, hi = min(p0, p1), max(p0, p1)
  valley = probs[lo:hi + 1].min()
  return valley < 0.5 * min(probs[p0], probs[p1]), [p0, p1]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100_v5/predictor.pkl')
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_succ_v4/predictor.pkl')
  ap.add_argument('--episodes', type=int, default=300)
  ap.add_argument('--seed0', type=int, default=900000)
  ap.add_argument('--reset-cost', type=float, default=30.0)
  ap.add_argument('--outlier-iqr-mult', type=float, default=1.5)
  ap.add_argument('--jump-thresh', type=float, default=5.0)
  ap.add_argument('--window', type=int, default=6)
  args = ap.parse_args()

  ck, nets, normalize_obs, normalize_action, unnormalize_action = load_diff_policy(args.diff_ckpt)
  _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
  reward = StgReward(args.dstg_ckpt, statistic='mean')

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  succ_lens = []
  episodes_data = []   # 각 원소: dict(mu_traj, probs_traj, outcome)

  kk = jax.random.PRNGKey(1234)
  for e in range(args.episodes):
    obs, info = env.reset(seed=args.seed0 + e)
    held_prev = False
    n_grasp_events = 0
    mu_traj, probs_traj = [], []

    for t in range(cfg.max_steps):
      o = {'frame': np.asarray(obs, np.float32)[None]}
      c = np.asarray(concat_obs(normalize_obs(o)))
      kk, s2 = jax.random.split(kk)
      a_n = np.asarray(_sample(ck['params'], jnp.asarray(c), s2))[0]
      a = np.asarray(unnormalize_action(a_n), np.float32)
      obs, r, term, trunc, info = env.step(a)

      held_now = env.is_held()
      if held_now and not held_prev:
        n_grasp_events += 1
      held_prev = held_now

      if n_grasp_events >= 1 and len(mu_traj) < args.window:
        stacked = env._stacked_obs()[None]
        probs = reward._probs(stacked)[0]
        mu, _ = reward.mean_std(stacked)
        mu_traj.append(float(mu[0]))
        probs_traj.append(np.asarray(probs))

      if term or trunc:
        break

    if info['outcome'] == 'success':
      succ_lens.append(info['steps'])
    if mu_traj:
      episodes_data.append(dict(mu=mu_traj, probs=probs_traj, outcome=info['outcome']))

  succ_lens = np.asarray(succ_lens, dtype=np.float64)
  q1, q3 = np.percentile(succ_lens, [25, 75])
  iqr = q3 - q1
  upper = q3 + args.outlier_iqr_mult * iqr
  typical = succ_lens[succ_lens <= upper]
  T_hat = float(typical.mean())
  B = args.reset_cost + T_hat
  print(f'성공 에피소드 길이: n={len(succ_lens)} mean={succ_lens.mean():.1f} '
        f'(이상치 제외 T_hat={T_hat:.1f}, 제외 {len(succ_lens)-len(typical)}개)')
  print(f'B = reset_cost({args.reset_cost}) + T_hat({T_hat:.1f}) = {B:.1f}\n')

  jump_cases = []
  for ep in episodes_data:
    mu = np.asarray(ep['mu'])
    diffs = np.diff(mu)
    if len(diffs) == 0 or diffs.max() <= args.jump_thresh:
      continue
    j = int(np.argmax(diffs))   # 급등이 일어난 스텝 인덱스(전->후)
    jump_cases.append(dict(mu_before=mu[j], mu_after=mu[j + 1],
                           probs_before=ep['probs'][j], probs_after=ep['probs'][j + 1],
                           outcome=ep['outcome']))

  print(f'=== 급등 케이스: {len(jump_cases)}/{len(episodes_data)} ===\n')
  under_B = 0
  bimodal_before_count = 0
  for i, c in enumerate(jump_cases):
    bim_before, peaks_before = is_bimodal(c['probs_before'])
    bim_after, peaks_after = is_bimodal(c['probs_after'])
    if c['mu_after'] < B:
      under_B += 1
    if bim_before:
      bimodal_before_count += 1
    if i < 10:
      print(f"  케이스{i}: mu {c['mu_before']:.1f} -> {c['mu_after']:.1f}  "
            f"(B={B:.1f}, 급등후<B: {c['mu_after'] < B})  outcome={c['outcome']}  "
            f"급등전 쌍봉={bim_before}(피크bin={peaks_before})  "
            f"급등후 쌍봉={bim_after}(피크bin={peaks_after})")

  n = len(jump_cases)
  if n:
    print(f'\n요약: 급등 후 mu < B  {under_B}/{n} ({under_B/n:.1%})')
    print(f'      급등 "전" 분포가 쌍봉  {bimodal_before_count}/{n} ({bimodal_before_count/n:.1%})')


if __name__ == '__main__':
  main()
