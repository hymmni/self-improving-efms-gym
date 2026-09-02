r"""학습된(스크립트가 아닌) diffusion BC 정책을 실제로 롤아웃해서, 그동안
v4 스크립트 데모/재파지 스캔에서 확인했던 통계량들을 정책 롤아웃 기준으로
다시 낸다.

배경: 지금까지 보여준 영상·통계(길이 분포, 재파지율, 박스 안/밖 재파지 비율,
파지 직후 STG μ 매끄러움)는 전부 **스크립트 시연 정책**(`ScriptedCarryPolicy`)
으로 만든 것이었다 — v4 데이터 자체가 그 정책으로 수집됐기 때문이다. 학습된
정책(`checkpoints/grasp_carry_diff100_v4`)을 실제로 굴려보면 숫자가 달라질
수 있다 — 특히 재파지처럼 스크립트 정책의 명시적 상태기계 로직에 기대던
행동은 학습된 정책이 데모에서 얼마나 잘 모방했는지에 달려 있다.

"재파지"는 학습된 정책엔 명시적 상태기계가 없으므로, `env.is_held()`가
False->True로 바뀌는 횟수(=파지 시도 횟수)로 프록시한다. 파지 시도가
2회 이상이면 재파지가 있었다고 본다. "박스 안/밖" 재파지는 그 두 번째
이후 파지 시점의 EE x좌표가 src_box 안인지 between-boxes 구간인지로
판정한다(`src/grasp_carry/policy.py::_regrasp_spot`의 판정 조건과 동일한
경계값을 재사용).

실행:
    python -m grasp_carry.scripts.analyze.rollout_carry_diff_stats --episodes 300
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from grasp_carry.train_carry_predictor import (
    make_normalizers, concat_obs, OBS_FIELDS)
from grasp_carry.diffusion_act import build_diffusion_act_chunk
from grasp_carry.carry_stg_reward import StgReward
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D


def load_diff_policy(ckpt_path):
  import pickle
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  # 2026-08-11: --horizon>1(액션 청킹) 체크포인트 지원. 청킹 안 쓰는 체크포인트는
  # meta에 horizon이 없으므로 기본값 1(기존 동작과 동일)로 되돌아간다.
  horizon = int(m.get('horizon', 1))
  nets = build_diffusion_act_chunk(
      (256, 256, 256), act_dim * horizon, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=horizon, act_dim=act_dim)
  normalize_obs, normalize_action, unnormalize_action = make_normalizers(ck['norm_stats'])
  return ck, nets, normalize_obs, normalize_action, unnormalize_action


def ee_x(env):
  # From: src/grasp_carry/policy.py _regrasp_spot()/_descend_limit() — 같은
  # x 접근 방식(gripper.base.position.x)을 써야 박스 안/밖 경계 판정이 일관됨.
  return float(env.gripper.base.position.x)


def classify_regrasp_spot(env, x):
  """policy.py::_regrasp_spot 경계(box outer/inner)와 동일한 기준으로 x가
  src_box 안인지 between-boxes(두 박스 사이 바닥)인지 분류."""
  src, tgt = env.src_box, env.tgt_box
  if src.left_outer <= x <= src.right_outer:
    return 'inside_src'
  if src.right_outer < x < tgt.left_outer:
    return 'between_boxes'
  return 'other'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100_v4/predictor.pkl')
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_succ_v4/predictor.pkl')
  ap.add_argument('--episodes', type=int, default=300)
  ap.add_argument('--seed0', type=int, default=900000,
                   help='학습 시드 대역(<900000)과 안 겹치는 held-out 시작 시드')
  args = ap.parse_args()

  ck, nets, normalize_obs, normalize_action, unnormalize_action = load_diff_policy(args.diff_ckpt)
  _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
  reward = StgReward(args.dstg_ckpt, statistic='mean')

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  outcomes = {}
  succ_lens = []
  regrasp_flags = []
  regrasp_spot_counts = {'inside_src': 0, 'between_boxes': 0, 'other': 0}
  mu_post_grasp = []   # 각 에피소드: 파지 직후 최대 6스텝의 mu 리스트

  kk = jax.random.PRNGKey(1234)
  for e in range(args.episodes):
    obs, info = env.reset(seed=args.seed0 + e)
    held_prev = False
    n_grasp_events = 0
    grasp_spots = []
    mu_traj = []

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
        grasp_spots.append(ee_x(env))
      held_prev = held_now

      if n_grasp_events >= 1 and len(mu_traj) < 6:
        stacked = env._stacked_obs()[None]
        mu, _ = reward.mean_std(stacked)
        mu_traj.append(float(mu[0]))

      if term or trunc:
        break

    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    if info['outcome'] == 'success':
      succ_lens.append(info['steps'])
    regrasp = n_grasp_events >= 2
    regrasp_flags.append(regrasp)
    if regrasp:
      spot = classify_regrasp_spot(env, grasp_spots[1])
      regrasp_spot_counts[spot] += 1
    if mu_traj:
      mu_post_grasp.append(mu_traj)

  n = args.episodes
  succ_rate = outcomes.get('success', 0) / n
  print(f'=== 학습된 diff BC 정책 롤아웃 통계 ({n}에피소드, seed {args.seed0}+) ===')
  print(f'outcomes = {outcomes}  (성공률 {succ_rate:.1%})')
  if succ_lens:
    a = np.asarray(succ_lens)
    print(f'성공 에피소드 길이: mean={a.mean():.1f} std={a.std():.1f} median={np.median(a):.0f} '
          f'p90={np.percentile(a,90):.0f} p95={np.percentile(a,95):.0f} '
          f'p99={np.percentile(a,99):.0f} max={a.max():.0f}')
  n_regrasp = sum(regrasp_flags)
  print(f'재파지(파지 시도>=2회) 비율: {n_regrasp}/{n} ({n_regrasp/n:.1%})')
  print(f'재파지 지점 분류: {regrasp_spot_counts}')

  # 파지 직후 mu 매끄러움: 각 스텝 대비 다음 스텝 mu가 급등(>5)하는 케이스 비율
  jump_count = 0
  for traj in mu_post_grasp:
    diffs = np.diff(traj)
    if len(diffs) and diffs.max() > 5.0:
      jump_count += 1
  if mu_post_grasp:
    print(f'파지직후 mu 급등(>5) 에피소드: {jump_count}/{len(mu_post_grasp)} '
          f'({jump_count/len(mu_post_grasp):.1%})')
    sample = mu_post_grasp[:6]
    for i, traj in enumerate(sample):
      print(f'  예시{i}: {["%.1f" % v for v in traj]}')


if __name__ == '__main__':
  main()
