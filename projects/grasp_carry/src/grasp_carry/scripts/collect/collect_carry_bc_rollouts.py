r"""학습된(스크립트 아님) diffusion BC 정책을 실제로 굴려서, 성공+실패가 섞인
Stage-2 스타일 롤아웃 데이터를 `collect_carry_demos.py`와 동일 포맷으로 저장한다.

`src/train_carry_dstg.py --include-failures --fail-mode deadline`이 기대하는
스키마(`observation.frame`, `action`, `is_success`, `time_to_success`,
`episode_id`, `meta.max_steps`, `meta.outcomes`)를 그대로 따른다 — 실패
에피소드의 모든 스텝은 `time_to_success = max_steps`(마지막 bin)로 라벨링되고,
`train_carry_dstg.py`의 deadline 모드가 이걸 다시 (판정 순간=B, 그 전=부트스트랩)
라벨로 바꿔서 학습한다.

시연(demo)과 다른 점: 정책은 시연자의 상태기계가 아니라 학습된 diffusion 정책
(`rollout_carry_diff_stats.py::load_diff_policy`)이 매 스텝 낸다 — 그래서 실패가
"인위적으로 만든" 게 아니라 정책이 실제로 저지른 것이다(SI-EFM Stage-2와 동일).

실행:
    python -m grasp_carry.scripts.collect.collect_carry_bc_rollouts \
        --diff-ckpt checkpoints/grasp_carry_diff100_v5/predictor.pkl \
        --episodes 600 --seed0 900000 \
        --out data/grasp_carry_bc_v5_rollouts.pkl
"""

import argparse
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from grasp_carry.scripts.analyze.rollout_carry_diff_stats import load_diff_policy
from grasp_carry.train_carry_predictor import concat_obs
from grasp_carry.config import CarryConfig
from grasp_carry.env import FRAME_FIELDS, GraspCarry2D


def collect(diff_ckpt: str, n_episodes: int, seed0: int) -> dict:
  ck, nets, normalize_obs, normalize_action, unnormalize_action = load_diff_policy(diff_ckpt)
  _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))

  # 2026-08-11: --horizon>1(액션 청킹) 체크포인트 지원. 청킹 안 쓰는 체크포인트는
  # meta에 horizon이 없으므로 기본값 1(기존 동작과 동일)로 되돌아간다.
  m = ck['meta']
  horizon = int(m.get('horizon', 1))
  exec_horizon = int(m.get('exec_horizon', horizon))
  act_dim = int(m.get('act_dim', len(ck['norm_stats']['act_mean'])))

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  obs_all, act_all, ttg_all, eid_all, succ_all = [], [], [], [], []
  outcomes: dict = {}
  kept = 0
  kk = jax.random.PRNGKey(1234)

  for e in range(n_episodes):
    obs, info = env.reset(seed=seed0 + e)
    obs_l, act_l = [obs], []
    terminated = truncated = False

    while len(act_l) < cfg.max_steps and not (terminated or truncated):
      o = {'frame': np.asarray(obs, np.float32)[None]}
      c = np.asarray(concat_obs(normalize_obs(o)))
      kk, s2 = jax.random.split(kk)
      chunk_n = np.asarray(_sample(ck['params'], jnp.asarray(c), s2))[0].reshape(horizon, act_dim)
      chunk = np.asarray(unnormalize_action(chunk_n), np.float32)
      # receding horizon: 청크 중 exec_horizon개만 실행하고 새 관측으로 재추론
      for h in range(min(exec_horizon, horizon)):
        if len(act_l) >= cfg.max_steps:
          break
        a = chunk[h]
        act_l.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
          break
        obs_l.append(obs)

    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    is_success = info['outcome'] == 'success'

    L = len(act_l)
    assert len(obs_l) == L, '관측/액션 스텝 수 불일치'
    if is_success:
      ttg = (L - 1 - np.arange(L)).astype(np.float32)
    else:
      ttg = np.full(L, cfg.max_steps, dtype=np.float32)
    obs_all.append(np.array(obs_l, dtype=np.float32))
    act_all.append(np.array(act_l, dtype=np.float32))
    succ_all.append(np.full(L, is_success, dtype=bool))
    ttg_all.append(ttg)
    eid_all.append(np.full(L, kept, dtype=np.int32))
    kept += 1

  data = {
      'observation': {'frame': np.concatenate(obs_all)},
      'action': np.concatenate(act_all),
      'time_to_success': np.concatenate(ttg_all),
      'episode_id': np.concatenate(eid_all),
      'is_success': np.concatenate(succ_all),
      'meta': {
          'source': f'collect_carry_bc_rollouts.py: diff BC policy ({diff_ckpt}) rollouts',
          'diff_ckpt': diff_ckpt,
          'frame_fields': list(FRAME_FIELDS),
          'obs_history': cfg.obs_history,
          'action_fields': ('x', 'y', 'theta', 'grip'),
          'keep_failures': True,
          'failure_bin': cfg.max_steps,
          'requested_episodes': n_episodes,
          'kept_episodes': kept,
          'outcomes': outcomes,
          'seed0': seed0,
          'control_hz': cfg.control_hz,
          'max_steps': cfg.max_steps,
      },
  }
  print(f'수집 완료: {kept}/{n_episodes} 에피소드(성공+실패 모두 보존), '
        f'{len(data["time_to_success"])} 스텝, outcomes={outcomes}')
  return data


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100_v5/predictor.pkl')
  ap.add_argument('--episodes', type=int, default=600)
  ap.add_argument('--seed0', type=int, default=900000,
                   help='BC 정책 학습 시드 대역(<900000)과 안 겹치는 held-out 시작 시드')
  ap.add_argument('--out', default='data/grasp_carry_bc_rollouts.pkl')
  args = ap.parse_args(argv)

  data = collect(args.diff_ckpt, args.episodes, args.seed0)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump(data, fp)
  print(f'saved {args.out}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
