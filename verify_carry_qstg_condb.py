r"""조건 B로 걸린 예측을 실제 물리로 재검증한다.

`probe_carry_qstg.py`는 저장된 관측에 가상 액션(리드 12.5 vs 150mm)을 넣어
예측기를 쿼리했다 — 실제로 그 액션을 실행해본 게 아니라서, 학습 분포 밖으로
외삽하고 있을 위험이 있었다. 이 스크립트는 조건 B로 걸린 상태들이 나온
**에피소드(시드)** 를 골라, 그 시드로 실제 `ScriptedCarryPolicy`를 두 속도로
(explore 없이, 고정 리드로) 처음부터 다시 굴려서 예측과 실측이 맞는지 본다.

주의: 이건 프로브가 본 "그 중간 상태에서부터"가 아니라 "그 시드(=은닉
질량·마찰·기하)로 처음부터" 비교다 — pymunk 물리 상태(속도 등)는 60차원
관측에 없어 중간 상태를 정확히 복원할 수 없기 때문이다. 대신
`calibrate_carry.py`가 원래 썼던 것과 같은 비교 단위(전체 롤아웃, 고정
속도)라 조건 A/B 정의와 맞는다.

    python verify_carry_qstg_condb.py
"""

import argparse
import pickle

import numpy as np

from probe_carry_qstg import SLOW_LEAD, FAST_LEAD, run_probe
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy


def rollout(seed: int, speed: float, cfg: CarryConfig) -> dict:
  env = GraspCarry2D(cfg)
  policy = ScriptedCarryPolicy(speed=speed, config=cfg)   # explore 없음(고정 속도)
  env.reset(seed=seed)
  policy.reset()
  info = env._info()
  for _ in range(cfg.max_steps):
    _, _, term, trunc, info = env.step(policy(env))
    if term or trunc:
      break
  return dict(outcome=info['outcome'], steps=info['steps'])


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ckpt', default='checkpoints/grasp_carry_qstg/predictor.pkl')
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  args = ap.parse_args()

  r = run_probe(args.ckpt, args.data, val_only=True)
  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  seed0 = int(data['meta']['seed0'])
  cfg = CarryConfig()

  flagged_eps = sorted(set(r['episode_id'][r['cond_b']].tolist()))
  print(f'조건 B로 걸린 상태가 나온 고유 에피소드: {len(flagged_eps)}개')
  if not flagged_eps:
    print('걸린 에피소드가 없다 — 검증할 게 없다.')
    return

  print(f"\n{'seed':>5} {'예측 E[성공|s]':>13} {'예측 q0.8[s]':>12} "
        f"{'예측 E[성공|f]':>13} {'예측 q0.8[f]':>12}  |  "
        f"{'실측 느림':>14} {'실측 빠름':>14}")
  n_a = 0
  for eid in flagged_eps:
    m = r['episode_id'] == eid
    seed = seed0 + int(eid)
    slow = rollout(seed, SLOW_LEAD, cfg)
    fast = rollout(seed, FAST_LEAD, cfg)
    pred_mean_s = r['mean_slow'][m].mean(); pred_q80_s = r['q80_slow'][m].mean()
    pred_mean_f = r['mean_fast'][m].mean(); pred_q80_f = r['q80_fast'][m].mean()
    real_a = (slow['outcome'] == 'success' and fast['outcome'] != 'success')
    n_a += int(real_a)
    slow_s = f"{slow['outcome']}/{slow['steps']}"
    fast_s = f"{fast['outcome']}/{fast['steps']}"
    print(f"{seed:>5} {pred_mean_s:>13.1f} {pred_q80_s:>12.1f} "
          f"{pred_mean_f:>13.1f} {pred_q80_f:>12.1f}  |  "
          f"{slow_s:>14} {fast_s:>14}")

  print(f'\n실측: 느림은 성공, 빠름은 실패(=진짜 조건 A) — {n_a}/{len(flagged_eps)}개 시드')
  print('(조건 B 자체는 "성공 궤적 분포의 분위수"라 단일 롤아웃 1회로는 '
        '재현 불가 — 위 표의 예측 q0.8은 참고용, 실측 검증은 조건 A만 가능)')


if __name__ == '__main__':
  main()
