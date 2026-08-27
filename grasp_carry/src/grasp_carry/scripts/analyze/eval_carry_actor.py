r"""`train_carry_actor.py`로 학습한 정책(AI-E/AI-R 보상)을 실제 시뮬레이터로 채점.

정책 가중치는 예측기만 쿼리해서(실제 롤아웃 없이) 학습됐다 — 여기서 처음으로
**진짜 물리**에 굴려서, 예측기(보상 모델)를 거쳐 학습된 정책이 실제로 얼마나
안전하고 효율적인지 확인한다.

    python eval_carry_actor.py --episodes 200
"""

import argparse
import pickle

import numpy as np
import jax.numpy as jnp

from grasp_carry.scripts.train.train_carry_actor import build_actor, LEAD_MIN, LEAD_MAX
from grasp_carry.scripts.analyze.run_bc_stg_guided import run, demos_per_1k_with_reset
from grasp_carry.config import CarryConfig
from grasp_carry.policy import ScriptedCarryPolicy

NOMINAL_SPEED = 150.0   # 연직 리드(_lift_lead)를 정하는 값 — 데이터 수집 때와 통일


class ActorSelector:
  """학습된 actor를 `speed_selector`로 감싼다."""

  def __init__(self, actor_ckpt: str):
    self.ck = pickle.load(open(actor_ckpt, 'rb'))
    self.apply_fn, _ = build_actor(self.ck['obs_dim'])

  def __call__(self, env, contact_len, arm):
    obs = np.asarray(env._stacked_obs(), dtype=np.float32)[None]
    obs_n = (obs - self.ck['norm_stats']['frame_mean']) / self.ck['norm_stats']['frame_std']
    lead = self.apply_fn(self.ck['params'], jnp.asarray(obs_n))
    return float(np.clip(np.asarray(lead)[0], LEAD_MIN, LEAD_MAX))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--seed0', type=int, default=900000)
  ap.add_argument('--exp-actor', default='checkpoints/grasp_carry_actor_exponly/actor.pkl')
  ap.add_argument('--risk-actor', default='checkpoints/grasp_carry_actor_risk/actor.pkl')
  ap.add_argument('--cvar-actor', default='checkpoints/grasp_carry_actor_cvar/actor.pkl')
  ap.add_argument('--renewal-actor', default='checkpoints/grasp_carry_actor_renewal/actor.pkl')
  ap.add_argument('--skip-cvar', action='store_true')
  ap.add_argument('--skip-renewal', action='store_true')
  ap.add_argument('--reset-cost-success', type=float, default=20.0)
  ap.add_argument('--reset-cost-timeout', type=float, default=20.0)
  ap.add_argument('--reset-cost-tipped', type=float, default=200.0)
  args = ap.parse_args()
  reset_cost = {'success': args.reset_cost_success, 'timeout': args.reset_cost_timeout,
                'tipped': args.reset_cost_tipped}

  cfg = CarryConfig()
  sel_e = ActorSelector(args.exp_actor)
  sel_r = ActorSelector(args.risk_actor)

  arms = [
      ('AI-E 보상(평균) 학습',
       ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED, speed_selector=sel_e)),
      ('AI-R 보상(평균) 학습',
       ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED, speed_selector=sel_r)),
      ('물리식 (학습 없음, 참고군)',
       ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED)),
  ]
  if not args.skip_cvar:
    sel_c = ActorSelector(args.cvar_actor)
    arms.insert(2, ('AI-R 보상(CVaR) 학습',
                    ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED,
                                        speed_selector=sel_c)))
  if not args.skip_renewal:
    sel_n = ActorSelector(args.renewal_actor)
    arms.insert(3 if not args.skip_cvar else 2,
                ('AI-R 보상(renewal) 학습',
                 ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED,
                                     speed_selector=sel_n)))

  rows = []
  for name, pol in arms:
    r = run(pol, args.episodes, args.seed0, cfg)
    rows.append((name, r))
    print(f'{name}: 완료', flush=True)

  print(f'\n{"팔":<26} {"성공률":>7} {"성공시 스텝":>10} '
        f'{"총 스텝":>9} {"1k당 데모":>10} {"리셋비용 반영":>13}')
  for name, r in rows:
    wr = demos_per_1k_with_reset(r, reset_cost)
    print(f'{name:<26} {r["success_rate"]:>7.1%} {r["mean_succ_steps"]:>10.1f} '
          f'{r["total_steps"]:>9d} {r["demos_per_1k_steps"]:>10.2f} {wr:>13.2f}')
  print(f'\n(리셋 비용 가정: 성공={reset_cost["success"]:g} '
        f'타임아웃={reset_cost["timeout"]:g} 전도={reset_cost["tipped"]:g} 스텝)')
  print()
  for name, r in rows:
    print(f'  {name}: outcomes={r["outcomes"]}')


if __name__ == '__main__':
  main()
