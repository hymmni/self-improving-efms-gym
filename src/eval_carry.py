r"""`GraspCarry2D` + `ScriptedCarryPolicy` 평가 CLI (phase 3, step 3).

    python -m src.eval_carry --episodes 50 [--speed S] [--no-regrasp] [--seed0 N]

출력: 성공률, 성공 에피소드의 스텝 통계(평균/중앙값/q0.8), 실패 원인 분류
(넘어짐/타임아웃/기타), 재파지 비율·평균 낙하 횟수·평균 파지 접촉 길이.

실패 원인에서 **타임아웃**은 정책이 어딘가에서 교착했다는 뜻이라 넘어짐과 성격이
다르다. 넘어짐은 은닉 물성 때문에 원리적으로 피할 수 없는 실패지만, 타임아웃은
정책의 결함이다. 그래서 두 가지를 반드시 나눠 센다.
"""

import argparse
from typing import List, Optional

import numpy as np

from .grasp_carry.config import CarryConfig
from .grasp_carry.env import GraspCarry2D
from .grasp_carry.policy import ScriptedCarryPolicy


def rollout(env: GraspCarry2D, policy: ScriptedCarryPolicy, seed: int) -> dict:
  """한 에피소드를 끝까지 굴리고 요약 dict를 준다."""
  env.reset(seed=seed)
  policy.reset()
  info = env._info()
  for _ in range(env.cfg.max_steps):
    _, _, terminated, truncated, info = env.step(policy(env))
    if terminated or truncated:
      break
  return {
      'seed': seed,
      'outcome': info['outcome'],
      'steps': info['steps'],
      'mass': info['mass'],
      'friction': info['friction'],
      'src_width': env.src_box.inner_width,
      'block_h': env.block_h,
      'regrasped': policy.regrasped,
      'n_drops': policy.n_drops,
      'contact': (float(np.mean(policy.grasp_contacts))
                  if policy.grasp_contacts else 0.0),
      'speeds': list(policy.grasp_speeds),
  }


def evaluate(episodes: int, seed0: int = 0, speed: Optional[float] = None,
             allow_regrasp: bool = True, privileged: bool = False,
             config: Optional[CarryConfig] = None) -> List[dict]:
  cfg = config or CarryConfig()
  env = GraspCarry2D(cfg)
  policy = ScriptedCarryPolicy(speed=speed, allow_regrasp=allow_regrasp,
                               privileged=privileged, config=cfg)
  return [rollout(env, policy, seed0 + i) for i in range(episodes)]


def summarize(rows: List[dict]) -> dict:
  n = len(rows)
  ok = [r for r in rows if r['outcome'] == 'success']
  steps = np.array([r['steps'] for r in ok], dtype=float)
  # 실패 원인 분류. 'tipped'는 넘어짐, 'timeout'은 교착, 나머지는 기타.
  tipped = sum(1 for r in rows if r['outcome'] == 'tipped')
  timeout = sum(1 for r in rows if r['outcome'] == 'timeout')
  other = n - len(ok) - tipped - timeout
  return {
      'n': n,
      'success': len(ok),
      'success_rate': len(ok) / n if n else 0.0,
      'mean_steps': float(steps.mean()) if len(steps) else float('nan'),
      'median_steps': float(np.median(steps)) if len(steps) else float('nan'),
      'q80_steps': (float(np.quantile(steps, 0.8)) if len(steps)
                    else float('nan')),
      'tipped': tipped,
      'timeout': timeout,
      'other': other,
      'regrasp_rate': float(np.mean([r['regrasped'] for r in rows])),
      'mean_drops': float(np.mean([r['n_drops'] for r in rows])),
      'mean_contact': float(np.mean([r['contact'] for r in rows])),
  }


def report(rows: List[dict], verbose: bool = False) -> dict:
  s = summarize(rows)
  print(f"episodes            : {s['n']}")
  print(f"success rate        : {s['success_rate']:.1%} "
        f"({s['success']}/{s['n']})")
  print(f"success steps       : mean {s['mean_steps']:.1f} | "
        f"median {s['median_steps']:.1f} | q0.8 {s['q80_steps']:.1f}")
  print(f"failures            : tipped {s['tipped']} | "
        f"timeout {s['timeout']} | other {s['other']}")
  print(f"regrasp rate        : {s['regrasp_rate']:.1%}")
  print(f"mean drops/episode  : {s['mean_drops']:.2f}")
  print(f"mean grasp contact  : {s['mean_contact']:.1f} mm")
  if verbose:
    print('\nseed outcome   steps src_w block_h mass  mu    rg drops contact')
    for r in rows:
      print(f"{r['seed']:4d} {r['outcome']:9s} {r['steps']:5d} "
            f"{r['src_width']:5.1f} {r['block_h']:7.1f} {r['mass']:.3f} "
            f"{r['friction']:.2f} {int(r['regrasped']):2d} "
            f"{r['n_drops']:5d} {r['contact']:7.1f}")
  return s


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=50)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--speed', type=float, default=None,
                  help='명목 명령 리드(mm/스텝). 기본은 max_accel/k_p.')
  ap.add_argument('--no-regrasp', action='store_true',
                  help='재파지를 금지하고 항상 직접 운반한다.')
  ap.add_argument('--privileged', action='store_true',
                  help='은닉 물성(질량·마찰)의 실제값을 보고 속도를 고른다.')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args(argv)

  rows = evaluate(args.episodes, seed0=args.seed0, speed=args.speed,
                  allow_regrasp=not args.no_regrasp,
                  privileged=args.privileged)
  report(rows, verbose=args.verbose)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
