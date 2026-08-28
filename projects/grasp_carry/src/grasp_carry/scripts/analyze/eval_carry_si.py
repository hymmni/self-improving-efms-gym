r"""Phase 4 step 4 — 대조군 + 3개 SI arm을 같은 시드 블록으로 평가한다.

4개 정책:
  control    체크포인트 checkpoints/grasp_carry_diff100 (자기개선 없음, 순수 BC)
  A mean-succ   d=succ 체크포인트, statistic=mean  (논문 그대로 — 기준선)
  B mean-fail   d=fail 체크포인트, statistic=mean  (A 대비: 실패를 아는가)
  C cvar-fail   d=fail 체크포인트, statistic=cvar  (B 대비: 평균이냐 꼬리냐)

정책 롤아웃·지표 정의는 전부 `run_bc_stg_guided.py`(phase 3 산출물)를
재사용한다 — 새로 짜면 지표 정의가 갈려 비교가 무의미해진다(이 step
지시사항). 네 정책 모두 STG 후보 선별 없이(`BCStgGuided(ckpt, None)`)
디퓨전 정책이 뽑은 액션을 그대로 쓴다 — SI 학습 자체의 효과만 보기
위해서다.

    python eval_carry_si.py --episodes 200 --seed0 900000
"""

import argparse

from grasp_carry.scripts.analyze.run_bc_stg_guided import BCStgGuided, run, demos_per_1k_with_reset
from grasp_carry.config import CarryConfig

POLICIES = [
    ('control (no SI)', 'checkpoints/grasp_carry_diff100/predictor.pkl'),
    ('A mean-succ', 'checkpoints/grasp_carry_si_A_mean_succ/predictor.pkl'),
    ('B mean-fail', 'checkpoints/grasp_carry_si_B_mean_fail/predictor.pkl'),
    ('C cvar-fail', 'checkpoints/grasp_carry_si_C_cvar_fail/predictor.pkl'),
]

# 리셋 비용 민감도로 볼 전도(tipped) 비용 후보 — 성공/타임아웃은 기본값(20) 고정.
TIPPED_COST_GRID = [50.0, 200.0, 500.0]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--seed0', type=int, default=900000)
  ap.add_argument('--reset-cost-success', type=float, default=20.0)
  ap.add_argument('--reset-cost-timeout', type=float, default=20.0)
  ap.add_argument('--reset-cost-tipped', type=float, default=200.0)
  args = ap.parse_args()

  if args.seed0 < 900_000:
    raise ValueError('--seed0는 학습 시드 대역(<900000)과 겹치면 안 된다 — '
                     '평가는 held-out 900000+ 대역을 써라.')

  cfg = CarryConfig()
  reset_cost = {'success': args.reset_cost_success,
                'timeout': args.reset_cost_timeout,
                'tipped': args.reset_cost_tipped}

  rows = []
  for name, ckpt in POLICIES:
    pol = BCStgGuided(ckpt, None, seed=1)
    r = run(pol, args.episodes, args.seed0, cfg)
    rows.append((name, r))
    print(f'{name}: 완료 (outcomes={r["outcomes"]})', flush=True)

  print(f'\n{"정책":<20} {"성공률":>7} {"전도율":>7} {"타임아웃률":>9} '
        f'{"성공시평균스텝":>12} {"총스텝":>8} {"demos/1k":>9} '
        f'{"demos/1k(리셋)":>13}')
  for name, r in rows:
    n = r['episodes']
    tip_rate = r['outcomes'].get('tipped', 0) / n
    to_rate = r['outcomes'].get('timeout', 0) / n
    with_reset = demos_per_1k_with_reset(r, reset_cost)
    print(f'{name:<20} {r["success_rate"]:>7.1%} {tip_rate:>7.1%} '
          f'{to_rate:>9.1%} {r["mean_succ_steps"]:>12.1f} '
          f'{r["total_steps"]:>8d} {r["demos_per_1k_steps"]:>9.2f} '
          f'{with_reset:>13.2f}')
  print(f'\n(기본 리셋 비용 가정: 성공={reset_cost["success"]:g} '
        f'타임아웃={reset_cost["timeout"]:g} 전도={reset_cost["tipped"]:g} 스텝)')

  # ---- 리셋 비용 민감도: 전도 비용을 바꿔가며 순위가 뒤집히는지 확인 -------
  print(f'\n전도(tipped) 리셋 비용 민감도 (성공={args.reset_cost_success:g} '
        f'타임아웃={args.reset_cost_timeout:g} 고정, demos/1k(리셋) 값):')
  header = f'{"정책":<20}' + ''.join(f'{f"tipped={c:g}":>14}' for c in TIPPED_COST_GRID)
  print(header)
  for name, r in rows:
    vals = []
    for tip_cost in TIPPED_COST_GRID:
      rc = {'success': args.reset_cost_success,
            'timeout': args.reset_cost_timeout, 'tipped': tip_cost}
      vals.append(demos_per_1k_with_reset(r, rc))
    print(f'{name:<20}' + ''.join(f'{v:>14.2f}' for v in vals))

  print('\n순위(demos/1k(리셋) 내림차순):')
  for tip_cost in [args.reset_cost_tipped] + TIPPED_COST_GRID:
    rc = {'success': args.reset_cost_success,
          'timeout': args.reset_cost_timeout, 'tipped': tip_cost}
    ranked = sorted(rows, key=lambda nr: -demos_per_1k_with_reset(nr[1], rc))
    order = ' > '.join(f'{name}({demos_per_1k_with_reset(r, rc):.2f})'
                       for name, r in ranked)
    print(f'  tipped={tip_cost:g}: {order}')


if __name__ == '__main__':
  main()
