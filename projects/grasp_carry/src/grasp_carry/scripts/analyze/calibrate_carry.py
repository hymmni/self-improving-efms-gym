r"""`GraspCarry2D` 캘리브레이션 — 위험 민감 구조가 실제로 성립하는지 검증한다
(phase 3, step 5).

    python calibrate_carry.py [--episodes N] [--seed0 N] [--speeds S...]

이 환경은 steps-to-go(STG) 예측 **분포의 분산·분위수**가 기댓값 이상의 정보를
주는지 검증하려고 만들었다. 그러려면 아래 구조가 물리적으로 성립해야 한다.
**이 스크립트는 그것을 확인하는 것이지 만들어내는 것이 아니다** — 조건이
성립하지 않으면 그 사실이 산출물이다(이전에 GraspAngleTransport 설계가 조건 B
0/96으로 실패했고, 그 보고가 설계 폐기의 근거가 됐다).

이전 버전(v1~v3)의 이 파일은 물리 없이 **슬립 모델만** 시뮬레이션하는 Phase 0
스윕이었다. 지금은 `src/grasp_carry`의 실제 pymunk 환경과 스크립트 정책을
그대로 굴린다.

검증 1. **얕은 파지 / 깊은 파지의 분리**
    소스 박스 내폭이 그리퍼 바깥폭(136mm)보다 좁으면 그리퍼가 rim 아래로
    못 내려가 블록 윗부분만 물어야 한다. 그 결과 접촉 길이가 짧아지고 무게중심
    까지의 지렛대(`arm`)가 길어져 회전 토크 여유가 줄어야 한다.

검증 2. **재파지의 효용**
    `allow_regrasp=True/False`를 같은 시드 집합에서 비교한다. 재파지가
    "느리지만 안전"이면 성공률은 오르고 평균 스텝은 늘어야 한다.

검증 3. **속도-위험 트레이드오프와 부호 전환**
    명령 리드(`speed`)를 바꿔가며,

        조건 A: mean(f) < mean(s)  AND  success(f) < success(s)
        조건 B: mean(f) < mean(s)  AND  q0.8(f)   > q0.8(s)

    를 만족하는 (빠른 f, 느린 s) 쌍을 센다. 조건 B가 "같은 결정 지점에서
    기댓값과 0.8분위수가 반대 행동을 선호한다"는 가설의 실증이다.

STG는 논문(SI-EFM)이 성공 데모만 학습에 쓰므로 **성공 에피소드만** 집계한다.
"""

import argparse
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from grasp_carry.eval_carry import rollout
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy

# 검증 3의 기본 스윕. 첫 값은 로봇 명목 리드(`max_accel/k_p` = 12.5mm)이고
# 나머지는 그 배수다. 명목보다 큰 리드는 가속도 스펙을 넘는 명령이지만,
# `speed`는 **정책 파라미터**이므로 환경 설정을 바꾸는 것이 아니다.
DEFAULT_SPEEDS = (12.5, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0)


# ============================================================== 롤아웃/집계
def run_config(seeds: Sequence[int], speed: Optional[float] = None,
               allow_regrasp: bool = True,
               cfg: Optional[CarryConfig] = None) -> List[dict]:
  """한 설정으로 `seeds` 전부를 굴린다.

  `env.reset(seed=...)`이 RNG를 그 시드로 다시 만들기 때문에, 설정이 달라도
  같은 시드는 **같은 에피소드**(블록 치수·은닉 물성·박스 내폭)를 준다. 설정 간
  비교가 짝지어진(paired) 비교가 되는 근거다.
  """
  cfg = cfg or CarryConfig()
  env = GraspCarry2D(cfg)
  policy = ScriptedCarryPolicy(speed=speed, allow_regrasp=allow_regrasp,
                               config=cfg)
  rows = []
  for s in seeds:
    row = rollout(env, policy, int(s))
    # `rollout`이 담지 않는 파지 진단(첫 파지의 접촉 길이·지렛대)을 덧붙인다.
    row['contacts'] = list(policy.grasp_contacts)
    row['arms'] = list(policy.grasp_arms)
    rows.append(row)
  return rows


def stg_stats(rows: List[dict]) -> dict:
  """성공 에피소드만 모은 STG 통계 + 결과 분류."""
  n = len(rows)
  ok = np.array([r['steps'] for r in rows if r['outcome'] == 'success'],
                dtype=float)
  q = ((lambda p: float(np.quantile(ok, p))) if len(ok)
       else (lambda p: float('nan')))
  return {
      'n': n,
      'success': len(ok),
      'success_rate': len(ok) / n if n else 0.0,
      'mean': float(ok.mean()) if len(ok) else float('nan'),
      'median': q(0.5),
      'q80': q(0.8),
      'q95': q(0.95),
      'tipped': sum(1 for r in rows if r['outcome'] == 'tipped'),
      'timeout': sum(1 for r in rows if r['outcome'] == 'timeout'),
      'other': sum(1 for r in rows
                   if r['outcome'] not in ('success', 'tipped', 'timeout')),
      'drops': float(np.mean([r['n_drops'] for r in rows])) if n else 0.0,
      'regrasp_rate': (float(np.mean([r['regrasped'] for r in rows]))
                       if n else 0.0),
      # 낙하가 **성공 쪽 꼬리**(재시도해서 결국 성공)로 가는지, **실패**로
      # 빠져나가는지. 조건 B는 전자가 있어야만 성립한다 — 꼬리가 실패로
      # 나가면 성공 분포의 분위수는 기댓값을 따라갈 뿐이다.
      'drop_eps': sum(1 for r in rows if r['n_drops'] > 0),
      'drop_recovered': sum(1 for r in rows
                            if r['n_drops'] > 0 and r['outcome'] == 'success'),
  }


# ================================================================== 검증 1
def torque_margin(cfg: CarryConfig, contact: float, arm: float, mass: float,
                  mu: float, lead: float) -> float:
  """회전 토크 여유 = 파지의 회전 저항 상한 / 명목 하중 토크.

  분자는 `cfg.tip_torque_capacity(L, mu) = mu*N*L` — 접촉력 전부가 최대 지렛대
  `L`에서 작용하는 상한이다. 분모는 명목 리드 `lead`로 수평 가속할 때 지렛대
  `arm`에 걸리는 모멘트 `m * (k_p*lead) * arm`이다. 1보다 크면 그 파지는 명목
  가속도를 원리적으로 버틴다.

  `arm`이 0이면(무게중심 높이에서 물었으면) 하중 모멘트가 없으므로 무한대다.
  은닉 물성(질량·마찰)의 **실제값**을 쓴다 — 정책이 아니라 분석이므로 허용된다.
  """
  load = mass * cfg.k_p * lead * arm
  if load <= 0.0:
    return float('inf')
  return cfg.tip_torque_capacity(contact, mu) / load


def check_grasp_split(rows: List[dict], cfg: CarryConfig) -> dict:
  """소스 박스 내폭(그리퍼 바깥폭 기준)으로 나눠 파지 품질을 비교한다."""
  lead = cfg.max_accel / cfg.k_p
  gow = cfg.gripper_outer_width
  groups: Dict[str, list] = {'narrow': [], 'wide': []}
  no_grasp = 0
  for r in rows:
    if not r['contacts']:
      no_grasp += 1
      continue
    # 첫 파지만 본다 — 재파지 이후의 파지는 박스 내폭이 아니라 정책의 선택
    # (블록을 어디로 옮겼는지)이 결정하므로 이 검증의 대상이 아니다.
    contact, arm = r['contacts'][0], r['arms'][0]
    rec = {
        'src_width': r['src_width'],
        'contact': contact,
        'arm': arm,
        'margin': torque_margin(cfg, contact, arm, r['mass'], r['friction'],
                                lead),
        # 정책이 쓰는 "얕다"의 정의: 유도된 안전 리드가 명목 리드에 못 미침.
        'shallow': bool(r['speeds'] and r['speeds'][0] < lead - 1e-9),
        'regrasped': r['regrasped'],
    }
    groups['narrow' if r['src_width'] < gow else 'wide'].append(rec)

  out = {'no_grasp': no_grasp, 'gripper_outer_width': gow, 'lead': lead}
  for name, recs in groups.items():
    if not recs:
      out[name] = None
      continue
    finite = [x['margin'] for x in recs if math.isfinite(x['margin'])]
    out[name] = {
        'n': len(recs),
        'src_width': float(np.mean([x['src_width'] for x in recs])),
        'contact': float(np.mean([x['contact'] for x in recs])),
        'arm': float(np.mean([x['arm'] for x in recs])),
        # 무한대(지렛대 0)가 섞이므로 평균이 아니라 중앙값으로 요약한다.
        'margin': float(np.median(finite)) if finite else float('inf'),
        'shallow_rate': float(np.mean([x['shallow'] for x in recs])),
        'regrasp_rate': float(np.mean([x['regrasped'] for x in recs])),
    }
  return out


def report_grasp_split(res: dict) -> dict:
  print('\n=== 검증 1: 얕은 파지 / 깊은 파지의 분리 '
        '(소스 박스 내폭 vs 그리퍼 바깥폭 %.0fmm) ==='
        % res['gripper_outer_width'])
  print('  접촉=첫 파지의 패드-블록 겹침(mm, 상한=패드 길이), '
        '지렛대=패드 중심~블록 무게중심(mm)')
  print('  토크여유=tip_torque_capacity / (m*k_p*%.1f*arm) 의 중앙값 '
        '(<1 이면 명목 가속도를 못 버팀)' % res['lead'])
  print('  %-6s %4s %8s %8s %8s %9s %9s %9s'
        % ('그룹', 'n', '내폭', '접촉', '지렛대', '토크여유', '얕은비율',
           '재파지율'))
  for name, label in (('narrow', '좁음'), ('wide', '넓음')):
    g = res[name]
    if g is None:
      print('  %-6s   -- (해당 에피소드 없음)' % label)
      continue
    print('  %-6s %4d %8.1f %8.1f %8.1f %9.2f %8.1f%% %8.1f%%'
          % (label, g['n'], g['src_width'], g['contact'], g['arm'],
             g['margin'], 100 * g['shallow_rate'], 100 * g['regrasp_rate']))
  if res['no_grasp']:
    print('  (파지에 도달하지 못한 에피소드 %d개는 제외)' % res['no_grasp'])

  nar, wid = res['narrow'], res['wide']
  verdict = {'contact_ok': False, 'margin_ok': False, 'ok': False,
             'narrow': nar, 'wide': wid}
  if nar and wid:
    verdict['contact_ok'] = nar['contact'] < wid['contact']
    verdict['margin_ok'] = nar['margin'] < wid['margin']
    verdict['ok'] = verdict['contact_ok'] and verdict['margin_ok']
    print('  -> 접촉 %.1f vs %.1f mm (%s), 토크여유 %.2f vs %.2f (%s) => 분리 %s'
          % (nar['contact'], wid['contact'],
             'O' if verdict['contact_ok'] else 'X',
             nar['margin'], wid['margin'],
             'O' if verdict['margin_ok'] else 'X',
             'O' if verdict['ok'] else 'X'))
    if not verdict['contact_ok'] and verdict['margin_ok']:
      # 얕게 물어도 패드(30mm)는 블록 옆면(높이 100~140mm) 안에 통째로 들어가
      # 겹침이 포화한다. 파지 깊이는 접촉 길이가 아니라 **지렛대**로 드러난다.
      print('     (접촉 길이는 패드 길이에서 포화한다 — 얕게 물어도 패드가 '
            '블록 옆면 안에 다 들어간다.')
      print('      파지 깊이는 지렛대로 드러나며(%.1f vs %.1f mm), '
            '난이도 분리는 그쪽에서 성립한다)'
            % (nar['arm'], wid['arm']))
  return verdict


# ================================================================== 검증 2
def report_regrasp(on: dict, off: dict) -> dict:
  print('\n=== 검증 2: 재파지의 효용 (같은 시드 집합, 명목 속도) ===')
  print('  %-11s %8s %8s %9s %6s %9s %9s %8s'
        % ('설정', '성공률', '넘어짐', '타임아웃', '기타', 'STG평균',
           'STG중앙', 'q0.8'))
  for label, s in (('재파지 ON', on), ('재파지 OFF', off)):
    print('  %-11s %7.1f%% %7.1f%% %8.1f%% %5.1f%% %9.1f %9.1f %8.1f'
          % (label, 100 * s['success_rate'], 100 * s['tipped'] / s['n'],
             100 * s['timeout'] / s['n'], 100 * s['other'] / s['n'],
             s['mean'], s['median'], s['q80']))
  print('  (재파지 ON의 실제 재파지 발생률 %.1f%%)' % (100 * on['regrasp_rate']))
  d_succ = 100 * (on['success_rate'] - off['success_rate'])
  d_step = on['mean'] - off['mean']
  ok = d_succ > 0.0 and d_step > 0.0
  print('  -> 재파지 성공률 %+.1f%%p, 평균 스텝 %+.1f => 트레이드오프 %s'
        % (d_succ, d_step, 'O' if ok else 'X'))
  if not ok:
    why = ('성공률이 오르지 않는다' if d_succ <= 0.0
           else '시간 비용이 없다(평균 스텝이 늘지 않는다)')
    print('     (%s — "느리지만 안전"이 성립하지 않는다)' % why)
  return {'ok': ok, 'd_success_pp': d_succ, 'd_mean_steps': d_step,
          'on': on, 'off': off}


# ================================================================== 검증 3
def condition_pairs(table: List[dict]) -> dict:
  """(빠른 f, 느린 s) 쌍마다 조건 A/B를 판정한다."""
  pairs_a, pairs_b, pairs_both, n_pairs = [], [], [], 0
  for f in table:
    for s in table:
      if f['speed'] <= s['speed']:
        continue
      n_pairs += 1
      sf, ss = f['stats'], s['stats']
      if not (math.isfinite(sf['mean']) and math.isfinite(ss['mean'])):
        continue
      faster_mean = sf['mean'] < ss['mean']
      cond_a = faster_mean and sf['success_rate'] < ss['success_rate']
      cond_b = faster_mean and sf['q80'] > ss['q80']
      rec = {
          'f': f['speed'], 's': s['speed'],
          'dmean': sf['mean'] - ss['mean'],
          'dq80': sf['q80'] - ss['q80'],
          'dsucc_pp': 100 * (sf['success_rate'] - ss['success_rate']),
      }
      if cond_a:
        pairs_a.append(rec)
      if cond_b:
        pairs_b.append(rec)
      if cond_a and cond_b:
        pairs_both.append(rec)
  return {'pairs_a': pairs_a, 'pairs_b': pairs_b, 'pairs_both': pairs_both,
          'n_pairs': n_pairs, 'table': table}


def report_speed_sweep(table: List[dict]) -> dict:
  print('\n=== 검증 3: 속도-위험 트레이드오프 (STG는 성공 에피소드만) ===')
  print('  %7s %8s %9s %9s %8s %8s %8s %9s %7s %12s'
        % ('리드', '성공률', 'STG평균', 'STG중앙', 'q0.8', 'q0.95',
           '넘어짐', '타임아웃', '낙하', '낙하후성공'))
  for e in table:
    s = e['stats']
    print('  %7.1f %7.1f%% %9.1f %9.1f %8.1f %8.1f %7.1f%% %8.1f%% %7.2f %11s'
          % (e['speed'], 100 * s['success_rate'], s['mean'], s['median'],
             s['q80'], s['q95'], 100 * s['tipped'] / s['n'],
             100 * s['timeout'] / s['n'], s['drops'],
             '%d/%d' % (s['drop_recovered'], s['drop_eps'])))
  print('  낙하후성공 = 낙하가 있었던 에피소드 중 그래도 성공한 수 — '
        '조건 B가 요구하는 **성공 쪽 꼬리**의 원천이다.')

  res = condition_pairs(table)
  print('\n  검사한 (빠른 f, 느린 s) 쌍: %d' % res['n_pairs'])
  _print_pairs('조건 A (기댓값은 빠른 쪽 선호, 성공률은 느린 쪽 선호)',
               res['pairs_a'])
  _print_pairs('조건 B (기댓값은 빠른 쪽 선호, q0.8은 느린 쪽 선호)',
               res['pairs_b'])
  if res['pairs_both']:
    print('\n  *** 조건 A와 B를 **동시에** 만족하는 쌍 %d개 — '
          '이 결정 지점에서는 분위수가 기댓값과 다른 행동을 고른다 ***'
          % len(res['pairs_both']))
    _print_pairs('A+B 동시 만족', res['pairs_both'], indent=4)
  else:
    print('\n  조건 A와 B를 **동시에** 만족하는 쌍: 0')
  return res


def _print_pairs(title: str, pairs: List[dict], indent: int = 2) -> None:
  pad = ' ' * indent
  print('%s%s: %d쌍' % (pad, title, len(pairs)))
  for p in pairs:
    print('%s  f=%.1f vs s=%.1f : dmean %+.1f, dq0.8 %+.1f, 성공률 %+.1f%%p'
          % (pad, p['f'], p['s'], p['dmean'], p['dq80'], p['dsucc_pp']))


# ==================================================================== main
def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=120)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--speeds', type=float, nargs='+',
                  default=list(DEFAULT_SPEEDS),
                  help='검증 3의 명령 리드 목록(mm/스텝).')
  args = ap.parse_args(argv)

  cfg = CarryConfig()
  seeds = list(range(args.seed0, args.seed0 + args.episodes))
  lead = cfg.max_accel / cfg.k_p

  print('env=GraspCarry2D  n=%d/설정  seeds=[%d,%d)'
        % (len(seeds), seeds[0], seeds[-1] + 1))
  print('config: 명목 리드 %.1fmm (max_accel %.0f / k_p %.0f), '
        '그리퍼 바깥폭 %.0fmm, 소스 내폭 %.1f~%.1fmm, 질량 %.3f~%.3fkg, '
        'mu %.2f~%.2f, max_steps %d'
        % (lead, cfg.max_accel, cfg.k_p, cfg.gripper_outer_width,
           cfg.src_box_width_range[0], cfg.src_box_width_range[1],
           cfg.object_mass_range[0], cfg.object_mass_range[1],
           cfg.object_friction_range[0], cfg.object_friction_range[1],
           cfg.max_steps))

  # 검증 3의 스윕을 먼저 돌고, 그중 명목 속도 실행을 검증 1/2가 재사용한다.
  sweep = []
  for sp in args.speeds:
    rows = run_config(seeds, speed=sp, cfg=cfg)
    sweep.append({'speed': sp, 'rows': rows, 'stats': stg_stats(rows)})

  # 검증 1/2는 **명목 리드**에서만 의미가 있다(파지 품질과 재파지 판단의 기준이
  # 명목 리드다). 스윕에 명목 리드가 들어 있으면 재사용하고, 아니면 따로 돌린다 —
  # `--speeds`로 다른 그리드를 줘도 검증 1/2의 조건은 바뀌지 않게 한다.
  nominal = next((e for e in sweep if abs(e['speed'] - lead) < 1e-9), None)
  if nominal is None:
    rows = run_config(seeds, speed=lead, cfg=cfg)
    nominal = {'speed': lead, 'rows': rows, 'stats': stg_stats(rows)}
  v1 = report_grasp_split(check_grasp_split(nominal['rows'], cfg))

  off_rows = run_config(seeds, speed=nominal['speed'], allow_regrasp=False,
                        cfg=cfg)
  v2 = report_regrasp(nominal['stats'], stg_stats(off_rows))

  v3 = report_speed_sweep(sweep)

  # ------------------------------------------------------------------ 요약
  print('\n' + '=' * 74)
  print('env=GraspCarry2D  n=%d/설정  명목 리드 %.1fmm, 그리퍼 바깥폭 %.0fmm, '
        '소스 내폭 %.1f~%.1fmm'
        % (len(seeds), lead, cfg.gripper_outer_width,
           cfg.src_box_width_range[0], cfg.src_box_width_range[1]))
  if v1['narrow'] and v1['wide']:
    print('검증1: 좁은 박스 접촉 %.1f mm / 토크여유 %.2f vs '
          '넓은 박스 접촉 %.1f mm / 토크여유 %.2f  -> 분리 %s'
          % (v1['narrow']['contact'], v1['narrow']['margin'],
             v1['wide']['contact'], v1['wide']['margin'],
             'O' if v1['ok'] else 'X'))
  else:
    print('검증1: 한쪽 그룹의 에피소드가 없어 비교 불가 -> 분리 X')
  print('검증2: 재파지 성공률 %+.1f%%p, 평균 스텝 %+.1f  -> 트레이드오프 %s'
        % (v2['d_success_pp'], v2['d_mean_steps'], 'O' if v2['ok'] else 'X'))
  print('검증3: 조건A 만족 %d쌍, 조건B 만족 %d쌍, 동시 만족 %d쌍 (검사 %d쌍)'
        % (len(v3['pairs_a']), len(v3['pairs_b']), len(v3['pairs_both']),
           v3['n_pairs']))
  print('=' * 74)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
