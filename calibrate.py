"""Phase 0 캘리브레이션 (references/context_7.md): GraspAngleTransport2D의
파라미터 영역 탐색.

목적:
  (A) 스펙 조건 — 기댓값(평균 완료 스텝) 순위와 성공률 순위가 반대인 영역:
        mean_steps(push)  < mean_steps(realign)   (기댓값은 강행이 유리)
        success_rate(push)< success_rate(realign) (성공률은 재정렬이 유리)
  (B) 진짜 가설 — 결정 지점에서 mean 순위와 0.8-분위수 순위가 뒤집히는가:
        mean(STG_push)   < mean(STG_realign)      (기댓값: push 선호)
        q0.8(STG_push)   > q0.8(STG_realign)      (분위수: realign 선호)  <- 부호전환
      STG 분포는 논문대로 '성공으로 끝난 에피소드'만 모은다.

신경망/학습 없음. 순수 동역학 시뮬 + 고정 스크립트 정책 2개. 억지로 숫자를
맞추지 않는다 — 조건을 만족하는 조합이 없으면 그대로 보고한다.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Config:
  # --- 결과 분기(스펙 그리드 대상) ---
  theta_max: float = 0.5     # 파지각 ~ U(-theta_max, theta_max)
  s: float = 0.3             # 성공확률 sharpness: P(theta)=exp(-theta^2/2s^2)
  p_recover: float = 0.5     # 낙하 시 회수 가능 확률(나머지는 영구 실패)
  # --- 스텝 비용(기하에서 유도, Phase 0에선 스텝 수로 고정) ---
  transport_steps: int = 12  # D: 파지 지점 -> 목표 이동
  realign_steps: int = 3     # k: 재정렬 정지 비용
  regrasp_steps: int = 4     # 낙하물 회수(되돌아가기+재파지) 비용
  T_max: int = 200


def simulate(cfg: Config, policy: str, rng: np.random.Generator):
  """한 에피소드. 반환: (outcome, total_steps). outcome in {success, failure}.

  push   : 재정렬 없이 현재 theta로 삽입 시도.
  realign: 매 파지마다 재정렬(k스텝) 후 theta=0으로 삽입(항상 성공).
  낙하 시: 확률 p_recover면 회수(regrasp 비용 + theta 재추첨) 후 재시도,
           아니면 영구 실패.
  """
  steps = 0
  theta = rng.uniform(-cfg.theta_max, cfg.theta_max)
  while steps <= cfg.T_max:
    if policy == 'realign':
      steps += cfg.realign_steps
      theta = 0.0
    steps += cfg.transport_steps
    if steps > cfg.T_max:
      return 'failure', steps                       # 타임아웃
    p_succ = np.exp(-theta * theta / (2.0 * cfg.s * cfg.s))
    if rng.random() < p_succ:
      return 'success', steps
    # 삽입 실패 -> 낙하
    if rng.random() < cfg.p_recover:
      steps += cfg.regrasp_steps
      theta = rng.uniform(-cfg.theta_max, cfg.theta_max)
      continue                                       # 재시도
    return 'failure', steps                          # 영구 실패
  return 'failure', steps


def rollout(cfg: Config, policy: str, n: int, seed: int):
  rng = np.random.default_rng(seed)
  outs, stps = [], []
  for _ in range(n):
    o, s = simulate(cfg, policy, rng)
    outs_ok = (o == 'success')
    outs.append(outs_ok)
    if outs_ok:
      stps.append(s)                                 # 성공 에피소드만 STG로
  outs = np.array(outs)
  stps = np.array(stps, dtype=float)
  return outs, stps


def summarize(cfg: Config, n=2000, seed=0):
  """두 정책의 (성공률, 성공에피소드 평균스텝, STG mean, STG q0.8)."""
  res = {}
  for i, pol in enumerate(('push', 'realign')):
    ok, stg = rollout(cfg, pol, n, seed + i)
    res[pol] = dict(
        succ=float(ok.mean()),
        mean_steps=float(stg.mean()) if len(stg) else float('nan'),
        stg_mean=float(stg.mean()) if len(stg) else float('nan'),
        stg_q80=float(np.quantile(stg, 0.8)) if len(stg) else float('nan'),
    )
  return res


def main():
  N = 2000
  theta_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  s_grid = [0.2, 0.3, 0.4, 0.5]
  prec_grid = [0.3, 0.5, 0.7, 0.9]

  print(f'env=GraspAngleTransport2D(sim)  n={N}/policy/combo  '
        f'D={Config.transport_steps} k={Config.realign_steps} '
        f'regrasp={Config.regrasp_steps} T_max={Config.T_max}')
  print('\n조건A: mean_steps(push)<mean_steps(realign)  AND  succ(push)<succ(realign)')
  print('조건B(핵심): mean(STG_push)<mean(STG_realign)  AND  q0.8(STG_push)>q0.8(STG_realign)\n')

  hdr = (f'{"theta":>5} {"s":>4} {"prec":>4} | '
         f'{"push:step/succ":>15} {"realign:step/succ":>17} | '
         f'{"A":>2} | {"pushμ/q80":>12} {"realignμ/q80":>13} {"B":>2}')
  print(hdr)
  print('-' * len(hdr))

  passA, passB = [], []
  for th in theta_grid:
    for s in s_grid:
      for pr in prec_grid:
        cfg = Config(theta_max=th, s=s, p_recover=pr)
        r = summarize(cfg, n=N, seed=0)
        p, a = r['push'], r['realign']
        condA = (p['mean_steps'] < a['mean_steps']) and (p['succ'] < a['succ'])
        condB = (p['stg_mean'] < a['stg_mean']) and (p['stg_q80'] > a['stg_q80'])
        if condA:
          passA.append((cfg, r))
        if condB:
          passB.append((cfg, r))
        markA = 'A' if condA else '.'
        markB = 'B' if condB else '.'
        print(f'{th:>5.1f} {s:>4.1f} {pr:>4.1f} | '
              f'{p["mean_steps"]:>7.1f}/{p["succ"]:>6.1%} '
              f'{a["mean_steps"]:>8.1f}/{a["succ"]:>7.1%} | '
              f'{markA:>2} | '
              f'{p["stg_mean"]:>5.1f}/{p["stg_q80"]:>5.1f} '
              f'{a["stg_mean"]:>6.1f}/{a["stg_q80"]:>5.1f} {markB:>2}')

  print(f'\n조건A 만족 조합: {len(passA)}개 / {len(theta_grid)*len(s_grid)*len(prec_grid)}')
  print(f'조건B(부호전환) 만족 조합: {len(passB)}개')

  # 추천: 조건B를 만족하면서 목표치(realign~15/98%, push~12/90%)에 가까운 것
  target_ps, target_pr = 12.0, 0.90
  target_as, target_ar = 15.0, 0.98
  pool = passB if passB else passA
  if not pool:
    print('\n[결과] 조건B(및 A)를 만족하는 조합이 없다. 환경 규칙을 바꾸지 않고 '
          '여기서 멈춘다. 스텝비용 구조(D/k/regrasp)를 재검토해야 할 수 있다.')
    return
  def dist(item):
    cfg, r = item
    return (abs(r['push']['mean_steps'] - target_ps) / target_ps
            + abs(r['push']['succ'] - target_pr)
            + abs(r['realign']['mean_steps'] - target_as) / target_as
            + abs(r['realign']['succ'] - target_ar))
  best = min(pool, key=dist)
  cfg, r = best
  print(f'\n[추천 조합] ({"조건B 통과" if passB else "조건A만 통과"})')
  print(f'  config: {asdict(cfg)}')
  print(f'  push   : mean_steps={r["push"]["mean_steps"]:.1f}  '
        f'succ={r["push"]["succ"]:.1%}  STG mean={r["push"]["stg_mean"]:.1f}  '
        f'q0.8={r["push"]["stg_q80"]:.1f}')
  print(f'  realign: mean_steps={r["realign"]["mean_steps"]:.1f}  '
        f'succ={r["realign"]["succ"]:.1%}  STG mean={r["realign"]["stg_mean"]:.1f}  '
        f'q0.8={r["realign"]["stg_q80"]:.1f}')
  dm = r['push']['stg_mean'] - r['realign']['stg_mean']
  dq = r['push']['stg_q80'] - r['realign']['stg_q80']
  print(f'  -> mean 차(push-realign)={dm:+.1f} (음수면 기댓값은 push 선호)')
  print(f'  -> q0.8 차(push-realign)={dq:+.1f} (양수면 분위수는 realign 선호)')
  if dm < 0 < dq:
    print('  ==> 부호 전환 확인: 같은 결정 지점에서 mean과 q0.8이 반대 행동을 '
          '선호한다. γ로는 재현 불가한 순위 역전의 실증 근거.')


if __name__ == '__main__':
  main()
