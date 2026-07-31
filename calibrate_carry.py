"""Phase 0 캘리브레이션: 2D 블록 운반(GraspCarry2D) 파라미터 영역 탐색.

물리 없이 **슬립 모델만** 시뮬레이션한다. 목적은 환경을 짓기 전에
"공격적으로 빨리 옮기기 vs 조심스럽게 천천히 옮기기"가 의미 있는 트레이드오프를
만드는 파라미터 영역이 실제로 존재하는지 확인하는 것.

슬립 모델 (접촉/전류 센서 없음, 그리퍼 최대 토크 고정):
    법선력   F_n = tau_max / r_finger            (고정 상수)
    슬립 조건 m*(g + a) > mu*F_n
    => 가속도 예산  a_crit = mu*F_n/m - g        <- 블록마다 다른 **은닉** 값
  에이전트는 a_crit을 볼 수 없다(질량·마찰 미관측). 시도해봐야 안다.

운반 시간: 거리 D를 가속도 a로 (가속-감속) => t(a) = 2*sqrt(D/a).
  a가 크면 빠르지만 a > a_crit이면 미끄러져 낙하한다.

두 고정 정책:
  aggressive  : a_max로 시작, 떨어뜨리면 a를 줄여 재시도
  conservative: 처음부터 낮은 a로 운반

찾는 조건:
  A) 기댓값 순위와 성공률 순위가 반대
       mean_steps(aggressive) < mean_steps(conservative)
       success(aggressive)    < success(conservative)
  B) (핵심) 같은 결정 지점에서 mean과 0.8분위수가 반대 행동을 선호
       mean(STG_agg) < mean(STG_con)   AND   q0.8(STG_agg) > q0.8(STG_con)
     STG는 논문과 동일하게 **성공 에피소드만** 모은다.

조건을 만족하는 조합이 없으면 그대로 보고한다(억지로 규칙을 바꾸지 않는다).
"""
from dataclasses import dataclass, asdict

import numpy as np

G = 9.81


@dataclass
class Config:
  # --- 은닉 물성: 가속도 예산 a_crit = mu*F_n/m - g ---
  # 캔 비유: 크기는 보이지만 채움 정도(fill)와 표면 물기(mu)는 안 보인다.
  ac_mean: float = 6.0       # a_crit 평균 (m/s^2). 절대 스케일은 무의미하고
  ac_spread: float = 0.8     # a_crit ~ U(mean*(1-spread), mean*(1+spread))
  # --- 정책이 고르는 가속도: a_crit 평균 대비 **비율**로 지정 ---
  # (절대값으로 두면 a_max가 a_crit 분포 밖으로 나가 '공격적 정책이 유리한
  #  구간'이 그리드에서 통째로 빠지는 일이 생긴다 -- 1차 스윕의 실패 원인)
  k_max: float = 0.9         # aggressive: a = k_max * ac_mean
  k_safe: float = 0.35       # conservative: a = k_safe * ac_mean
  a_reduce: float = 0.6      # 낙하 후 재시도 시 가속도 배율
  # --- 스텝 비용 (제어 주기 단위) ---
  D: float = 40.0            # 운반 거리 (시간 상수 t=2*sqrt(D/a)에 사용)
  grasp_steps: int = 6       # 접근 + 파지
  recover_steps: int = 10    # 낙하 후 재접근/재파지
  # --- 결과 분기 ---
  p_lost: float = 0.15       # 낙하 시 회수 불가(영구 실패) 확률
  T_max: int = 300


def transport_steps(cfg: Config, a: float) -> int:
  """거리 D를 가속도 a로 가속-감속 이동할 때의 스텝 수."""
  return max(1, int(np.ceil(2.0 * np.sqrt(cfg.D / max(a, 1e-6)))))


def simulate(cfg: Config, policy: str, rng: np.random.Generator):
  """한 에피소드. 반환: (성공여부, 총 스텝).

  블록의 a_crit은 에피소드 내내 고정(같은 물체)이므로, 낙하는 그 자체로
  'a가 너무 컸다'는 정보가 된다 -- 실패로 배우는 구조.
  """
  lo, hi = cfg.ac_mean * (1 - cfg.ac_spread), cfg.ac_mean * (1 + cfg.ac_spread)
  a_crit = rng.uniform(lo, hi)
  k = cfg.k_max if policy == 'aggressive' else cfg.k_safe
  a = k * cfg.ac_mean

  steps = cfg.grasp_steps
  while steps <= cfg.T_max:
    t = transport_steps(cfg, a)
    if a <= a_crit:                       # 슬립 없음 -> 운반 성공
      steps += t
      return (steps <= cfg.T_max), steps
    # 슬립: 운반 도중(절반쯤)에서 낙하한다고 본다
    steps += max(1, t // 2)
    if rng.random() < cfg.p_lost:
      return False, steps                 # 회수 불가 -> 영구 실패
    steps += cfg.recover_steps
    a *= cfg.a_reduce                     # 더 조심해서 재시도
  return False, steps                     # 타임아웃


def rollout(cfg: Config, policy: str, n: int, seed: int):
  rng = np.random.default_rng(seed)
  ok, stg = [], []
  for _ in range(n):
    s, steps = simulate(cfg, policy, rng)
    ok.append(s)
    if s:
      stg.append(steps)                   # 성공 에피소드만 STG로
  return np.array(ok), np.array(stg, float)


def summarize(cfg: Config, n=4000, seed=0):
  out = {}
  for i, pol in enumerate(('aggressive', 'conservative')):
    ok, stg = rollout(cfg, pol, n, seed + 991 * i)
    out[pol] = dict(
        succ=float(ok.mean()),
        mean_steps=float(stg.mean()) if len(stg) else float('nan'),
        q80=float(np.quantile(stg, 0.8)) if len(stg) else float('nan'),
        q95=float(np.quantile(stg, 0.95)) if len(stg) else float('nan'),
    )
  # 첫 시도 슬립 확률 P(a_crit < a) -- 메커니즘 확인용(해석적)
  lo, hi = cfg.ac_mean * (1 - cfg.ac_spread), cfg.ac_mean * (1 + cfg.ac_spread)
  out['p_slip_agg'] = float(np.clip((cfg.k_max * cfg.ac_mean - lo) / (hi - lo), 0, 1))
  return out


def main():
  N = 4000
  km_grid = [0.75, 0.9]                    # aggressive 가속도 / a_crit 평균
  spr_grid = [0.3, 0.5, 0.7]               # 물체 간 편차 (은닉 불확실성 크기)
  pl_grid = [0.1, 0.25]                    # 낙하 시 영구 실패 확률
  D_grid = [250.0]                         # 운반 거리 (속도 이득의 크기를 좌우)
  rec_grid = [4, 10, 20]                   # 재시도 비용 (속도 이득을 잠식)
  ks_grid = [0.25, 0.35]                   # conservative 가속도 비율

  print(f'env=GraspCarry2D(slip-model only)  n={N}/정책/조합  '
        f'ac_mean={Config.ac_mean} k_safe={Config.k_safe} '
        f'grasp={Config.grasp_steps} recover={Config.recover_steps} '
        f'a_reduce={Config.a_reduce}')
  print('\n조건A: mean_steps(agg)<mean_steps(con)  AND  succ(agg)<succ(con)')
  print('조건B: mean(STG_agg)<mean(STG_con)  AND  q0.8(STG_agg)>q0.8(STG_con)\n')

  hdr = (f'{"kmax":>5} {"ksafe":>6} {"spr":>4} {"plost":>5} {"rec":>4} '
         f'{"pslip":>6} | {"agg:step/succ":>14} {"con:step/succ":>15} | '
         f'{"dmean":>6} {"dq80":>6} {"dsucc":>6} | {"A":>2} {"B":>2}')
  print(hdr); print('-' * len(hdr))

  passA, passB = [], []
  for D in D_grid:
    for km in km_grid:
      for ks in ks_grid:
        for spr in spr_grid:
          for pl in pl_grid:
            for rec in rec_grid:
              cfg = Config(k_max=km, k_safe=ks, ac_spread=spr, p_lost=pl,
                           D=D, recover_steps=rec)
              r = summarize(cfg, n=N)
              a_, c_ = r['aggressive'], r['conservative']
              dmean = a_['mean_steps'] - c_['mean_steps']
              dq80 = a_['q80'] - c_['q80']
              dsucc = c_['succ'] - a_['succ']
              condA = (dmean < 0) and (dsucc > 0)
              condB = (dmean < 0) and (dq80 > 0)
              if condA:
                passA.append((cfg, r))
              if condB:
                passB.append((cfg, r))
              print(f'{km:>5.2f} {ks:>6.2f} {spr:>4.1f} {pl:>5.2f} {rec:>4d} '
                    f'{r["p_slip_agg"]:>6.1%} | '
                    f'{a_["mean_steps"]:>6.1f}/{a_["succ"]:>6.1%} '
                    f'{c_["mean_steps"]:>7.1f}/{c_["succ"]:>6.1%} | '
                    f'{dmean:>+6.1f} {dq80:>+6.1f} {dsucc:>+6.1%} | '
                    f'{"A" if condA else ".":>2} {"B" if condB else ".":>2}')

  total = (len(D_grid) * len(km_grid) * len(ks_grid) * len(spr_grid)
           * len(pl_grid) * len(rec_grid))
  print(f'\n조건A 만족: {len(passA)}/{total}')
  print(f'조건B(부호 전환) 만족: {len(passB)}/{total}')

  pool = passB or passA
  if not pool:
    print('\n[결과] 조건 A/B 모두 만족하는 조합 없음. 환경 규칙을 바꾸지 않고 보고한다.')
    return

  # 추천: 세 여유(mean 우위, q80 역전, 성공률 우위)가 **동시에** 큰 조합.
  # 하나라도 칼날 위면(margin~0) 실제 환경의 노이즈에 묻히므로 최소값을 본다.
  def score(item):
    cfg, r = item
    a_, c_ = r['aggressive'], r['conservative']
    m_mean = (c_['mean_steps'] - a_['mean_steps']) / c_['mean_steps']  # 클수록 좋음
    m_q80 = (a_['q80'] - c_['q80']) / c_['q80']
    m_succ = c_['succ'] - a_['succ']
    return min(m_mean, m_q80, m_succ)
  best = max(pool, key=score)
  cfg, r = best
  a_, c_ = r['aggressive'], r['conservative']
  print(f'\n[추천 조합] ({"조건B 통과" if passB else "조건A만 통과"})')
  print(f'  config: {asdict(cfg)}')
  print(f'  aggressive  : mean={a_["mean_steps"]:.1f}  q0.8={a_["q80"]:.1f}  '
        f'q0.95={a_["q95"]:.1f}  succ={a_["succ"]:.1%}')
  print(f'  conservative: mean={c_["mean_steps"]:.1f}  q0.8={c_["q80"]:.1f}  '
        f'q0.95={c_["q95"]:.1f}  succ={c_["succ"]:.1%}')
  dm = a_['mean_steps'] - c_['mean_steps']
  dq = a_['q80'] - c_['q80']
  print(f'  -> mean 차(agg-con) = {dm:+.1f}  (음수면 기댓값은 공격적 선호)')
  print(f'  -> q0.8 차(agg-con) = {dq:+.1f}  (양수면 분위수는 보수적 선호)')
  if dm < 0 < dq:
    print('  ==> 부호 전환 확인: 같은 상태에서 mean과 q0.8이 반대 행동을 고른다.')


if __name__ == '__main__':
  main()
