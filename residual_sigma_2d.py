"""잔차 실험 (references/context_5.md): 2D 장애물 환경에서 거리(μ) 효과를
제거한 뒤에도 σ가 예측기의 자기 오차를 추가로 설명하는지 검정.

절차:
  1. tangent 컨트롤러(학습 데이터 생성 정책)로 성공 궤적 다수 수집.
  2. 각 스텝 t에서 (μ_t, σ_t, T_true_t=남은 실제 스텝) 기록.
  3. μ를 구간(bin)으로 나눠 거리 효과 제거.
  4. 각 구간 안에서 corr(σ, e), e=|μ - T_true| 계산.

읽는 법: 구간 내 corr>0 이면 '거리가 같을 때 σ 큰 지점에서 실제로 더 틀린다'
= 예측기가 자기 오차를 안다. |corr|<0.1 신호없음, 0.1~0.3 약함, >0.3 확실.
"""
import numpy as np
from scipy import stats

from src.obstacle_env import ObstacleAvoidPoint2D, demo_action
from src.probe_generic import GenericSTGProbe

CKPT = 'checkpoints/obstacle_clean/predictor.pkl'
N_EPISODES = 400
MAX_STEPS = 300


def collect():
  probe = GenericSTGProbe(CKPT)
  env = ObstacleAvoidPoint2D()
  mus, sigmas, Ttrue = [], [], []
  n_succ = 0
  for ep in range(N_EPISODES):
    np.random.seed(ep)
    ts = env.reset()
    obs = ts.observation
    ep_mu, ep_sig = [], []
    step = 0
    while (not env.success()) and step < MAX_STEPS:
      rec = probe.query(obs)
      ep_mu.append(rec.expectation)
      ep_sig.append(np.sqrt(max(rec.variance, 0.0)))
      ts = env.step(np.asarray(demo_action(obs), dtype=np.float32))
      obs = ts.observation
      step += 1
    if not env.success():
      continue                       # 실패 에피소드는 T_true 정의 불가 -> 제외
    n_succ += 1
    T = len(ep_mu)
    for t in range(T):
      mus.append(ep_mu[t])
      sigmas.append(ep_sig[t])
      Ttrue.append(T - t)            # 그 시점에서 실제로 남은 스텝 수
  print(f'성공 에피소드 {n_succ}/{N_EPISODES}, 총 {len(mus)}개 시점')
  return (np.array(mus), np.array(sigmas), np.array(Ttrue))


def main():
  mu, sig, Tt = collect()
  err = np.abs(mu - Tt)

  # 전역 상관 (거리 미제거) — 혼동 확인용
  r_global = stats.pearsonr(sig, err)[0]
  print(f'\n[거리 미제거] corr(sigma, error) 전역 = {r_global:+.3f}')

  # μ 구간별 (거리 제거)
  edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 500]
  print(f'\n[거리 제거: μ 구간별] corr(sigma, error)')
  print(f'{"μ 구간":>12} {"n":>6} {"σ 평균":>8} {"σ 표준":>8} {"corr":>8} {"p":>8}')
  rows = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mu >= lo) & (mu < hi)
    n = int(m.sum())
    if n < 30:
      continue
    s = sig[m]
    e = err[m]
    if s.std() < 1e-9:
      r, p = 0.0, 1.0
    else:
      r, p = stats.pearsonr(s, e)
    rows.append((n, r))
    print(f'{lo:5d}~{hi:<5d} {n:>6d} {s.mean():>8.2f} {s.std():>8.2f} '
          f'{r:>+8.3f} {p:>8.1e}')

  # n-가중 평균 구간 내 상관
  ns = np.array([r[0] for r in rows], float)
  rs = np.array([r[1] for r in rows], float)
  wavg = float(np.sum(ns * rs) / ns.sum())
  print(f'\nn-가중 구간내 평균 corr(σ,e) = {wavg:+.3f}')

  # --- 편상관: 구간 내 μ를 통제해도 σ가 e를 설명하나? (μ 누수 검정) ---
  # partial corr(σ,e | μ) = corr(resid(σ~μ), resid(e~μ)) 구간 안에서
  print(f'\n[μ 누수 검정] 구간 내 μ 통제 후 편상관 partial-corr(σ,e|μ)')
  print(f'{"μ 구간":>12} {"n":>6} {"corr(μ,e)":>10} {"partial(σ,e|μ)":>15}')

  def _resid(y, x):                    # y를 x에 선형회귀한 잔차
    b = np.polyfit(x, y, 1)
    return y - (b[0] * x + b[1])

  prows = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mu >= lo) & (mu < hi)
    n = int(m.sum())
    if n < 30:
      continue
    s, e, u = sig[m], err[m], mu[m]
    if s.std() < 1e-9 or u.std() < 1e-9:
      continue
    r_mu_e = stats.pearsonr(u, e)[0]
    rs_ = _resid(s, u)
    re_ = _resid(e, u)
    pc = 0.0 if rs_.std() < 1e-9 else stats.pearsonr(rs_, re_)[0]
    prows.append((n, pc))
    print(f'{lo:5d}~{hi:<5d} {n:>6d} {r_mu_e:>+10.3f} {pc:>+15.3f}')

  pn = np.array([r[0] for r in prows], float)
  pr = np.array([r[1] for r in prows], float)
  pwavg = float(np.sum(pn * pr) / pn.sum())
  print(f'\nn-가중 편상관 partial-corr(σ,e|μ) = {pwavg:+.3f}')
  print('해석: |corr|<0.1 신호없음 / 0.1~0.3 약함 / >0.3 확실')
  print('  단순 corr은 유지되나 편상관이 붕괴하면 -> σ는 구간내 μ 프록시일 뿐')


if __name__ == '__main__':
  main()
