"""인간 데모 vs 정책 롤아웃 STG 비교 (소스 구분 가설 검증).

두 데이터셋의 STG 라벨을 (학습 없이) 직접 비교한다:
  1. 주변 분포: 성공까지 스텝 수(에피소드 길이) 분포 + 꼬리 지표
  2. 상태 매칭 gap: 관측 공간에서 최근접 이웃으로 같은 상태를 찾아
     gap(o) = T_policy(o) - T_human(o) 를 추정
  3. gap의 상태 종속성: gap이 특정 상태에 몰리는지(구조적) 아니면
     상태와 무관하게 균일한지(단순 스케일 차이)

왜 최근접 이웃인가: 두 데이터셋은 시작 시드가 달라 같은 상태가 정확히
겹치지 않는다. 예측기 두 개를 학습하기 전에, 라벨만으로 gap의 존재/구조를
값싸게 확인하는 단계다.
"""
import argparse
import pickle

import numpy as np


def load(path):
  d = pickle.load(open(path, 'rb'))
  obs = np.concatenate([d['observation']['agent_pos'],
                        d['observation']['env_state']], axis=-1)
  return dict(obs=obs, ttg=d['time_to_success'], eid=d['episode_id'],
              cov=d['coverage'], meta=d['meta'])


def ep_lengths(d):
  return np.array([int((d['eid'] == e).sum()) for e in np.unique(d['eid'])])


def tail_stats(x, name):
  q = np.percentile(x, [50, 75, 90, 95, 99])
  print(f'  {name:<10} n={len(x):>5}  평균={x.mean():>6.1f}  '
        f'중앙={q[0]:>5.1f}  p75={q[1]:>5.1f}  p90={q[2]:>5.1f}  '
        f'p95={q[3]:>5.1f}  p99={q[4]:>5.1f}  최대={x.max():>5.0f}')
  return q


def marginal(h, p):
  print('\n[1] 주변 분포: 성공까지 스텝 수')
  hl, pl = ep_lengths(h), ep_lengths(p)
  qh = tail_stats(hl, '인간')
  qp = tail_stats(pl, '정책')
  print(f'  -> 중앙 차이 {qp[0]-qh[0]:+.1f},  p90 차이 {qp[2]-qh[2]:+.1f},  '
        f'p99 차이 {qp[4]-qh[4]:+.1f}')
  print('     (중앙은 비슷한데 p90/p99만 크면 = 꼬리에서만 못한다는 뜻)')
  # 꼬리 두께 지표: p90/중앙 비율
  print(f'  꼬리 비대칭(p90/중앙): 인간 {qh[2]/qh[0]:.2f}  정책 {qp[2]/qp[0]:.2f}')


def matched_gap(h, p, k=5, n_query=4000, seed=0):
  """정책 상태들을 질의로, 인간 데이터에서 kNN 이웃의 STG 중앙값과 비교."""
  print(f'\n[2] 상태 매칭 gap (kNN k={k}, 질의 {n_query}개)')
  rng = np.random.default_rng(seed)
  # 관측 스케일 정규화(픽셀 좌표 18차원) — 인간+정책 합쳐 표준화
  allo = np.concatenate([h['obs'], p['obs']], 0)
  mu, sd = allo.mean(0), allo.std(0) + 1e-8
  H = (h['obs'] - mu) / sd
  P = (p['obs'] - mu) / sd

  idx = rng.choice(len(P), size=min(n_query, len(P)), replace=False)
  Q = P[idx]
  q_ttg = p['ttg'][idx]

  # 청크 단위 유클리드 kNN (메모리 절약)
  gaps, dists, h_ttgs = [], [], []
  CH = 256
  for s in range(0, len(Q), CH):
    q = Q[s:s + CH]
    d2 = ((q[:, None, :] - H[None, :, :]) ** 2).sum(-1)   # (chunk, N_h)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    nd = np.take_along_axis(d2, nn, 1)
    h_t = h['ttg'][nn]                                     # (chunk, k)
    h_med = np.median(h_t, axis=1)
    gaps.append(q_ttg[s:s + CH] - h_med)
    dists.append(np.sqrt(nd.mean(1)))
    h_ttgs.append(h_med)
  gap = np.concatenate(gaps)
  dist = np.concatenate(dists)
  h_med = np.concatenate(h_ttgs)

  # 매칭 품질이 나쁜(먼) 질의는 제외 — 외삽 gap은 허구
  thr = np.percentile(dist, 50)
  m = dist <= thr
  print(f'  매칭 거리 중앙 {np.median(dist):.2f} (표준화 단위), '
        f'가까운 절반({m.sum()}개)만 사용')
  g = gap[m]
  print(f'  gap = T_policy - T_human_NN:  평균 {g.mean():+.1f}  '
        f'중앙 {np.median(g):+.1f}  표준편차 {g.std():.1f}')
  print(f'  gap>0 비율 {float((g>0).mean()):.1%}  '
        f'(0.5면 차이 없음, 클수록 정책이 느림)')
  print(f'  gap 분위: p10 {np.percentile(g,10):+.0f}  p25 {np.percentile(g,25):+.0f}  '
        f'p75 {np.percentile(g,75):+.0f}  p90 {np.percentile(g,90):+.0f}')
  return g, h_med[m], q_ttg[m], dist[m]


def gap_structure(g, h_med):
  """gap이 상태(=남은 거리)에 따라 구조적으로 다른가."""
  print('\n[3] gap의 구조: 인간 기준 남은 스텝(=태스크 진행도) 구간별')
  edges = [0, 10, 20, 40, 60, 90, 130, 300]
  print(f'  {"인간STG 구간":>14} {"n":>6} {"gap 평균":>9} {"gap 중앙":>9} {"gap>0":>7}')
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (h_med >= lo) & (h_med < hi)
    if m.sum() < 20:
      continue
    gg = g[m]
    print(f'  {lo:5d}~{hi:<6d} {int(m.sum()):>6d} {gg.mean():>+9.1f} '
          f'{np.median(gg):>+9.1f} {float((gg>0).mean()):>7.1%}')
  print('  해석: 구간마다 gap이 크게 다르면 = 상태 종속적(구조적 무능력).')
  print('        구간에 무관하게 일정하면 = 단순 스케일 차이(정보 적음).')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--human', default='data/pusht_demos.pkl')
  ap.add_argument('--policy', default='data/pusht_dp_rollouts.pkl')
  ap.add_argument('--k', type=int, default=5)
  ap.add_argument('--n-query', type=int, default=4000)
  args = ap.parse_args()

  h, p = load(args.human), load(args.policy)
  print(f'인간: {args.human}  ({len(np.unique(h["eid"]))}ep, {len(h["ttg"])}스텝)')
  print(f'정책: {args.policy}  ({len(np.unique(p["eid"]))}ep, {len(p["ttg"])}스텝)')
  print(f'정책 meta: {p["meta"].get("policy")}')

  marginal(h, p)
  g, h_med, q_ttg, dist = matched_gap(h, p, k=args.k, n_query=args.n_query)
  gap_structure(g, h_med)


if __name__ == '__main__':
  main()
