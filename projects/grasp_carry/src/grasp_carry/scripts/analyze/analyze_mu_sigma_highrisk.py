r"""고위험(μ 큰) 구간에서도 μ-σ가 붙어 있는지 검증 (references/context_8.md).

## 배경
이전에(약 한 달간) "기댓값(μ)과 분산(σ)이 여러 태스크에서 거의 같이 움직여서
분산이 추가 정보를 못 준다"는 결론을 얻었다(`experiments/2026-07-16_mu-sigma-relationship.md`
등). 그 결론은 **태스크 전체 평균 수준**에서 나온 것이다. 지금 검증하려는 renewal
theory 기반 "포기(조기 리셋) 판정" — μ(o) > B(=T_reset+T̂)면 포기 — 은 μ가 이미
나쁜(큰) 상태에서만 실제로 작동하는 결정이라, "고위험 구간에서만 놓고 봤을 때도
여전히 μ,σ가 붙어 있는지"가 알고 싶은 질문이다. 안 갈리면 평균 기준 포기판정으로
충분하고, 갈리면 CDF(분포 적분) 기준을 쓸 근거가 생긴다.

## 이 분석에 쓰는 예측기
`checkpoints/grasp_carry_dstg_deadline/predictor.pkl` — 성공 라벨은 논문 그대로,
실패 라벨은 "판정 순간=B(=reset_cost+T̂, 실측), 그 전=succ 예측기의 부트스트랩
평균"으로 학습한 obs-only STG 예측기(`src/train_carry_dstg.py --fail-mode deadline`
참고). fail_bin=None이라 `StgReward`가 그냥 평범한 카테고리컬로 다룬다.

## 데이터
`data/grasp_carry_demos_v3.pkl`의 **held-out(val) 에피소드만** 쓴다 — 예측기
학습 때 이미 본 transition에서 분석하면 과적합한 지점의 σ가 인위적으로 좁아
보일 수 있다(`src/carry_stg_reward._val_episode_ids`로 학습 때와 같은 분할을
재현 — seed는 dstg_deadline 학습에 쓴 기본값 0).

실행:
    python analyze_mu_sigma_highrisk.py
    python analyze_mu_sigma_highrisk.py --top-pct 10   # 고위험 정의를 상위 10%로
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from grasp_carry.carry_stg_reward import StgReward, _val_episode_ids

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_held_out(data_path: str, seed: int):
  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  val_eps = _val_episode_ids(data, seed)
  mask = np.isin(data['episode_id'], list(val_eps))
  return dict(
      obs=data['observation']['frame'][mask],
      episode_id=data['episode_id'][mask],
      is_success=np.asarray(data['is_success'])[mask],
      n_val_eps=len(val_eps),
  )


def corr_report(mu, sigma, label):
  pear = stats.pearsonr(mu, sigma)
  spear = stats.spearmanr(mu, sigma)
  print(f'  [{label}]  n={len(mu):5d}  '
        f'Pearson r={pear.statistic:+.3f} (p={pear.pvalue:.1e})  '
        f'Spearman ρ={spear.statistic:+.3f} (p={spear.pvalue:.1e})')
  return dict(n=len(mu), pearson_r=float(pear.statistic), pearson_p=float(pear.pvalue),
              spearman_r=float(spear.statistic), spearman_p=float(spear.pvalue))


def bin_sigma_stats(mu, sigma, n_bins):
  """μ를 등분위(quantile) 구간으로 나누고, 구간별 σ의 median/IQR을 낸다.
  등분위를 쓰는 이유: 등폭 구간은 표본이 몰린 구간과 희박한 구간의 표본수가
  극단적으로 갈려(특히 고위험 부분집합처럼 좁은 범위에서) bin당 표본이 너무
  적어질 수 있다."""
  edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
  edges[-1] += 1e-6   # 오른쪽 끝 포함
  bin_idx = np.clip(np.digitize(mu, edges[1:-1]), 0, n_bins - 1)
  rows = []
  for b in range(n_bins):
    sel = bin_idx == b
    if sel.sum() == 0:
      continue
    s = sigma[sel]
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    rows.append(dict(bin=b, mu_lo=float(edges[b]), mu_hi=float(edges[b + 1]),
                     n=int(sel.sum()), sigma_median=float(med),
                     sigma_iqr=float(q3 - q1), sigma_iqr_over_median=float((q3 - q1) / max(med, 1e-6))))
  return rows, bin_idx, edges


def find_example_pairs(mu, sigma, bin_idx, is_succ, n_bins, k=3):
  """같은 μ 구간 안에서 σ가 가장 크게 갈리는 표본 쌍을 몇 개 뽑는다."""
  pairs = []
  for b in range(n_bins):
    sel = np.where(bin_idx == b)[0]
    if len(sel) < 2:
      continue
    lo = sel[np.argmin(sigma[sel])]
    hi = sel[np.argmax(sigma[sel])]
    if sigma[hi] - sigma[lo] < 1e-6:
      continue
    pairs.append(dict(bin=b, mu_lo_idx=int(lo), mu_hi_idx=int(hi),
                      mu=float(mu[lo]), sigma_low=float(sigma[lo]), sigma_high=float(sigma[hi]),
                      mu_pair=float(mu[hi]), succ_low=bool(is_succ[lo]), succ_high=bool(is_succ[hi]),
                      sigma_gap=float(sigma[hi] - sigma[lo])))
  pairs.sort(key=lambda r: -r['sigma_gap'])
  return pairs[:k]


def make_plots(mu, sigma, high_risk_mask, out_dir):
  os.makedirs(out_dir, exist_ok=True)

  # 1) 전체 산점도
  fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
  ax.scatter(mu, sigma, s=8, alpha=0.35, c='tab:blue')
  ax.set_xlabel('μ(o) — 예측 평균 잔여 스텝'); ax.set_ylabel('σ(o) — 예측 표준편차')
  ax.set_title('μ vs σ — 전체 held-out 표본')
  fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mu_sigma_all.png')); plt.close(fig)

  # 2) 고위험만 산점도
  fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
  ax.scatter(mu[high_risk_mask], sigma[high_risk_mask], s=10, alpha=0.5, c='tab:red')
  ax.set_xlabel('μ(o)'); ax.set_ylabel('σ(o)')
  ax.set_title('μ vs σ — 고위험 부분집합만')
  fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mu_sigma_highrisk.png')); plt.close(fig)

  # 3) 박스플롯 (전체 vs 고위험, μ 등분위 구간별 σ)
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110, sharey=True)
  for ax, (mask, title) in zip(axes, [(np.ones(len(mu), bool), '전체'),
                                       (high_risk_mask, '고위험(top)')]):
    m, s = mu[mask], sigma[mask]
    n_bins = 5
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bidx = np.clip(np.digitize(m, edges[1:-1]), 0, n_bins - 1)
    data = [s[bidx == b] for b in range(n_bins) if (bidx == b).sum() > 0]
    labels = [f'{edges[b]:.0f}~{edges[b+1]:.0f}' for b in range(n_bins) if (bidx == b).sum() > 0]
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title); ax.set_xlabel('μ 구간'); ax.tick_params(axis='x', rotation=30)
  axes[0].set_ylabel('σ(o)')
  fig.suptitle('μ 구간별 σ 분포 — 전체 vs 고위험')
  fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mu_bins_sigma_boxplot.png')); plt.close(fig)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--dstg-ckpt', default='checkpoints/grasp_carry_dstg_deadline/predictor.pkl')
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--seed', type=int, default=0, help='dstg 학습 때 쓴 val 분할 시드')
  ap.add_argument('--top-pct', type=float, default=20.0, help='고위험 집합 = μ 상위 X%%')
  ap.add_argument('--n-bins', type=int, default=5, help='μ 등분위 구간 수')
  ap.add_argument('--out-dir', default='results/mu_sigma_highrisk')
  args = ap.parse_args()

  held = load_held_out(args.data, args.seed)
  print(f'held-out: {held["n_val_eps"]}개 에피소드, {len(held["obs"])}개 transition '
        f'(성공율 {held["is_success"].mean():.1%})')

  reward = StgReward(args.dstg_ckpt, statistic='mean')
  mu, sigma = reward.mean_std(held['obs'])

  thresh = np.percentile(mu, 100 - args.top_pct)
  high_risk = mu >= thresh
  print(f'\n고위험 문턱(μ 상위 {args.top_pct:g}%): μ >= {thresh:.1f}  '
        f'(고위험 표본 n={high_risk.sum()})')

  print('\n=== 상관계수 ===')
  corr_all = corr_report(mu, sigma, '전체')
  corr_hr = corr_report(mu[high_risk], sigma[high_risk], f'고위험(top {args.top_pct:g}%)')

  print('\n=== μ 구간별 σ (전체) ===')
  bins_all, bin_idx_all, _ = bin_sigma_stats(mu, sigma, args.n_bins)
  for r in bins_all:
    print(f"  μ [{r['mu_lo']:6.1f}, {r['mu_hi']:6.1f})  n={r['n']:4d}  "
          f"σ median={r['sigma_median']:5.2f}  IQR={r['sigma_iqr']:5.2f}  "
          f"IQR/median={r['sigma_iqr_over_median']:.2f}")

  print(f'\n=== μ 구간별 σ (고위험, top {args.top_pct:g}%) ===')
  bins_hr, bin_idx_hr, _ = bin_sigma_stats(mu[high_risk], sigma[high_risk], args.n_bins)
  for r in bins_hr:
    print(f"  μ [{r['mu_lo']:6.1f}, {r['mu_hi']:6.1f})  n={r['n']:4d}  "
          f"σ median={r['sigma_median']:5.2f}  IQR={r['sigma_iqr']:5.2f}  "
          f"IQR/median={r['sigma_iqr_over_median']:.2f}")

  print('\n=== 같은 μ 구간, σ가 가장 다른 표본 쌍 (고위험 집합 안에서) ===')
  examples = find_example_pairs(mu[high_risk], sigma[high_risk], bin_idx_hr,
                                held['is_success'][high_risk], args.n_bins)
  for e in examples:
    print(f"  bin{e['bin']}: μ≈{e['mu']:.1f}(성공={e['succ_low']}) σ={e['sigma_low']:.2f}  "
          f"vs  μ≈{e['mu_pair']:.1f}(성공={e['succ_high']}) σ={e['sigma_high']:.2f}  "
          f"(σ차이={e['sigma_gap']:.2f})")

  make_plots(mu, sigma, high_risk, args.out_dir)
  print(f'\n플롯 저장: {args.out_dir}/mu_sigma_all.png, mu_sigma_highrisk.png, '
        f'mu_bins_sigma_boxplot.png')

  return dict(corr_all=corr_all, corr_hr=corr_hr, bins_all=bins_all, bins_hr=bins_hr,
              examples=examples, n_total=len(mu), n_high_risk=int(high_risk.sum()),
              threshold=float(thresh), top_pct=args.top_pct)


if __name__ == '__main__':
  main()
