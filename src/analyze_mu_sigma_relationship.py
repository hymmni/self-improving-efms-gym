"""μ(steps-to-go 기댓값)와 σ²(분산)의 전역적 관계 분석 (context_4.md 액션아이템 1).

교수님 피드백: "식을 먼저 정의하지 말고 성질 파악이 우선" — 국소 사건(전환,
순간이동, 스파이크) 중심이었던 지금까지 분석과 달리, 여기서는 전체 상태공간에서
μ-σ² 관계 자체를 정량화한다.

  A1. μ vs σ² 전역 상관관계 (원값 + μ 구간별 조건부 분포)
  A2. 캘리브레이션: σ가 실제 예측 오차(|μ - 실제 잔여 스텝|)를 예견하는가
      (다 성공한 에피소드만 대상 — 사후에 "실제 몇 스텝 남았었는지"를 정확히 안다)
  A3. μ 자체가 실제 잔여 스텝의 충실한 대리 지표인가 (편향 여부 — 웨이포인트
      구조로 인한 체계적 과대추정 가능성)
  A4. μ 수준(=진행 단계)에 따라 μ-σ² 관계가 달라지는가 — "초반엔 분산 커도
      자연스럽지만 후반엔 작아야 의미있다"는 교수님 직관의 직접 검증
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from stg_probe import STGProbe
from src.diagnose_multimodal import demo_episode, learned_episode


def collect(probe, n_episodes):
  rows = []  # dict per step: mu, var, dist, progress, substrate, success, actual_remaining
  for substrate, fn in [('demo', demo_episode), ('learned', learned_episode)]:
    for seed in range(n_episodes):
      if substrate == 'demo':
        recs, _, _, succ = fn(probe, seed)
      else:
        recs, _, succ = fn(probe, seed)
      n = len(recs)
      if n < 10:
        continue
      goal = recs[-1].obs['goal_pos']
      for i, r in enumerate(recs):
        dist = float(np.linalg.norm(r.obs['cur_pos'] - goal))
        rows.append(dict(
            substrate=substrate, seed=seed, step=i, n_total=n, success=succ,
            mu=r.expectation, var=r.variance, dist=dist,
            progress=i / (n - 1) if n > 1 else 0.0,
            actual_remaining=(n - 1 - i) if succ else np.nan,
        ))
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--episodes', type=int, default=150)
  ap.add_argument('--out', default='results/diag_multimodal')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  probe = STGProbe(args.checkpoint)
  rows = collect(probe, args.episodes)
  mu = np.array([r['mu'] for r in rows])
  var = np.array([r['var'] for r in rows])
  sigma = np.sqrt(np.clip(var, 0, None))
  dist = np.array([r['dist'] for r in rows])
  progress = np.array([r['progress'] for r in rows])
  succ_mask = np.array([r['success'] for r in rows])
  actual = np.array([r['actual_remaining'] for r in rows], dtype=float)
  print(f'표본 {len(rows)}개 (성공 에피소드 스텝 {int(np.nansum(~np.isnan(actual)))}개)')

  summary = {'n_rows': len(rows)}

  # ---------------------------------------------------- A1. μ vs σ² 전역 관계
  r_pearson, p_pearson = stats.pearsonr(mu, var)
  r_spearman, p_spearman = stats.spearmanr(mu, var)
  r_pearson_sigma, _ = stats.pearsonr(mu, sigma)
  summary['A1_global_corr'] = dict(
      pearson_mu_var=float(r_pearson), spearman_mu_var=float(r_spearman),
      pearson_mu_sigma=float(r_pearson_sigma))
  print(f'[A1] Pearson(μ,σ²)={r_pearson:.3f}  Spearman(μ,σ²)={r_spearman:.3f}  '
        f'Pearson(μ,σ)={r_pearson_sigma:.3f}')

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
  hb = axes[0].hexbin(mu, var, gridsize=45, cmap='viridis', mincnt=1, bins='log')
  axes[0].set_xlabel('μ (예측 잔여 스텝)'); axes[0].set_ylabel('σ² (분산)')
  axes[0].set_title(f'μ vs σ² 밀도 (Pearson r={r_pearson:.2f})')
  fig.colorbar(hb, ax=axes[0], label='log(count)')

  # μ 구간별 조건부 σ² 분포 (중앙값 + IQR 밴드)
  edges = np.quantile(mu, np.linspace(0, 1, 21))
  centers, med, q25, q75 = [], [], [], []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mu >= lo) & (mu < hi)
    if m.sum() >= 20:
      centers.append(0.5 * (lo + hi))
      med.append(np.median(var[m])); q25.append(np.percentile(var[m], 25))
      q75.append(np.percentile(var[m], 75))
  axes[1].plot(centers, med, color='C3', lw=2, label='σ² 중앙값')
  axes[1].fill_between(centers, q25, q75, color='C3', alpha=0.2, label='IQR')
  axes[1].set_xlabel('μ 구간'); axes[1].set_ylabel('σ² (조건부)')
  axes[1].set_title('μ 수준별 조건부 σ² 분포'); axes[1].legend(fontsize=8)

  # 변동계수 CV = σ/μ — "상대적" 불확실성이 μ 수준에 따라 어떻게 변하나
  cv_med = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mu >= lo) & (mu < hi) & (mu > 1e-6)
    if m.sum() >= 20:
      cv_med.append(np.median(sigma[m] / mu[m]))
    else:
      cv_med.append(np.nan)
  axes[2].plot(centers, cv_med, color='C0', marker='o', ms=3, lw=1.5)
  axes[2].axhline(1.0, color='gray', ls=':', lw=0.8, label='σ=μ')
  axes[2].set_xlabel('μ 구간'); axes[2].set_ylabel('변동계수 σ/μ (중앙값)')
  axes[2].set_title('μ 수준별 상대적 불확실성(σ/μ)'); axes[2].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'mv1_global_mu_sigma.png'), dpi=130)
  plt.close(fig)

  # -------------------------------------------- A2/A3. 캘리브레이션 (성공 에피소드만)
  ok = ~np.isnan(actual)
  err = np.abs(mu[ok] - actual[ok])
  r_calib, p_calib = stats.spearmanr(sigma[ok], err)
  r_bias, p_bias = stats.pearsonr(mu[ok], actual[ok])
  bias = float(np.mean(mu[ok] - actual[ok]))
  slope, intercept = np.polyfit(actual[ok], mu[ok], 1)
  summary['A2_calibration'] = dict(
      spearman_sigma_abserror=float(r_calib), p=float(p_calib),
      n=int(ok.sum()))
  summary['A3_mu_bias'] = dict(
      pearson_mu_actual=float(r_bias), mean_bias_mu_minus_actual=bias,
      slope=float(slope), intercept=float(intercept))
  print(f'[A2] Spearman(σ, |μ-실제잔여|) = {r_calib:.3f} (p={p_calib:.2e}, n={ok.sum()})')
  print(f'[A3] Pearson(μ,실제잔여)={r_bias:.3f}  평균 편향(μ-실제)={bias:.2f}  '
        f'회귀: μ ≈ {slope:.2f}·실제 + {intercept:.2f}')

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
  lim = np.percentile(actual[ok], 99)
  axes[0].hexbin(actual[ok], mu[ok], gridsize=45, cmap='viridis', mincnt=1, bins='log')
  axes[0].plot([0, lim], [0, lim], 'w--', lw=1.2, label='y=x (완벽 예측)')
  axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
  axes[0].set_xlabel('실제 잔여 스텝 (사후 확인)'); axes[0].set_ylabel('μ (예측)')
  axes[0].set_title(f'μ의 정확성 (Pearson r={r_bias:.2f}, 편향={bias:+.1f})')
  axes[0].legend(fontsize=8)

  err_edges = np.quantile(sigma[ok], np.linspace(0, 1, 16))
  ec, em = [], []
  for lo, hi in zip(err_edges[:-1], err_edges[1:]):
    m = (sigma[ok] >= lo) & (sigma[ok] < hi)
    if m.sum() >= 15:
      ec.append(0.5 * (lo + hi)); em.append(np.median(err[m]))
  axes[1].plot(ec, em, color='C3', marker='o', ms=4)
  axes[1].set_xlabel('σ (예측 표준편차)'); axes[1].set_ylabel('실제 오차 |μ-실제| 중앙값')
  axes[1].set_title(f'캘리브레이션: σ가 오차를 예견하는가\n(Spearman r={r_calib:.2f})')

  axes[2].hexbin(sigma[ok], err, gridsize=45, cmap='viridis', mincnt=1, bins='log')
  lim2 = np.percentile(sigma[ok], 99)
  axes[2].set_xlim(0, lim2)
  axes[2].set_ylim(0, np.percentile(err, 99))
  axes[2].set_xlabel('σ'); axes[2].set_ylabel('|μ - 실제 잔여|')
  axes[2].set_title('σ vs 실제 오차 (산점)')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'mv2_calibration.png'), dpi=130)
  plt.close(fig)

  # ---------------------------------------- A4. 진행 단계(progress)별 μ-σ² 관계 변화
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
  bands = [(0.0, 0.33, '초반 (progress<0.33)', 'C0'),
           (0.33, 0.67, '중반', 'C1'),
           (0.67, 1.01, '후반 (progress>0.67)', 'C3')]
  corr_by_band = {}
  for ax_i, ((lo, hi, label, color)) in enumerate(bands):
    m = (progress >= lo) & (progress < hi)
    r_b, _ = stats.spearmanr(mu[m], var[m])
    corr_by_band[label] = dict(n=int(m.sum()), spearman=float(r_b),
                               median_var=float(np.median(var[m])),
                               median_mu=float(np.median(mu[m])))
    axes[0].scatter(mu[m][::5], var[m][::5], s=4, alpha=0.25, color=color,
                    label=f'{label} (r={r_b:.2f})')
  axes[0].set_xlabel('μ'); axes[0].set_ylabel('σ²')
  axes[0].set_title('진행 단계별 μ-σ² 산점 (일부 표본)')
  axes[0].legend(fontsize=7, markerscale=3)

  labels = [b[2] for b in bands]
  meds = [corr_by_band[l]['median_var'] for l in labels]
  axes[1].bar(labels, meds, color=['C0', 'C1', 'C3'])
  axes[1].set_ylabel('σ² 중앙값'); axes[1].set_title('진행 단계별 σ² 중앙값')
  axes[1].tick_params(axis='x', labelsize=8)

  rs = [corr_by_band[l]['spearman'] for l in labels]
  axes[2].bar(labels, rs, color=['C0', 'C1', 'C3'])
  axes[2].axhline(0, color='gray', lw=0.8)
  axes[2].set_ylabel('Spearman(μ,σ²)'); axes[2].set_title('진행 단계별 μ-σ² 상관')
  axes[2].tick_params(axis='x', labelsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'mv3_progress_dependence.png'), dpi=130)
  plt.close(fig)
  summary['A4_progress_band_corr'] = corr_by_band

  with open(os.path.join(args.out, 'mu_sigma_relationship.json'), 'w') as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
  main()
