"""분산의 '급격한 변화(스파이크)'를 만드는 요인 분석 (단기 목표, 2026-07-15).

σ²는 고정된 예측기 f(cur_pos, cur_vel, goal)의 출력이므로 Δσ²는 블랙박스가
아니라 정확히 분해 가능하다. 네 갈래로 연다:

  [A] 분포 내부 분해  σ² = between-mode + within-mode.
      가설: 스파이크는 봉우리(간격~14-30 step) 사이의 확률 질량 이동이 만들며
      (질량 δ 이동 -> Δσ² ~ δ·gap²), 봉우리가 넓어지는 게 아니다.
  [B] 입력 귀속  관측을 하나만 바꿔 f 재호출: Δσ²를 위치 변화분/속도 변화분/
      상호작용으로 정확 배분.
  [C] 민감도 지형  goal·속도 고정 후 위치 그리드에서 σ² 히트맵 -> '절벽' 지도.
  [D] 선형화 검증  ∇σ²(obs_t)·Δobs 가 실제 Δσ²를 얼마나 맞추는지 (백박스가
      닫히는지의 합격선).

스파이크 정의: |Δσ²| ≥ k × (에피소드 내 |Δσ²| 중앙값), k∈{3,5,8} 민감도 체크,
성공 에피소드의 마지막 6 step(알려진 종말 붕괴)은 제외.
"""

import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from stg_probe import STGProbe
from src.diagnose_multimodal import demo_episode, learned_episode, dist_peaks

TERMINAL_SKIP = 6  # 성공 직전 붕괴 구간 제외 폭


# ------------------------------------------------ [A] between/within 분해
def mode_decompose(probs, bin_vals, rel_prom=0.2):
  """봉우리 세그먼트(경계=인접 피크 사이 최소점)로 나눠
  between-mode / within-mode 분산과 모드 목록(질량, 평균)을 반환."""
  peaks = dist_peaks(probs, rel_prom)
  mu_tot = float(np.sum(bin_vals * probs))
  var_tot = float(np.sum(bin_vals ** 2 * probs) - mu_tot ** 2)
  if len(peaks) < 2:
    return dict(between=0.0, within=var_tot, var=var_tot,
                modes=[(1.0, mu_tot)], peaks=peaks)
  bounds = [0]
  for a, b in zip(peaks[:-1], peaks[1:]):
    bounds.append(a + int(np.argmin(probs[a:b + 1])))
  bounds.append(len(probs))
  modes, between, within = [], 0.0, 0.0
  for lo, hi in zip(bounds[:-1], bounds[1:]):
    mass = float(probs[lo:hi].sum())
    if mass < 1e-9:
      continue
    mu_k = float(np.sum(bin_vals[lo:hi] * probs[lo:hi]) / mass)
    var_k = float(np.sum(bin_vals[lo:hi] ** 2 * probs[lo:hi]) / mass - mu_k ** 2)
    modes.append((mass, mu_k))
    between += mass * (mu_k - mu_tot) ** 2
    within += mass * var_k
  return dict(between=between, within=within, var=var_tot, modes=modes,
              peaks=peaks)


def mass_transfer(modes_a, modes_b, match_tol=5.0):
  """모드를 평균 위치로 매칭해 이동한 질량과 (질량가중) 유효 간격을 추정."""
  used = set()
  moved = 0.0
  for mass_a, mu_a in modes_a:
    best, best_d = None, match_tol
    for j, (mass_b, mu_b) in enumerate(modes_b):
      if j in used:
        continue
      d = abs(mu_a - mu_b)
      if d < best_d:
        best, best_d = j, d
    if best is None:
      moved += mass_a  # 사라진 모드
    else:
      used.add(best)
      moved += abs(modes_b[best][0] - mass_a)
  for j, (mass_b, _) in enumerate(modes_b):
    if j not in used:
      moved += mass_b  # 새로 생긴 모드
  return moved / 2.0


# ------------------------------------------------ [B]/[D] 예측기 백박스 접근
def make_var_fns(probe):
  bin_vals = jnp.asarray(probe.bin_vals)
  params, network = probe.params, probe.nets.network

  def var_of_concat(concat):           # concat: (6,) 정규화 관측
    preds = network.apply(params, concat[None])
    p = jax.nn.softmax(preds.dist_to_succ_dist_params.logits[0])
    mu = jnp.sum(bin_vals * p)
    return jnp.sum(bin_vals ** 2 * p) - mu ** 2

  def var_batch(concats):              # (N, 6)
    preds = network.apply(params, concats)
    p = jax.nn.softmax(preds.dist_to_succ_dist_params.logits, axis=-1)
    mu = jnp.sum(bin_vals * p, axis=-1)
    return jnp.sum(bin_vals ** 2 * p, axis=-1) - mu ** 2

  return (jax.jit(var_of_concat, backend='cpu'),
          jax.jit(jax.grad(var_of_concat), backend='cpu'),
          jax.jit(var_batch, backend='cpu'))


def norm_concat(probe, obs):
  norm = probe.normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
  return np.concatenate(
      [norm['cur_pos'], norm['cur_vel'], norm['goal_pos']], axis=-1)[0]


# --------------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--episodes', type=int, default=60)
  ap.add_argument('--out', default='results/diag_multimodal')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  probe = STGProbe(args.checkpoint)
  var_fn, grad_fn, var_batch = make_var_fns(probe)
  bins = probe.bin_vals

  # ---- 롤아웃 수집 (기질 2종)
  episodes = []  # (name, records, success)
  for seed in range(args.episodes):
    recs, _, _, succ = demo_episode(probe, seed)
    if len(recs) >= 15:
      episodes.append(('demo', recs, succ))
  for seed in range(args.episodes):
    recs, _, succ = learned_episode(probe, seed)
    if len(recs) >= 15:
      episodes.append(('learned', recs, succ))

  # ---- 전이(step t -> t+1) 단위 특징 수집
  trans = []  # dict per transition
  for name, recs, succ in episodes:
    n = len(recs)
    last_ok = n - 1 - (TERMINAL_SKIP if succ else 0)
    var = np.array([r.variance for r in recs])
    med = np.median(np.abs(np.diff(var))) or 1.0
    decomp = [mode_decompose(r.probs, bins) for r in recs]
    for t in range(min(last_ok, n - 1)):
      r0, r1 = recs[t], recs[t + 1]
      d0, d1 = decomp[t], decomp[t + 1]
      dvar = var[t + 1] - var[t]

      # [B] 입력 귀속 (치환)
      c00 = norm_concat(probe, r0.obs)
      c11 = norm_concat(probe, r1.obs)
      c_pos = c00.copy(); c_pos[0:2] = c11[0:2]          # 위치만 교체
      c_vel = c00.copy(); c_vel[2:4] = c11[2:4]          # 속도만 교체
      v00, v_pos, v_vel, v11 = map(float, np.asarray(
          var_batch(jnp.stack([c00, c_pos, c_vel, c11]))))
      attr_pos, attr_vel = v_pos - v00, v_vel - v00
      inter = (v11 - v00) - attr_pos - attr_vel

      # [D] 선형화 예측
      g = np.asarray(grad_fn(jnp.asarray(c00)))
      dvar_lin = float(g @ (c11 - c00))

      speed0 = float(np.linalg.norm(r0.obs['cur_vel']))
      speed1 = float(np.linalg.norm(r1.obs['cur_vel']))
      trans.append(dict(
          substrate=name, ep_median=float(med), dvar=float(dvar),
          d_between=float(d1['between'] - d0['between']),
          d_within=float(d1['within'] - d0['within']),
          transfer=float(mass_transfer(d0['modes'], d1['modes'])),
          n_modes=len(d0['modes']),
          attr_pos=float(attr_pos), attr_vel=float(attr_vel),
          interaction=float(inter), dvar_lin=dvar_lin,
          grad_pos=float(np.linalg.norm(g[0:2])),
          grad_vel=float(np.linalg.norm(g[2:4])),
          speed_before=speed0, speed_after=speed1, dspeed=speed1 - speed0,
      ))

  dvar = np.array([tr['dvar'] for tr in trans])
  adv = np.abs(dvar)
  med = np.array([tr['ep_median'] for tr in trans])
  summary = {'n_transitions': len(trans)}
  print(f'전이 {len(trans)}개 (에피소드 {len(episodes)}개)')

  # ---- 스파이크 판정 (k 민감도)
  spike_masks = {k: adv >= k * med for k in (3, 5, 8)}
  for k, m in spike_masks.items():
    print(f'  k={k}: 스파이크 {m.sum()}개 ({m.mean():.1%})')
  spike = spike_masks[5]

  # ---- [A] between vs within
  d_bet = np.array([tr['d_between'] for tr in trans])
  d_wit = np.array([tr['d_within'] for tr in trans])
  share = np.abs(d_bet) / (np.abs(d_bet) + np.abs(d_wit) + 1e-12)
  res_a = {}
  for k, m in spike_masks.items():
    _, p = stats.mannwhitneyu(share[m], share[~m])
    res_a[k] = dict(spike_median=float(np.median(share[m])),
                    normal_median=float(np.median(share[~m])), p=float(p))
    print(f'[A] k={k}: between 기여율 중앙값 — 스파이크 '
          f'{res_a[k]["spike_median"]:.2f} vs 평상시 '
          f'{res_a[k]["normal_median"]:.2f} (MWU p={p:.2e})')
  transfer = np.array([tr['transfer'] for tr in trans])
  _, p_tr = stats.mannwhitneyu(transfer[spike], transfer[~spike])
  res_a['mass_transfer'] = dict(
      spike_median=float(np.median(transfer[spike])),
      normal_median=float(np.median(transfer[~spike])), p=float(p_tr))
  print(f'[A] 질량 이동량 중앙값 — 스파이크 {np.median(transfer[spike]):.3f} '
        f'vs 평상시 {np.median(transfer[~spike]):.3f} (p={p_tr:.2e})')
  summary['A_between_share'] = res_a

  # ---- [B] 입력 귀속
  a_pos = np.array([tr['attr_pos'] for tr in trans])
  a_vel = np.array([tr['attr_vel'] for tr in trans])
  vel_share = np.abs(a_vel) / (np.abs(a_pos) + np.abs(a_vel) + 1e-12)
  summary['B_attribution'] = dict(
      spike_vel_share_median=float(np.median(vel_share[spike])),
      normal_vel_share_median=float(np.median(vel_share[~spike])),
      spike_interaction_share=float(np.median(
          np.abs([tr['interaction'] for tr in trans])[spike] /
          (adv[spike] + 1e-12))))
  print(f"[B] 스파이크의 속도 귀속률 중앙값 "
        f"{summary['B_attribution']['spike_vel_share_median']:.2f} "
        f"(평상시 {summary['B_attribution']['normal_vel_share_median']:.2f})")

  # ---- [D] 선형화 검증
  dlin = np.array([tr['dvar_lin'] for tr in trans])
  def r2(y, yhat):
    ss = ((y - yhat) ** 2).sum()
    return float(1 - ss / ((y - y.mean()) ** 2).sum())
  summary['D_linearization'] = dict(
      r2_all=r2(dvar, dlin), r2_spike=r2(dvar[spike], dlin[spike]),
      corr_all=float(np.corrcoef(dvar, dlin)[0, 1]))
  print(f"[D] 선형화 R²: 전체 {summary['D_linearization']['r2_all']:.3f}, "
        f"스파이크만 {summary['D_linearization']['r2_spike']:.3f}")

  # ---- 검증 회귀: |Δσ²| ~ 기계론 특징 (이전 운동학 회귀 R²=0.091과 비교)
  y = adv
  feats_mech = np.column_stack([
      np.ones(len(trans)), transfer,
      transfer * np.array([tr['n_modes'] for tr in trans]),
      np.abs(dlin)])
  coef, *_ = np.linalg.lstsq(feats_mech, y, rcond=None)
  r2_mech = r2(y, feats_mech @ coef)
  summary['E_regression'] = dict(r2_mechanistic=float(r2_mech),
                                 r2_kinematic_prev=0.091)
  print(f'[E] |Δσ²| 회귀 R² — 기계론 특징 {r2_mech:.3f} '
        f'(이전 운동학 회귀 0.091)')

  # ================================================================ 그림
  # vs0: 예시 에피소드 (가장 큰 스파이크 포함)
  best_ep = max(episodes, key=lambda e: np.max(np.abs(np.diff(
      [r.variance for r in e[1]][:len(e[1]) - (TERMINAL_SKIP if e[2] else 0)]))))
  name, recs, succ = best_ep
  var_ep = np.array([r.variance for r in recs])
  dec_ep = [mode_decompose(r.probs, bins) for r in recs]
  bet_ep = np.array([d['between'] for d in dec_ep])
  med_ep = np.median(np.abs(np.diff(var_ep)))
  spikes_ep = [t for t in range(len(var_ep) - 1)
               if abs(var_ep[t + 1] - var_ep[t]) >= 5 * med_ep]
  fig, ax = plt.subplots(figsize=(11, 4))
  ax.fill_between(range(len(var_ep)), 0, bet_ep, color='C1', alpha=0.5,
                  label='between-mode 성분 (봉우리 사이)')
  ax.fill_between(range(len(var_ep)), bet_ep, var_ep, color='C0', alpha=0.35,
                  label='within-mode 성분 (봉우리 내부)')
  ax.plot(var_ep, color='C3', lw=1.5, label='σ² 전체')
  for t in spikes_ep:
    ax.axvline(t + 0.5, color='red', ls=':', lw=1.2)
  ax.set_xlabel('step'); ax.set_ylabel('σ²')
  ax.set_title(f'예시 에피소드({name}) — 빨간 점선=스파이크(|Δσ²|≥5×중앙값)')
  ax.legend(fontsize=9)
  fig.tight_layout(); fig.savefig(os.path.join(args.out, 'vs0_example.png'),
                                  dpi=130); plt.close(fig)

  # vs1: [A] 분해
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
  bp = axes[0].boxplot([share[~spike], share[spike]], tick_labels=['평상시', '스파이크'],
                       showmeans=True, widths=0.5)
  axes[0].set_ylabel('|Δbetween| / (|Δbetween|+|Δwithin|)')
  axes[0].set_title('분산 변화 중 봉우리-사이 성분의 비중')
  axes[0].axhline(0.5, color='gray', ls=':', lw=0.8)
  axes[1].scatter(dvar[~spike], d_bet[~spike], s=6, alpha=0.25, color='gray',
                  label='평상시')
  axes[1].scatter(dvar[spike], d_bet[spike], s=14, alpha=0.7, color='C3',
                  label='스파이크')
  lim = np.percentile(np.abs(dvar), 99.5)
  axes[1].plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='y=x (전부 between)')
  axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
  axes[1].set_xlabel('Δσ² (실제)'); axes[1].set_ylabel('Δbetween')
  axes[1].set_title('Δσ² vs 봉우리-사이 성분 변화'); axes[1].legend(fontsize=9)
  fig.tight_layout(); fig.savefig(os.path.join(args.out, 'vs1_decomposition.png'),
                                  dpi=130); plt.close(fig)

  # vs2: [B] 입력 귀속
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
  axes[0].boxplot([vel_share[~spike], vel_share[spike]],
                  tick_labels=['평상시', '스파이크'], showmeans=True, widths=0.5)
  axes[0].axhline(0.5, color='gray', ls=':', lw=0.8)
  axes[0].set_ylabel('|속도 귀속| / (|위치|+|속도|)')
  axes[0].set_title('Δσ²의 입력 귀속 — 속도 변화의 비중')
  axes[1].scatter(a_pos[spike], a_vel[spike], s=14, alpha=0.7, color='C3')
  lim2 = np.percentile(np.abs(np.r_[a_pos[spike], a_vel[spike]]), 99)
  axes[1].plot([-lim2, lim2], [-lim2, lim2], 'k--', lw=0.8)
  axes[1].axhline(0, color='gray', lw=0.6); axes[1].axvline(0, color='gray', lw=0.6)
  axes[1].set_xlim(-lim2, lim2); axes[1].set_ylim(-lim2, lim2)
  axes[1].set_xlabel('위치 변화 귀속분'); axes[1].set_ylabel('속도 변화 귀속분')
  axes[1].set_title('스파이크의 귀속 산점도')
  fig.tight_layout(); fig.savefig(os.path.join(args.out, 'vs2_attribution.png'),
                                  dpi=130); plt.close(fig)

  # vs3: [C] 민감도 지형 (goal 고정, 속도 3조건)
  goal = np.array([0.5, 0.5], dtype=np.float32)
  gs = 61
  xs = np.linspace(-1, 1, gs, dtype=np.float32)
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
  for ax, speed in zip(axes, [0.0, 0.004, 0.008]):
    concats = []
    for y_ in xs:
      for x_ in xs:
        pos = np.array([x_, y_], dtype=np.float32)
        d = goal - pos
        nrm = np.linalg.norm(d)
        vel = (speed * d / nrm if nrm > 1e-6 else np.zeros(2)).astype(np.float32)
        concats.append(norm_concat(probe, dict(cur_pos=pos, cur_vel=vel,
                                               goal_pos=goal)))
    vmap = np.asarray(var_batch(jnp.asarray(np.stack(concats)))).reshape(gs, gs)
    im = ax.imshow(vmap, origin='lower', extent=(-1, 1, -1, 1), cmap='magma')
    ax.scatter(*goal, marker='*', s=180, color='cyan')
    ax.set_title(f'σ² 지형 — 목표방향 속력 {speed}')
    fig.colorbar(im, ax=ax, fraction=0.046)
  fig.suptitle('민감도 지형: 같은 위치라도 속력에 따라 σ² 절벽이 이동', fontsize=13)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'vs3_landscape.png'), dpi=130)
  plt.close(fig)

  # vs4: [D] 선형화
  fig, ax = plt.subplots(figsize=(5.6, 5))
  ax.scatter(dlin[~spike], dvar[~spike], s=6, alpha=0.25, color='gray',
             label='평상시')
  ax.scatter(dlin[spike], dvar[spike], s=14, alpha=0.7, color='C3',
             label='스파이크')
  lim3 = np.percentile(np.abs(dvar), 99.5)
  ax.plot([-lim3, lim3], [-lim3, lim3], 'k--', lw=1)
  ax.set_xlim(-lim3, lim3); ax.set_ylim(-lim3, lim3)
  ax.set_xlabel('∇σ²·Δobs (1차 예측)'); ax.set_ylabel('Δσ² (실제)')
  ax.set_title(f"선형화 검증 R²={summary['D_linearization']['r2_all']:.2f}")
  ax.legend(fontsize=9)
  fig.tight_layout(); fig.savefig(os.path.join(args.out, 'vs4_linearization.png'),
                                  dpi=130); plt.close(fig)

  # ---- [F] 가속 vs 감속 — 속력이 커지는 중인지 작아지는 중인지가 부호를 가르나
  dspeed = np.array([tr['dspeed'] for tr in trans])
  speeding_up = dspeed > 0
  slowing_down = dspeed < 0
  frac_up_accel = (dvar[speeding_up] > 0).mean()
  frac_up_decel = (dvar[slowing_down] > 0).mean()
  _, p_f = stats.mannwhitneyu(dvar[speeding_up], dvar[slowing_down])
  r_f, p_corr = stats.spearmanr(dspeed, dvar)
  summary['F_accel_vs_decel'] = dict(
      n_accel=int(speeding_up.sum()), n_decel=int(slowing_down.sum()),
      frac_var_up_when_accel=float(frac_up_accel),
      frac_var_up_when_decel=float(frac_up_decel),
      mwu_p=float(p_f), spearman_dspeed_dvar=float(r_f), spearman_p=float(p_corr))
  print(f'[F] 가속 중 분산증가비율={frac_up_accel:.2f}  감속 중 분산증가비율='
        f'{frac_up_decel:.2f}  (MWU p={p_f:.2e})')
  print(f'[F] Spearman(dspeed, dvar) r={r_f:.3f} p={p_corr:.2e}')

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
  axes[0].boxplot([dvar[slowing_down], dvar[speeding_up]],
                  tick_labels=['감속 중\n(dspeed<0)', '가속 중\n(dspeed>0)'],
                  showmeans=True, widths=0.5, showfliers=False)
  axes[0].axhline(0, color='gray', ls=':', lw=0.8)
  axes[0].set_ylabel('Δσ² (양수=분산 증가)')
  axes[0].set_title(f'가속/감속에 따른 Δσ² (MWU p={p_f:.1e})')

  edges = np.quantile(dspeed, np.linspace(0, 1, 12))
  centers, means, fracup = [], [], []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dspeed >= lo) & (dspeed < hi)
    if m.sum() >= 20:
      centers.append(0.5 * (lo + hi))
      means.append(dvar[m].mean())
      fracup.append((dvar[m] > 0).mean())
  axes[1].axhline(0, color='gray', lw=0.8)
  axes[1].axvline(0, color='gray', lw=0.8)
  axes[1].bar(centers, means, width=(edges[1] - edges[0]) * 0.6,
             color=['#3a7ca5' if c < 0 else '#d1495b' for c in centers])
  axes[1].set_xlabel('Δ속력 (음수=감속, 양수=가속)')
  axes[1].set_ylabel('평균 Δσ²')
  axes[1].set_title('속력 변화량 vs 분산 변화 (구간 평균)')
  fig.suptitle('가속 중이냐 감속 중이냐가 분산 증감 부호를 가르는가', fontsize=13)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'vs5_accel_decel.png'), dpi=130)
  plt.close(fig)

  with open(os.path.join(args.out, 'variance_spikes.json'), 'w') as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
  main()
