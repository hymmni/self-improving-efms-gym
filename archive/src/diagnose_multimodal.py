"""STG 다봉성/분산 급증의 원인 진단 (references/context_3.md).

가설 A (진동 아티팩트): PD 제어가 underdamped라 goal을 빗겨나가면 감쇠 진동
주기 T_d 간격으로 재접근하고, 그 간격이 STG 분포의 봉우리 간격으로 나타난다.
가설 B (웨이포인트 정체성 모호성): 같은 (pos, vel, goal) 관측이라도 남은
웨이포인트 수가 달라 time_to_success 라벨이 웨이포인트 구간 이동시간만큼
떨어진 값들로 갈라지고, 그 간격이 봉우리 간격으로 나타난다.

분석 1  PD 응답이 underdamped인가 (이론 고유값 + 실측 링잉)
분석 2  분산 sigma^2_t 궤적: 웨이포인트 통과/goal 미스 시점과의 정렬, 점프성 판정
분석 3  다봉성 판정과 인접 봉우리 간격 측정
분석 4  봉우리 간격 vs {T_d, 웨이포인트 구간 이동시간} 비교

실행:
  python -m src.diagnose_multimodal --checkpoint checkpoints/std/predictor_f100.pkl
"""

import argparse
import json
import os

import numpy as np
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pointmass_core import Point2D, pd_controller
from stg_probe import STGProbe

KP, KD, SUBSTEPS = 2e-4, 1.25e-2, 10  # pointmass_core.pd_controller / Point2D
SUCCESS_RADIUS = 0.15


# ---------------------------------------------------------------- 분석 1: PD 응답
def theoretical_pd_modes():
  """env step 1회(=substep 10회, 액션 zero-order hold)의 정확한 선형 사상.

  substep: v <- v + a, x <- x + v 이므로 10 substep 후
    v' = v + 10a,  x' = x + 10v + 55a,  a = -Kp*e - Kd*v (e = x - goal)
  """
  A = np.array([[1 - 55 * KP, SUBSTEPS - 55 * KD],
                [-SUBSTEPS * KP, 1 - SUBSTEPS * KD]])
  eig = np.linalg.eigvals(A)
  lam = eig[0]
  mag, theta = float(np.abs(lam)), float(np.abs(np.angle(lam)))
  underdamped = bool(np.iscomplex(eig).any())
  T_d = 2 * np.pi / theta if underdamped else np.inf
  # 이산 고유값 -> 등가 연속계 감쇠비
  ln_mag = np.log(mag)
  zeta = float(-ln_mag / np.sqrt(ln_mag ** 2 + theta ** 2)) if underdamped else 1.0
  # 연속 근사 (substep 단위 dt=1, m=1)
  omega_n = np.sqrt(KP)
  zeta_cont = KD / (2 * np.sqrt(KP))
  T_d_cont = 2 * np.pi / (omega_n * np.sqrt(1 - zeta_cont ** 2)) / SUBSTEPS
  return dict(eigenvalues=[complex(e) for e in eig], underdamped=underdamped,
              T_d_steps=float(T_d), zeta=zeta,
              zeta_continuous=float(zeta_cont), T_d_continuous=float(T_d_cont))


def empirical_pd_response(n_steps=250):
  """정지 상태에서 단일 목표를 향한 PD 접근 실측 (success에서 멈추지 않고 계속)."""
  env = Point2D()
  env._cur_pos = np.array([-0.8, 0.0], dtype=np.float32)
  env._cur_vel = np.zeros(2, dtype=np.float32)
  env.set_goal(np.array([0.6, 0.0], dtype=np.float32))
  env._cur_episode_traj = [env._cur_pos.copy()]
  errs, vels = [], []
  for _ in range(n_steps):
    act = pd_controller(env._cur_pos, env._cur_vel, env._goal_pos)
    env.step(act)
    errs.append(float(env._cur_pos[0] - env._goal_pos[0]))
    vels.append(float(env._cur_vel[0]))
  errs, vels = np.array(errs), np.array(vels)

  # 오버슈트 피크(양/음 번갈아)에서 로그 감쇠율과 주기 추정
  pk_pos, _ = find_peaks(errs)
  pk_neg, _ = find_peaks(-errs)
  peaks = sorted([(i, abs(errs[i])) for i in np.r_[pk_pos, pk_neg]])
  overshoot = bool(pk_pos.size and errs[pk_pos].max() > 0)  # 목표를 지나쳤는가
  T_d_emp = zeta_emp = None
  if len(pk_pos) >= 2:
    T_d_emp = float(np.mean(np.diff(pk_pos)))  # 같은 부호 피크 간격 = 감쇠 주기
    ratios = errs[pk_pos][:-1] / errs[pk_pos][1:]
    ratios = ratios[ratios > 0]
    if ratios.size:
      delta = float(np.mean(np.log(ratios)))
      zeta_emp = float(delta / np.sqrt(4 * np.pi ** 2 + delta ** 2))
  return errs, vels, peaks, dict(overshoot=overshoot, T_d_emp_steps=T_d_emp,
                                 zeta_emp=zeta_emp)


# ------------------------------------------- 롤아웃 수집 (데모 PD / 학습 정책)
def demo_episode(probe, seed, num_waypoints=5, max_steps=400):
  """generate_dataset의 웨이포인트 순회 로직을 재현하되 STG 분포를 함께 기록.

  # From: pointmass_core.generate_dataset (웨이포인트 전환 로직 부분)
  관측은 항상 최종 goal만 담으므로(학습 데이터와 동일 분포) 예측기 입장에선
  컨트롤러가 어떤 웨이포인트를 쫓는지 보이지 않는다.
  """
  np.random.seed(seed)
  env = Point2D()
  ts = env.reset()
  cur_obs = ts.observation
  key = jax.random.PRNGKey(seed)

  waypoint_idx = 0
  cur_waypoint = env.sample_goal()
  records, dists, switch_steps = [], [], []
  step = 0
  while not env.success() and step < max_steps:
    if env.success(waypoint=cur_waypoint):
      waypoint_idx = min(waypoint_idx + 1, num_waypoints)
      cur_waypoint = (cur_obs['goal_pos'] if waypoint_idx == num_waypoints
                      else env.sample_goal())
      switch_steps.append(step)
    key, sub = jax.random.split(key)
    _, logits = probe._logits_and_act(cur_obs, sub)
    records.append(probe._record(step, cur_obs, logits))
    dists.append(float(np.linalg.norm(cur_obs['cur_pos'] - cur_obs['goal_pos'])))
    act = pd_controller(cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
    ts = env.step(act)
    cur_obs = ts.observation
    step += 1
  return records, np.array(dists), switch_steps, env.success()


def learned_episode(probe, seed, max_steps=200):
  env = Point2D()
  records = probe.rollout(env, max_steps=max_steps, policy='learned', seed=seed)
  dists = np.array([float(np.linalg.norm(r.obs['cur_pos'] - r.obs['goal_pos']))
                    for r in records])
  return records, dists, env.success()


def find_miss_events(dists, near=0.45):
  """goal에 접근했다가(0.45 이내) 성공 반경 밖에서 되돌아 멀어진 시점."""
  idx, _ = find_peaks(-dists)
  return [int(i) for i in idx if SUCCESS_RADIUS < dists[i] < near]


# ------------------------------------------------- 분석 3: 다봉 판정/봉우리 간격
def dist_peaks(probs, rel_prominence=0.2, min_distance=4):
  """경계 bin의 봉우리도 잡히도록 0-패딩 후 탐지. bin 폭=1 step이라
  반환되는 bin 인덱스가 곧 steps-to-go 값."""
  padded = np.r_[0.0, probs, 0.0]
  idx, _ = find_peaks(padded, prominence=rel_prominence * probs.max(),
                      distance=min_distance)
  return idx - 1


def peak_spacings(records, rel_prominence=0.2):
  """다봉 스텝들의 (스텝 인덱스, 인접 봉우리 간격 리스트)."""
  out = []
  for r in records:
    pk = dist_peaks(r.probs, rel_prominence)
    if len(pk) >= 2:
      out.append((r.step_idx, np.diff(pk).tolist()))
  return out


# ------------------------------------------------------ 분석 2: 점프성 정량화
def jump_stats(varis, events, window=3):
  """이벤트 주변 |d(sigma^2)| 최대 vs 평상시 중앙값 비율. 비율이 크면 이산 점프."""
  dv = np.abs(np.diff(varis))
  if dv.size == 0:
    return None
  mask = np.zeros(dv.size, dtype=bool)
  for e in events:
    mask[max(0, e - 1):min(dv.size, e + window)] = True
  base = np.median(dv[~mask]) if (~mask).any() else np.nan
  ev = float(dv[mask].max()) if mask.any() else np.nan
  return dict(baseline_median=float(base), event_max=ev,
              ratio=float(ev / base) if base and base > 0 else np.nan)


# --------------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default='checkpoints/std/predictor_f100.pkl')
  ap.add_argument('--episodes', type=int, default=60)
  ap.add_argument('--out', default='results/diag_multimodal')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  probe = STGProbe(args.checkpoint)
  assert probe.bin_vals[1] - probe.bin_vals[0] == 1.0, \
      'bin 폭=1step 예측기가 필요 (봉우리 간격을 스텝 단위로 읽기 위함)'
  summary = {}

  # ---- 분석 1
  theo = theoretical_pd_modes()
  errs, vels, peaks, emp = empirical_pd_response()
  summary['analysis1'] = {**theo, **emp}
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  axes[0].plot(errs, label='위치 오차 e(t)')
  axes[0].axhline(0, color='gray', lw=0.8)
  axes[0].axhline(SUCCESS_RADIUS, color='green', ls='--', lw=0.8, label='성공 반경')
  axes[0].axhline(-SUCCESS_RADIUS, color='green', ls='--', lw=0.8)
  axes[0].set_xlabel('env step'); axes[0].set_ylabel('오차')
  axes[0].set_title(f"PD 단일 구간 응답 (이론 ζ={theo['zeta']:.3f}, "
                    f"T_d={theo['T_d_steps']:.1f} step)")
  axes[0].legend()
  axes[1].plot(vels, color='C1')
  axes[1].axhline(0, color='gray', lw=0.8)
  axes[1].set_xlabel('env step'); axes[1].set_ylabel('속도')
  axes[1].set_title('속도 (링잉 확인)')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'a1_pd_response.png'), dpi=130)
  plt.close(fig)

  # ---- 데이터 수집 (두 기질: 데모 PD 궤적 / 학습 정책 궤적)
  demo_eps, learned_eps = [], []
  leg_durations = []
  for seed in range(args.episodes):
    recs, dists, switches, succ = demo_episode(probe, seed)
    if len(recs) < 15:
      continue
    demo_eps.append((recs, dists, switches, succ))
    bounds = [0] + switches + [len(recs)]
    leg_durations += list(np.diff(bounds))
  for seed in range(args.episodes):
    recs, dists, succ = learned_episode(probe, seed)
    if len(recs) < 15:
      continue
    learned_eps.append((recs, dists, succ))
  leg_durations = np.array(leg_durations)
  summary['waypoint_leg_steps'] = dict(
      median=float(np.median(leg_durations)), mean=float(leg_durations.mean()),
      p25=float(np.percentile(leg_durations, 25)),
      p75=float(np.percentile(leg_durations, 75)), n=int(leg_durations.size))

  # ---- 분석 2: 분산 궤적 + 이벤트 마킹 (데모 4개 + 학습 4개 예시 그림)
  fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharex=False)
  jump_demo, jump_learned = [], []
  for col in range(4):
    for row, (label, eps) in enumerate([('데모(PD+웨이포인트)', demo_eps),
                                        ('학습 정책', learned_eps)]):
      if col >= len(eps):
        continue
      ax = axes[row, col]
      if row == 0:
        recs, dists, switches, succ = eps[col]
      else:
        recs, dists, succ = eps[col]
        switches = []
      varis = np.array([r.variance for r in recs])
      misses = find_miss_events(dists)
      ax.plot(varis, color='C3', lw=1.2)
      for s in switches:
        ax.axvline(s, color='green', ls='--', lw=1, alpha=0.8)
      for m in misses:
        ax.axvline(m, color='red', ls=':', lw=1.5, alpha=0.9)
      ax.set_title(f'{label} ep{col} (성공={succ})', fontsize=10)
      ax.set_xlabel('step'); ax.set_ylabel('σ²')
  fig.suptitle('분산 궤적 — 초록 파선: 웨이포인트 통과, 빨강 점선: goal 미스',
               fontsize=13)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'a2_variance_events.png'), dpi=130)
  plt.close(fig)

  for recs, dists, switches, succ in demo_eps:
    js = jump_stats(np.array([r.variance for r in recs]),
                    find_miss_events(dists) + switches)
    if js and np.isfinite(js['ratio']):
      jump_demo.append(js['ratio'])
  for recs, dists, succ in learned_eps:
    js = jump_stats(np.array([r.variance for r in recs]), find_miss_events(dists))
    if js and np.isfinite(js['ratio']):
      jump_learned.append(js['ratio'])
  summary['analysis2_jump_ratio'] = dict(
      demo_median=float(np.median(jump_demo)) if jump_demo else None,
      learned_median=float(np.median(jump_learned)) if jump_learned else None,
      note='이벤트 주변 |Δσ²| 최대 / 평상시 |Δσ²| 중앙값; >>1 이면 이산 점프')

  # ---- 분석 3: 다봉성/봉우리 간격 (민감도: prominence 0.1/0.2/0.3)
  spac = {}
  for prom in (0.1, 0.2, 0.3):
    all_sp, miss_sp = {'demo': [], 'learned': []}, {'demo': [], 'learned': []}
    mm_frac = {}
    for name, eps in [('demo', demo_eps), ('learned', learned_eps)]:
      n_steps = n_mm = 0
      for ep in eps:
        recs, dists = ep[0], ep[1]
        misses = set()
        for m in find_miss_events(dists):
          misses.update(range(m, m + 10))
        for r in recs:
          pk = dist_peaks(r.probs, prom)
          n_steps += 1
          if len(pk) >= 2:
            n_mm += 1
            sp = np.diff(pk).tolist()
            all_sp[name] += sp
            if r.step_idx in misses:
              miss_sp[name] += sp
      mm_frac[name] = n_mm / max(n_steps, 1)
    spac[prom] = dict(
        multimodal_fraction=mm_frac,
        spacing_median={k: float(np.median(v)) if v else None
                        for k, v in all_sp.items()},
        spacing_median_after_miss={k: float(np.median(v)) if v else None
                                   for k, v in miss_sp.items()},
        n_spacings={k: len(v) for k, v in all_sp.items()},
        _raw=all_sp)
  summary['analysis3'] = {
      str(k): {kk: vv for kk, vv in v.items() if kk != '_raw'}
      for k, v in spac.items()}

  # ---- 분석 4: 후보 비교 그림
  T_d = summary['analysis1']['T_d_emp_steps'] or summary['analysis1']['T_d_steps']
  leg_med = summary['waypoint_leg_steps']['median']
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  for ax, name, title in [(axes[0], 'demo', '데모(PD+웨이포인트) 궤적'),
                          (axes[1], 'learned', '학습 정책 궤적')]:
    sp = np.array(spac[0.2]['_raw'][name])
    if sp.size:
      ax.hist(sp, bins=np.arange(0, 100, 3), color='C0', alpha=0.75)
    ax.axvline(T_d, color='red', lw=2,
               label=f'후보 A: PD 감쇠 주기 T_d={T_d:.0f}')
    ax.axvline(leg_med, color='green', lw=2,
               label=f'후보 B: 웨이포인트 구간={leg_med:.0f} (중앙값)')
    ax.axvspan(np.percentile(leg_durations, 25),
               np.percentile(leg_durations, 75), color='green', alpha=0.12,
               label='구간 이동시간 IQR')
    ax.set_xlabel('인접 봉우리 간격 (steps-to-go 단위)')
    ax.set_ylabel('빈도'); ax.set_title(title); ax.legend(fontsize=9)
  fig.suptitle('분석 4 — 봉우리 간격 vs 두 후보 주기', fontsize=13)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'a4_spacing_vs_candidates.png'), dpi=130)
  plt.close(fig)
  summary['analysis4'] = dict(
      T_d_used=float(T_d), leg_median=float(leg_med),
      spacing_median_demo=spac[0.2]['spacing_median']['demo'],
      spacing_median_learned=spac[0.2]['spacing_median']['learned'])

  with open(os.path.join(args.out, 'summary.json'), 'w') as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False, default=str)
  print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
  main()
