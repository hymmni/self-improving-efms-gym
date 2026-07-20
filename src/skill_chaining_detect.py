"""스킬 체이닝 1단계 — σ² 기반 반응형 스킬 경계 탐지 (새 학습 없이 즉시 사용).

배경: 오늘 확인한 대로 σ²(aleatoric)는 "진짜 갈림길"에서 신뢰할 수 있게
상승한다(차단영역 2.45x). 이걸 그대로 옵션(Sutton et al.)의 종료 조건으로
쓴다 — "σ²가 (에피소드 자체 기준) 평소보다 높다"에 진입하면 '갈림길 진입',
다시 평소 수준으로 가라앉으면 '결정 완료'로 보고 에피소드를 하위 스킬
구간으로 자동 분할한다. 예측기는 기존 checkpoints/obstacle/predictor.pkl을
그대로 쓰고 재학습은 전혀 하지 않는다.

임계값은 에피소드마다 자기 자신의 σ² 중앙값 × factor로 적응적으로 정한다
(맵마다 절대 스케일이 다르므로 절대 임계값보다 안정적).

실행:
  python -m src.skill_chaining_detect --episodes 12 --factor 1.5
"""

import argparse
import os

import numpy as np
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.obstacle_env import ObstacleAvoidPoint2D
from src.probe_generic import GenericSTGProbe

CKPT = 'checkpoints/obstacle/predictor.pkl'


def rollout(probe, seed, max_steps=300):
  np.random.seed(seed)
  env = ObstacleAvoidPoint2D()
  ts = env.reset()
  key = jax.random.PRNGKey(seed)
  positions, varis = [env._cur_pos.copy()], []
  step = 0
  while (not env.success()) and step < max_steps:
    obs = ts.observation
    key, sub = jax.random.split(key)
    act_norm, logits = probe._logits_and_act(obs, sub)
    rec = probe._record(step, obs, logits)
    varis.append(rec.variance)
    action = probe.unnormalize_action(act_norm)[0]
    ts = env.step(np.asarray(action, dtype=np.float32))
    positions.append(env._cur_pos.copy())
    step += 1
  # 마지막 관측의 분산도 기록 (positions와 길이 맞춤)
  obs = ts.observation
  key, sub = jax.random.split(key)
  _, logits = probe._logits_and_act(obs, sub)
  varis.append(probe._record(step, obs, logits).variance)
  return (np.array(positions), np.array(varis), env._cur_obstacle,
          env._goal_pos.copy(), env.success())


def _smooth(x, w=5):
  if len(x) < w:
    return x
  k = np.ones(w) / w
  return np.convolve(x, k, mode='same')


def detect_segments(varis, factor, smooth_w=5, hysteresis=0.75, min_gap=5):
  """평활화 + 이력현상(hysteresis) 기반 경계 탐지.

  raw 임계값 하나로 온/오프를 정하면 (a) 값이 임계값 근처에서 떨릴 때 잘게
  여러 번 넘나드는 채터링이 생기고, (b) 에피소드 전체가 요동치면(실패
  에피소드 등) 중앙값 자체가 밀려 올라가 진짜 구조를 놓친다. 그래서:
    1. 이동평균으로 잔노이즈를 죽이고
    2. 진입 임계값(high=median*factor)과 이탈 임계값(low=high*hysteresis,
       더 낮음)을 분리해 한번 켜지면 확실히 가라앉을 때까지 안 꺼지게 하고
    3. min_gap보다 짧은 구간은 잡음으로 보고 병합한다
  """
  s = _smooth(np.asarray(varis, dtype=np.float64), smooth_w)
  high = float(np.median(s)) * factor
  low = high * hysteresis

  state = s[0] > high
  entry_step = 0 if state else None
  boundaries, entries = [], []
  for t in range(1, len(s)):
    if state and s[t] < low:
      boundaries.append(t)
      entries.append(entry_step if entry_step is not None else 0)
      state = False
      entry_step = None
    elif not state and s[t] > high:
      state = True
      entry_step = t
  # 너무 짧은 구간(직전 경계와 min_gap 미만) 제거 (entries도 짝 맞춰 필터)
  filtered_b, filtered_e = [], []
  last = 0
  for b, e in zip(boundaries, entries):
    if b - last >= min_gap:
      filtered_b.append(b)
      filtered_e.append(e)
      last = b
  return filtered_b, high, filtered_e


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT)
  ap.add_argument('--episodes', type=int, default=12)
  ap.add_argument('--factor', type=float, default=1.5,
                  help='진입 임계값 = 에피소드 자체 σ² 중앙값 × factor')
  ap.add_argument('--hysteresis', type=float, default=0.75,
                  help='이탈 임계값 = 진입 임계값 × hysteresis (더 낮음)')
  ap.add_argument('--smooth', type=int, default=5, help='이동평균 폭')
  ap.add_argument('--min-gap', type=int, default=5,
                  help='이보다 짧은 구간은 잡음으로 보고 병합')
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--out', default='results/obstacle_env/skill_chaining.png')
  args = ap.parse_args()

  probe = GenericSTGProbe(args.checkpoint)

  episodes = []
  for k in range(args.episodes):
    pos, varis, obst, goal, succ = rollout(probe, args.seed0 + k)
    if len(pos) < 10:
      continue
    boundaries, thresh, _entries = detect_segments(
        varis, args.factor, args.smooth, args.hysteresis, args.min_gap)
    episodes.append(dict(pos=pos, varis=varis, obst=obst, goal=goal,
                         succ=succ, boundaries=boundaries, thresh=thresh,
                         seed=args.seed0 + k))

  n_show = min(6, len(episodes))
  cmap = get_cmap('tab10')

  fig, axes = plt.subplots(2, n_show, figsize=(4.3 * n_show, 8.6))
  for i in range(n_show):
    ep = episodes[i]
    pos, varis, boundaries = ep['pos'], ep['varis'], ep['boundaries']
    seg_ids = np.zeros(len(pos), dtype=int)
    b_full = [0] + boundaries + [len(pos)]
    for s, (lo, hi) in enumerate(zip(b_full[:-1], b_full[1:])):
      seg_ids[lo:hi] = s
    n_seg = seg_ids.max() + 1

    ax = axes[0, i]
    ax.add_patch(patches.Circle(ep['obst'][0], ep['obst'][1], facecolor='gray',
                                edgecolor='black', alpha=0.4))
    ax.scatter(*ep['goal'], marker='*', s=160, color='orange', zorder=5)
    for s in range(n_seg):
      m = seg_ids == s
      ax.plot(pos[m, 0], pos[m, 1], '.-', color=cmap(s % 10), ms=3, lw=1.3,
              label=f'스킬 {s+1}' if i == 0 else None)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.set_title(f'seed {ep["seed"]}: {n_seg}개 구간 검출 '
                 f'({"성공" if ep["succ"] else "실패"})', fontsize=10)
    if i == 0:
      ax.legend(fontsize=7, loc='lower left')

    ax2 = axes[1, i]
    ax2.plot(varis, color='C3', lw=1.2)
    ax2.axhline(ep['thresh'], color='gray', ls=':', lw=1.2, label='임계값')
    for b in boundaries:
      ax2.axvline(b, color='black', ls='--', lw=1.0)
    ax2.set_xlabel('step'); ax2.set_ylabel('σ²')
    ax2.set_title('σ² 궤적 (점선=검출된 경계)', fontsize=9)
    if i == 0:
      ax2.legend(fontsize=7)

  fig.suptitle(f'스킬 체이닝 1단계 — σ² 임계값(중앙값×{args.factor}) 넘나듦으로 '
               f'스킬 경계 자동 탐지 (재학습 없음)', fontsize=13)
  fig.tight_layout()
  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(args.out, dpi=130)
  plt.close(fig)

  # ---- 요약 통계 1: 경계가 장애물 근처에서 실제로 찍히는지 (약한 기준 —
  # 이 환경엔 장애물 회피 외에도 골 근처 배회 등 다른 종류의 갈림길도 있어
  # 모든 경계가 장애물 근처일 필요는 없음)
  n_segs = [len(ep['boundaries']) + 1 for ep in episodes]
  near_frac = []
  for ep in episodes:
    obst_c, obst_r = ep['obst']
    for b in ep['boundaries']:
      d = float(np.linalg.norm(ep['pos'][b] - obst_c))
      near_frac.append(d < obst_r + 0.35)
  near_frac = np.array(near_frac)
  print(f'에피소드 {len(episodes)}개, 평균 검출 구간 수: {np.mean(n_segs):.2f} '
        f'(표준편차 {np.std(n_segs):.2f})')
  if len(near_frac):
    print(f'검출된 경계 중 장애물 근처(반경+0.35 이내)에서 찍힌 비율: '
          f'{near_frac.mean():.0%}  ({len(near_frac)}개 경계)')

  # ---- 요약 통계 2 (핵심 검증): "한 번도 해소 안 됨(경계 0개)"이 실패와
  # 연관되는가 — 기존 "분산=실패 예측" 발견과 이 탐지기를 직접 잇는 검증
  never_resolved = np.array([len(ep['boundaries']) == 0 for ep in episodes])
  succ = np.array([ep['succ'] for ep in episodes])
  if (~succ).sum() > 0 and succ.sum() > 0:
    print(f'\n[핵심 검증] "한 번도 해소 안 됨" 비율 — '
          f'성공 에피소드: {never_resolved[succ].mean():.0%} '
          f'({succ.sum()}개)  vs  실패 에피소드: '
          f'{never_resolved[~succ].mean():.0%} ({(~succ).sum()}개)')
  else:
    print(f'\n[핵심 검증] "한 번도 해소 안 됨" 비율: {never_resolved.mean():.0%} '
          f'(성공 {succ.sum()} / 실패 {(~succ).sum()} — 비교하려면 '
          f'--episodes를 늘려 두 그룹을 확보할 것)')
  print(f'-> {args.out}')


if __name__ == '__main__':
  main()
