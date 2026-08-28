"""스킬 체이닝 1단계(σ² 기반 경계 탐지)를 mp4로 녹화 — 일시정지/되감기로 검토.

왼쪽: 궤적이 자라나며 스킬 구간마다 색이 바뀐다(경계 도달 시점에 정확히
전환). 오른쪽 위: STG 카테고리컬 분포 실시간 막대그래프. 오른쪽 아래: σ²
추이 + 임계값(점선) + 검출된 경계(빨간 세로선, 도달하는 순간 나타남).

--deterministic(기본 on): 학습된 정책의 행동 분포에서 샘플링하지 않고
평균(mode)을 그대로 쓴다. 샘플링 방식은 매 스텝의 작은 무작위성이 누적돼
데모에는 없던 배회 루프를 만든다(실측: 확률샘플링 215스텝/누적이동거리
데모의 8.3배 vs mode 69스텝/1.2배, 데모 원본 25~30스텝과 정합) — 정책이
못 배운 게 아니라 순전히 롤아웃 방식의 산물이었다. --no-deterministic으로
끄면 원래(확률 샘플링) 동작을 볼 수 있다.

  python record_skill_chaining.py                    # 학습된 정책 8 에피소드
  python record_skill_chaining.py --checkpoint checkpoints/obstacle_clean/predictor.pkl
  python record_skill_chaining.py --no-deterministic  # 확률 샘플링 (배회 루프 재현)
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.obstacle_env import ObstacleAvoidPoint2D, demo_action
from src.probe_generic import GenericSTGProbe
from src.skill_chaining_detect import detect_segments, refine_resolve_point
from interactive_control import _write_video

CKPT = 'checkpoints/obstacle/predictor.pkl'


def fig_to_frame(fig):
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
      int(h), int(w), 3)


def rollout_full(probe, seed, max_steps=300, deterministic=True,
                 driver='learned', demo_noise_std=0.0):
  """롤아웃하며 위치·분산·전체 확률분포(probs)·관측을 기록.

  driver='learned'(기본): 학습된 정책이 조종.
    deterministic=True: 행동 분포의 평균(mode) 사용 — 확률 샘플링은 매 스텝
    작은 무작위성이 누적돼 데모에 없던 배회를 만든다(모듈 docstring 참조).
  driver='demo': 원본 데모 컨트롤러(접선점 조준, demo_action)가 조종하고,
    분포/분산은 그대로 주어진 checkpoint의 예측기로 조회 — '데모가 실제로
    어떻게 움직이고 그 순간 예측기는 뭐라고 보는지'를 같이 보기 위함.
    demo_noise_std>0이면 train_obstacle_predictor.py --tangent-noise-std와
    동일하게 demo_action 출력에 가우시안 노이즈를 더해, 그 데이터셋 수집 때
    실제로 섞인 노이즈 크기를 그대로 재현한다.
  """
  np.random.seed(seed)
  env = ObstacleAvoidPoint2D()
  ts = env.reset()
  key = jax.random.PRNGKey(seed)
  positions, varis, probs_list, obs_list = [env._cur_pos.copy()], [], [], []
  step = 0
  while (not env.success()) and step < max_steps:
    obs = ts.observation
    obs_list.append(obs)
    key, sub = jax.random.split(key)
    act_norm, logits = probe._logits_and_act(obs, sub)
    rec = probe._record(step, obs, logits)
    varis.append(rec.variance)
    probs_list.append(rec.probs)
    if driver == 'demo':
      action = np.asarray(demo_action(obs), dtype=np.float32)
      if demo_noise_std > 0:
        action = action + np.random.normal(0, demo_noise_std, size=2).astype(
            np.float32)
    elif deterministic:
      norm = probe.normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None], obs))
      concat = jnp.concatenate([norm[f] for f in probe.obs_fields], axis=-1)
      preds = probe.nets.network.apply(probe.params, concat)
      key, sub = jax.random.split(key)
      act_norm = probe.nets.sample_act_mode(preds.act_dist_params, sub)
      action = probe.unnormalize_action(act_norm)[0]
    else:
      action = probe.unnormalize_action(act_norm)[0]
    ts = env.step(np.asarray(action, dtype=np.float32))
    positions.append(env._cur_pos.copy())
    step += 1
  obs_list.append(ts.observation)
  obs = ts.observation
  key, sub = jax.random.split(key)
  _, logits = probe._logits_and_act(obs, sub)
  rec = probe._record(step, obs, logits)
  varis.append(rec.variance)
  probs_list.append(rec.probs)
  return (np.array(positions), np.array(varis), np.array(probs_list),
          obs_list, env._cur_obstacle, env._goal_pos.copy(), env.success())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT)
  ap.add_argument('--episodes', type=int, default=6)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--factor', type=float, default=1.4)
  ap.add_argument('--hysteresis', type=float, default=0.85)
  ap.add_argument('--smooth', type=int, default=3)
  ap.add_argument('--min-gap', type=int, default=4)
  ap.add_argument('--fps', type=int, default=20)
  ap.add_argument('--dist-xmax', type=int, default=100)
  ap.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                  default=True)
  ap.add_argument('--driver', choices=['learned', 'demo'], default='learned',
                  help='learned=학습된 정책이 조종, demo=원본 데모 컨트롤러가'
                       ' 조종(예측기는 그대로 조회 — 학습 데이터 자체를 보기)')
  ap.add_argument('--demo-noise-std', type=float, default=0.0,
                  help='driver=demo일 때 demo_action에 더할 가우시안 액션'
                       ' 노이즈 표준편차 (그 데이터셋 수집 때 쓴 값과 동일하게)')
  ap.add_argument('--out', default='results/videos/skill_chaining.mp4')
  args = ap.parse_args()

  probe = GenericSTGProbe(args.checkpoint)
  cmap = get_cmap('tab10')

  fig = plt.figure(figsize=(15, 5.6), dpi=100)
  gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15])
  ax = fig.add_subplot(gs[:, 0])
  ax_dist = fig.add_subplot(gs[:, 1])
  ax2 = fig.add_subplot(gs[:, 2])

  if args.driver == 'demo':
    mode_label = ('원본 데모 컨트롤러(접선점 조준)' +
                  (f' + 노이즈 std={args.demo_noise_std:g}'
                   if args.demo_noise_std > 0 else ''))
  else:
    mode_label = '학습된 정책 — ' + ('결정론적(mode)' if args.deterministic
                                  else '확률 샘플링')
  frames = []
  n_success = 0
  for k in range(args.episodes):
    seed = args.seed0 + k
    pos, varis, probs, obs_list, obst, goal, succ = rollout_full(
        probe, seed, deterministic=args.deterministic, driver=args.driver,
        demo_noise_std=args.demo_noise_std)
    boundaries, thresh, entries = detect_segments(
        varis, args.factor, args.smooth, args.hysteresis, args.min_gap)
    boundaries = [max(refine_resolve_point(varis, b), e + 1)
                 for b, e in zip(boundaries, entries)]
    n_success += int(succ)
    T = len(pos)

    ax.clear()
    ax.add_patch(patches.Circle(obst[0], obst[1], facecolor='gray',
                                edgecolor='black', alpha=0.4))
    ax.scatter(*goal, marker='*', s=180, color='orange', zorder=5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')

    ax_dist.clear()
    ax_dist.set_xlim(0, args.dist_xmax)
    ax_dist.set_ylim(0, float(probs.max()) * 1.2 + 1e-6)
    ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')
    dist_bar = ax_dist.bar(probe.bin_vals[:args.dist_xmax], np.zeros(
        min(args.dist_xmax, len(probe.bin_vals))), width=1.0, color='C0')

    ax2.clear()
    ax2.axhline(thresh, color='gray', ls=':', lw=1.2, label='임계값')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, max(varis) * 1.1)
    ax2.set_xlabel('step'); ax2.set_ylabel('σ²')
    ax2.legend(fontsize=8, loc='upper right')

    seg_id = 0
    b_ptr = 0
    seg_start = 0
    line_var, = ax2.plot([], [], color='C3', lw=1.3)
    dot, = ax.plot([], [], 'o', color='red', ms=8, zorder=6)

    for t in range(1, T + 1):
      if b_ptr < len(boundaries) and t - 1 == boundaries[b_ptr]:
        ax2.axvline(boundaries[b_ptr], color='red', ls='--', lw=1.2)
        seg_start = t - 1
        seg_id += 1
        b_ptr += 1

      seg_pos = pos[seg_start:t]
      ax.plot(seg_pos[:, 0], seg_pos[:, 1], '-', color=cmap(seg_id % 10),
              lw=1.6)
      dot.set_data([pos[t - 1, 0]], [pos[t - 1, 1]])
      line_var.set_data(range(t), varis[:t])

      p = probs[t - 1][:args.dist_xmax]
      for rect, h in zip(dist_bar, p):
        rect.set_height(h)

      ax.set_title(f'[{mode_label}] seed {seed}  step {t}/{T}  '
                   f'스킬 구간 {seg_id + 1}  성공 {n_success}/{k + (1 if t==T else 0)}',
                   fontsize=10)
      ax_dist.set_title(f'STG 분포  E={float(np.sum(probe.bin_vals*probs[t-1])):.1f}  '
                        f'σ²={varis[t-1]:.0f}', fontsize=10)
      ax2.set_title('σ² 실시간 추이 (빨강 점선=검출된 스킬 경계)', fontsize=10)
      frames.append(fig_to_frame(fig))

    end_title = (f'[{mode_label}] seed {seed} 종료: '
                f'{"성공" if succ else "실패"} ({T-1} steps, '
                f'{seg_id + 1}개 스킬 구간 검출)')
    ax.set_title(end_title, fontsize=10, fontweight='bold')
    frames += [fig_to_frame(fig)] * args.fps
    print(f'ep{k} (seed {seed}): {"성공" if succ else "실패"} {T-1} steps, '
          f'{seg_id + 1}개 구간  (누적 프레임 {len(frames)})', flush=True)

  plt.close(fig)
  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  out = _write_video(frames, args.out, fps=args.fps)
  print(f'저장: {out}  ({len(frames)} frames, 약 {len(frames)/args.fps:.0f}초)')


if __name__ == '__main__':
  main()
