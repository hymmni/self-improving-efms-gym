r"""GraspCarry2D 롤아웃 영상 녹화 (phase 3, step 4).

같은 시드(= 같은 블록, 같은 은닉 물성)를 **느린 속도 vs 빠른 속도**로 나란히
굴려서 "속도가 곧 위험"과 "은닉 물성이 결과를 가른다"를 한 화면에 보인다.

렌더러는 단순 시각화가 아니라 **물리가 의도대로 동작하는지 확인하는 검증
도구**다. 그리퍼 다각형·블록 정점·박스 기하는 전부 `env`/`env.gripper`가
노출하는 값을 그대로 쓰고, 렌더러에서 다시 계산하지 않는다.

  python -m grasp_carry.scripts.record.record_carry --seeds 3 7 11 --speeds 30 60 \
      --out results/videos/grasp_carry.mp4
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

plt.rcParams['axes.unicode_minus'] = False

from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D
from grasp_carry.policy import ScriptedCarryPolicy

# 작업 영역 크롭의 기본 상단(y, mm) — 정상 속도 범위(<=60mm/스텝)에서 실측한
# EE 최저점(~182mm)보다 넉넉히 낮다(=화면 위쪽으로 더 크게 잡는다). 4속도(20/
# 30/60/90) x 60시드 스윕에서 speed<=60은 전부 182mm 이상에 머물렀고, speed=90
# 에서만 PD 오버슈트로 116mm까지 내려갔다(`_view_limits`가 이 경우를 동적으로
# 확장해 처리한다).
_DEFAULT_TOP_Y = 130.0
# 동적 확장 시 그리퍼 다각형 최상단에 남기는 여유(mm).
_VIEW_MARGIN = 15.0
# 바닥 아래로 남기는 여유(mm) — 바닥선이 패널 맨 아래에 딱 붙지 않게 한다.
_FLOOR_MARGIN = 30.0


def _ee_block_dist(env) -> float:
  """EE(그리퍼 베이스)와 블록 중심 사이 거리(mm). 미끄러짐 계산의 원재료."""
  ex, ey, _ = env.gripper.pose
  bx = float(env.block_body.position.x)
  by = float(env.block_body.position.y)
  return math.hypot(bx - ex, by - ey)


def view_limits(env):
  """작업 영역만 잡는 (xlim, ylim). 매 프레임 고정값이다(그리퍼 위치와 무관).

  `+y`가 아래인 세계 좌표라 `ylim`은 (바닥 쪽 큰 값, 위쪽 작은 값) 순으로
  준다. 2026-08-10: 예전엔 그리퍼 다각형이 `_DEFAULT_TOP_Y`/월드 폭을 넘어가면
  (빠른 속도의 PD 오버슈트 등) 그만큼 동적으로 확장해서 프레임 크기가
  매 프레임 달라졌었다 — 영상 가로세로가 흔들리는 게 더 보기 안 좋다는
  피드백으로, 이제 고정 범위만 쓰고 그 밖으로 나가는 그리퍼는 그냥 화면
  밖으로 잘리게 둔다.
  """
  cfg = env.cfg
  return (0.0, cfg.world_width), (cfg.floor_y + _FLOOR_MARGIN, _DEFAULT_TOP_Y)


def draw_env(ax, env, action=None, side_label='', block_color='tab:blue',
            base_offset=None):
  """`env`의 현재 상태 한 프레임을 `ax`에 그린다. `env`를 변경하지 않는다.

  `action`은 이번 스텝에 실제로 인가된 액션(x, y, theta, grip) — 닫힘/열림
  색 구분과 처짐(sag) 오버레이에 쓴다. `base_offset`은 파지 시작 시점의
  EE-블록 거리(mm) — 미끄러짐(=그 이후의 증가분)을 계산하는 기준값이다.
  """
  cfg = env.cfg
  ax.clear()
  xlim, ylim = view_limits(env)
  ax.set_xlim(*xlim); ax.set_ylim(*ylim)
  ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

  info = env._info()
  held = bool(env.is_held())
  closing = action is not None and float(action[3]) > 0.5

  # 바닥
  ax.plot([0.0, cfg.world_width], [cfg.floor_y, cfg.floor_y],
          color='0.35', lw=4, zorder=1)

  # 박스(소스=회색/좁음, 타겟=초록/넓음) — 소스 내폭은 에피소드마다 다르므로
  # 매번 env.src_box에서 읽는다.
  for box, color in ((env.src_box, '0.45'), (env.tgt_box, 'seagreen')):
    ax.plot([box.left_outer, box.right_outer],
            [box.inner_floor_y, box.inner_floor_y], color=color, lw=4, zorder=1)
    for x in (box.left_inner, box.right_inner):
      ax.plot([x, x], [box.rim_y, box.inner_floor_y], color=color, lw=4,
              zorder=1)

  # 그리퍼 — env가 노출하는 다각형을 그대로 그린다. 잡는 중(물림)/닫는 중/
  # 열림을 색으로 구분한다.
  gcol = 'crimson' if held else ('darkorange' if closing else '0.30')
  for poly in env.gripper.polygons():
    ax.add_patch(Polygon(poly, closed=True, facecolor=gcol, edgecolor='k',
                         lw=0.8, alpha=0.95, zorder=2))

  # 블록 — pymunk shape 꼭짓점을 회전·평행이동만 해서 그린다.
  verts = np.array([env.block_body.local_to_world(v)
                    for v in env.block_shape.get_vertices()])
  ax.add_patch(Polygon(verts, closed=True, facecolor=block_color,
                       edgecolor='k', lw=1.2, alpha=0.9, zorder=3))

  if info['outcome'] == 'success':
    state = 'success'
  elif info['outcome'] == 'tipped':
    state = 'tipped'
  elif held:
    state = 'held'
  elif closing:
    state = 'closing'
  else:
    state = 'open'
  ax.set_title(f"{side_label}step {info['steps']}  drops {info['n_drops']}  "
               f"[{state}]", fontsize=9)


def render_frame(env, action=None, side_label='', block_color='tab:blue',
                 base_offset=None, figsize=(4.6, 4.4), dpi=100):
  """`env`의 현재 상태를 `(H, W, 3)` uint8 RGB 배열 프레임으로 렌더링한다.

  `env`를 변경하지 않는다 — 읽기만 한다.
  """
  fig = plt.figure(figsize=figsize, dpi=dpi)
  ax = fig.add_subplot(1, 1, 1)
  draw_env(ax, env, action=action, side_label=side_label,
          block_color=block_color, base_offset=base_offset)
  fig.tight_layout()
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  w, h = fig.get_size_inches() * fig.get_dpi()
  frame = (np.frombuffer(canvas.buffer_rgba(), np.uint8)
          .reshape(int(h), int(w), 4)[..., :3].copy())
  plt.close(fig)
  return frame


def run_pair(seeds, speeds, cfg, fps, out, explore_range=None):
  """같은 시드를 두 속도로 나란히 굴려 mp4로 저장한다.

  `explore_range`를 주면 `speeds`는 라벨에만 쓰이고, 실제 속도는
  `_speed_cap()`의 안전식을 끄고 이 구간에서 매 파지마다 무작위로 뽑는다
  (패널마다 다른 RNG 시드라 같은 블록에서도 결과가 갈린다).
  """
  env_a, env_b = GraspCarry2D(cfg), GraspCarry2D(cfg)
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=110)
  frames = []
  for seed in seeds:
    envs = [env_a, env_b]
    if explore_range is not None:
      policies = [ScriptedCarryPolicy(
          speed=float(speeds[i]), config=cfg, explore_range=explore_range,
          rng=np.random.default_rng(seed * 2 + i)) for i in range(2)]
    else:
      policies = [ScriptedCarryPolicy(speed=float(s), config=cfg)
                  for s in speeds]
    for e, p in zip(envs, policies):
      e.reset(seed=seed); p.reset()
    done = [False, False]
    actions = [None, None]
    base_offset = [None, None]     # 파지 시작 시점의 EE-블록 거리(미끄러짐 기준)
    colors = ('tab:blue', 'tab:red')

    for _ in range(cfg.max_steps):
      for i, (e, p) in enumerate(zip(envs, policies)):
        if done[i]:
          continue
        actions[i] = p(e)
        _, _, term, trunc, _ = e.step(actions[i])
        if term or trunc:
          done[i] = True
      for i, e in enumerate(envs):
        if e.is_held() and base_offset[i] is None:
          base_offset[i] = _ee_block_dist(e)
        elif not e.is_held():
          base_offset[i] = None

      for i, (e, sp) in enumerate(zip(envs, speeds)):
        label = (f'explore[{explore_range[0]:g},{explore_range[1]:g}]   '
                 if explore_range is not None else f'speed={sp:g}   ')
        draw_env(axes[i], e, action=actions[i],
                side_label=label, block_color=colors[i],
                base_offset=base_offset[i])
      info_a = env_a._info()
      fig.suptitle(
          f"[hidden] mass={info_a['mass']:.2f}kg friction={info_a['friction']:.2f}"
          f"   |   [observable] block {env_a.block_w:.0f}x{env_a.block_h:.0f}mm"
          f"  src inner-width {env_a.src_box.inner_width:.0f}mm   (seed {seed})",
          fontsize=10)
      fig.tight_layout(rect=[0, 0, 1, 0.90])
      canvas = FigureCanvasAgg(fig)
      canvas.draw()
      w, h = fig.get_size_inches() * fig.get_dpi()
      frames.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                    .reshape(int(h), int(w), 4)[..., :3].copy())
      if all(done):
        break

    frames += [frames[-1]] * fps      # 에피소드 끝에서 잠시 정지
    outcomes = [e._info()['outcome'] for e in envs]
    steps = [e._info()['steps'] for e in envs]
    print(f'seed {seed}: ' + '  '.join(
        f'speed{sp:g}->{st}스텝/{oc}' for sp, st, oc in
        zip(speeds, steps, outcomes)), flush=True)

  plt.close(fig)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  imageio.mimsave(out, frames, fps=fps)
  print(f'saved {out} ({len(frames)} frames)')


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 11])
  ap.add_argument('--speeds', type=float, nargs=2, default=[28.0, 58.0])
  ap.add_argument('--fps', type=int, default=10)
  ap.add_argument('--out', default='results/videos/grasp_carry.mp4')
  ap.add_argument('--explore-range', type=float, nargs=2, default=None,
                  metavar=('LOW', 'HIGH'),
                  help=('켜면 두 패널 다 안전식을 끄고 이 구간(mm)에서 매 '
                        '파지마다 무작위 속도를 강제한다(패널마다 다른 RNG라 '
                        '같은 블록에서도 결과가 갈린다). --speeds는 라벨에만 '
                        '쓰인다.'))
  args = ap.parse_args(argv)
  cfg = CarryConfig()
  run_pair(args.seeds, args.speeds, cfg, args.fps, args.out,
           explore_range=(tuple(args.explore_range)
                          if args.explore_range else None))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
