"""관측 가능한 장애물 회피 환경 (phase 3: 인위적 피크 없는 gym 재구성).

배경(2026-07-16 논의): 기존 데모의 웨이포인트 5개 경유는 steps-to-go 분포에
~14 step 간격의 인위적 다봉성을 만드는 원천이었다(관측 불가능한 "남은 웨이포인트
수"가 라벨을 가름). 이 환경은 웨이포인트를 없애고, 대신 **관측 가능한** 장애물
하나가 만드는 자연스러운 우회를 유일한 비효율 원천으로 삼는다 — 이후 나타나는
다봉성/분산 패턴은 전부 관측값으로 설명 가능해야 한다.

설계 (사용자 합의 스펙):
  - 시작/골: 매 에피소드 랜덤 (원본과 동일, 단 최소 거리 보장 — 아래 참조)
  - 웨이포인트: 없음
  - 장애물: 1개. 시작->골 선분의 30~70% 지점, 선분 수직 방향으로 반경 이내
    지터(항상 직선 경로를 막도록), 반경 U(0.1, 0.25)
  - 관측: cur_pos, cur_vel, goal_pos + obstacle_rel_pos(장애물중심-현재위치),
    obstacle_radius — 5필드
  - 데모 컨트롤러: pd_controller + 포텐셜필드 반발항(+ 약한 접선 성분)

원본 대비 의도된 편차:
  1. 시작-골 최소 거리 0.6 보장(그보다 가까우면 장애물이 시작/골과 겹쳐 회피
     문제가 성립하지 않음).
  2. 맵 경계 [-1,1]^2 가 물리적 벽 — 원본은 무한 평면이라 에이전트가 화면
     밖으로 나갔다 돌아올 수 있는데(노이즈 데모에서 빈번), 여기선 장애물
     충돌과 동일한 의미론(표면에 고정 + 해당 축 속도 0)으로 막는다. 벽 위치는
     cur_pos로 관측 가능하므로 "관측으로 설명 가능한 비효율" 원칙과 정합.
물리/성공판정은 EnhancedPoint2D(=Point2D + 벽형 장애물 충돌)를 그대로 상속한다.
"""

from typing import Optional

import numpy as np
from dm_env import specs

# From: pointmass_core.pd_controller (notebook cell 6) — 그대로 재사용
from pointmass_core import pd_controller, BOUNDS_X, BOUNDS_Y
from envs_enhanced import EnhancedPoint2D


class ObstacleAvoidPoint2D(EnhancedPoint2D):
  """랜덤 시작/골 + 경로를 막는 관측 가능한 원형 장애물 1개."""

  def __init__(self,
               radius_range=(0.10, 0.25),
               path_frac_range=(0.30, 0.70),
               perp_jitter_frac=0.5,
               min_start_goal_dist=0.6,
               min_wall_gap=0.25):
    super().__init__()
    self._radius_range = radius_range
    self._path_frac_range = path_frac_range
    self._perp_jitter_frac = perp_jitter_frac  # 수직 지터 = frac * radius 이내
    self._min_start_goal_dist = min_start_goal_dist
    # 장애물 표면과 벽 사이 최소 통로 폭 — 벽이 물리벽이 된 뒤 "벽-장애물
    # 구석" 함정(교착)이 생기는 것을 배치 단계에서 차단한다
    self._min_wall_gap = min_wall_gap
    self._cur_obstacle: Optional[tuple] = None  # (center(2,), radius)

  # ------------------------------------------------------------------ reset
  def reset(self):
    # 시작-골이 너무 가까우면 장애물이 양끝과 겹치므로 재샘플 (의도된 편차)
    ts = super().reset()
    while (np.linalg.norm(self._cur_pos - self._goal_pos)
           < self._min_start_goal_dist):
      ts = super().reset()

    # EnhancedPoint2D는 장애물을 에피소드 간 유지하므로(개입 지속 설계) 직접 비움
    self._obstacles = {}
    self._place_obstacle()
    return ts._replace(observation=self._augment(ts.observation))

  def _place_obstacle(self):
    start, goal = self._cur_pos, self._goal_pos
    seg = goal - start
    seg_len = float(np.linalg.norm(seg))
    direction = seg / seg_len
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    for _ in range(20):
      radius = float(np.random.uniform(*self._radius_range))
      frac = float(np.random.uniform(*self._path_frac_range))
      offset = float(np.random.uniform(-1.0, 1.0)) * self._perp_jitter_frac * radius
      center = (start + frac * seg + offset * perp).astype(np.float32)
      wall_gap = min(
          center[0] - BOUNDS_X[0], BOUNDS_X[1] - center[0],
          center[1] - BOUNDS_Y[0], BOUNDS_Y[1] - center[1]) - radius
      # 시작/골을 삼키지 않고(골은 성공반경까지 여유), 벽과의 통로도 확보
      if (np.linalg.norm(center - start) > radius + 0.05 and
          np.linalg.norm(center - goal) > radius + self._success_radius + 0.05
          and wall_gap >= self._min_wall_gap):
        break
      radius *= 0.8  # 좁은 배치면 줄여서라도 성립시킴
    self._cur_obstacle = (center, radius)
    self.add_obstacle(center, radius)  # 물리 충돌은 부모 훅 재사용

  # ------------------------------------------------------------------ obs
  def _augment(self, obs: dict) -> dict:
    center, radius = self._cur_obstacle
    obs = dict(obs)
    obs['obstacle_rel_pos'] = (center - self._cur_pos).astype(np.float32)
    obs['obstacle_radius'] = np.array([radius], dtype=np.float32)
    return obs

  # 부모의 물리 substep 루프가 장애물 충돌 처리를 위해 매 substep 호출하는
  # 훅 — 여기에 맵 경계 벽을 같은 의미론(표면 고정 + 속도 0)으로 얹는다.
  def _resolve_obstacle_collisions(self) -> None:
    super()._resolve_obstacle_collisions()
    for axis, (lo, hi) in enumerate([BOUNDS_X, BOUNDS_Y]):
      if self._cur_pos[axis] < lo:
        self._cur_pos[axis] = lo
        self._cur_vel[axis] = 0.0
      elif self._cur_pos[axis] > hi:
        self._cur_pos[axis] = hi
        self._cur_vel[axis] = 0.0

  def step(self, action):
    ts = super().step(action)
    return ts._replace(observation=self._augment(ts.observation))

  def observation_spec(self):
    spec = dict(super().observation_spec())
    spec['obstacle_rel_pos'] = specs.Array((2,), dtype=np.float32)
    spec['obstacle_radius'] = specs.Array((1,), dtype=np.float32)
    return spec


# ---------------------------------------------------------------- controller
def _rot(v, angle):
  c, s = np.cos(angle), np.sin(angle)
  return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def avoid_controller(cur_pos, cur_vel, goal_pos, obstacle_rel_pos,
                     obstacle_radius, clearance=0.12):
  """접선점 조준(tangent steering): 골로 가는 시야선이 (여유 반경만큼 부풀린)
  장애물에 막히면, PD의 조준점을 골 대신 장애물 가장자리의 접선점으로 옮긴다.
  막히지 않으면 평범한 골 추종 PD와 동일.

  포텐셜필드(반발력 합성) 방식은 정면 기하에서 PD와 상쇄되는 평형점 교착과,
  도는 방향의 매 스텝 뒤집힘(매듭 진동)을 실측으로 보여 폐기했다. 이 방식은
  힘을 섞지 않고 조준점만 바꾸므로 교착이 없고, 도는 방향도 "장애물이 시야선의
  어느 쪽에 있나"로 정해져 우회 중 자연스럽게 안정된다. 여전히 현재 관측만
  쓰는 국소 규칙이며 모터 프리미티브는 pd_controller 하나다.
  """
  pos = np.asarray(cur_pos, dtype=np.float32)
  goal = np.asarray(goal_pos, dtype=np.float32)
  rel = np.asarray(obstacle_rel_pos, dtype=np.float32)  # 나 -> 장애물중심
  d = float(np.linalg.norm(rel))
  radius = float(np.asarray(obstacle_radius).reshape(-1)[0])
  R = radius + clearance

  to_goal = goal - pos
  L = float(np.linalg.norm(to_goal))
  if L < 1e-8:
    return pd_controller(pos, cur_vel, goal)

  if d <= R:
    # 부풀린 경계 안쪽(스폰 직후 등): 일단 바깥으로 밀어내는 조준점
    out = -rel / max(d, 1e-8)
    return pd_controller(pos, cur_vel, pos + out * (R - d + 0.15))

  u = to_goal / L
  proj = float(np.dot(rel, u))                    # 시야선 위 장애물 투영 거리
  perp = float(u[0] * rel[1] - u[1] * rel[0])     # 시야선 기준 부호 있는 수직거리
  blocking = (0.0 < proj < L) and (abs(perp) < R)
  if not blocking:
    return pd_controller(pos, cur_vel, goal)

  # 접선점: 나->중심 방향을 ±theta 회전 (theta = asin(R/d)), 장애물이 시야선
  # 왼쪽(perp>0)에 있으면 오른쪽 접선(-theta)으로 돈다
  theta = float(np.arcsin(min(R / d, 1.0)))
  sign = -1.0 if perp > 0 else 1.0
  tangent_dir = _rot(rel / d, sign * theta)
  tangent_len = float(np.sqrt(max(d * d - R * R, 1e-6)))
  subgoal = pos + tangent_dir * tangent_len
  return pd_controller(pos, cur_vel, subgoal)


def avoid_controller_pf(cur_pos, cur_vel, goal_pos, obstacle_rel_pos,
                        obstacle_radius,
                        k_rep=4e-4, influence_margin=0.30, k_tan=1.5,
                        noise_std=1.5e-4):
  """포텐셜필드(선형 램프 반발 + 접선) + 가우시안 액션 노이즈.

  순수 포텐셜필드는 정면 평형점 교착/회전방향 뒤집힘 진동을 보였다(실측).
  노이즈는 (1) 평형점을 흔들어 깨고 (2) 데모에 자연스러운 비효율/다양성을
  심는 목적(사용자 제안). PD의 감쇠항(Kd*(-v))이 매 스텝 유지되므로 phase-1
  random_act에서 봤던 속도 폭주는 일어나지 않는다.
  """
  act = pd_controller(cur_pos, cur_vel, goal_pos)

  d_vec = -np.asarray(obstacle_rel_pos, dtype=np.float32)  # 장애물중심 -> 나
  dist = float(np.linalg.norm(d_vec))
  radius = float(np.asarray(obstacle_radius).reshape(-1)[0])
  gap = dist - radius
  if dist > 1e-8 and gap < influence_margin:
    radial = d_vec / dist
    ramp = 1.0 - max(gap, 0.0) / influence_margin
    to_goal = np.asarray(goal_pos, dtype=np.float32) - (
        np.asarray(cur_pos, dtype=np.float32)
        + np.asarray(obstacle_rel_pos))
    side = float(np.sign(to_goal[0] * d_vec[1] - to_goal[1] * d_vec[0])) or 1.0
    tangential = np.array([-radial[1], radial[0]], dtype=np.float32) * side
    act = act + k_rep * ramp * (radial + k_tan * tangential)

  if noise_std > 0:
    act = act + np.random.normal(0.0, noise_std, size=2).astype(np.float32)
  return act


def demo_action(obs: dict) -> np.ndarray:
  """관측 dict를 받아 데모 액션을 내는 편의 래퍼 (데이터 생성/프로브용)."""
  return avoid_controller(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
                          obs['obstacle_rel_pos'], obs['obstacle_radius'])


def demo_action_pf(obs: dict, noise_std=1.5e-4) -> np.ndarray:
  return avoid_controller_pf(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
                             obs['obstacle_rel_pos'], obs['obstacle_radius'],
                             noise_std=noise_std)


# ------------------------------------------------------------------ demo viz
def _render_demo_grid(action_fn, title, out_path,
                      n_episodes=8, max_steps=300, seed0=0):
  import os
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False

  fig, axes = plt.subplots(2, 4, figsize=(18, 9))
  lengths = []
  for k, ax in enumerate(axes.flat):
    np.random.seed(seed0 + k)
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    traj = [env._cur_pos.copy()]
    step = 0
    while not env.success() and step < max_steps:
      ts = env.step(action_fn(ts.observation))
      traj.append(env._cur_pos.copy())
      step += 1
    traj = np.array(traj)
    lengths.append(step)

    center, radius = env._cur_obstacle
    straight = np.linalg.norm(traj[0] - env._goal_pos)
    ax.add_patch(patches.Circle(center, radius, facecolor='gray',
                                edgecolor='black', alpha=0.5))
    ax.plot(traj[:, 0], traj[:, 1], '.-', color='blue', ms=3, lw=1)
    ax.scatter(*traj[0], marker='s', s=70, color='green', zorder=5, label='시작')
    ax.scatter(*env._goal_pos, marker='*', s=160, color='orange', zorder=5,
               label='골')
    ax.add_patch(patches.Circle(env._goal_pos, env._success_radius,
                                edgecolor='green', ls='--', fill=False))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.set_title(f'seed {seed0 + k}: {step} steps '
                 f'(직선거리 {straight:.2f}, 성공={env.success()})', fontsize=10)
    if k == 0:
      ax.legend(fontsize=8, loc='lower left')
  fig.suptitle(title, fontsize=14)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f'{title}\n  에피소드 길이: {lengths}  ->  {out_path}')
  return lengths


if __name__ == '__main__':
  _render_demo_grid(demo_action, '장애물 회피 데모 (접선점 조준)',
                    'results/obstacle_env/demo_trajs.png')
  for std in (1e-4, 2e-4):
    _render_demo_grid(
        lambda obs, s=std: demo_action_pf(obs, noise_std=s),
        f'장애물 회피 데모 (포텐셜필드 + 노이즈 std={std:g})',
        f'results/obstacle_env/demo_trajs_pf_noise{std:g}.png')
