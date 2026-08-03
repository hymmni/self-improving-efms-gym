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


# ADDED 2026-07-21 (additive; avoid_controller는 그대로 유지)
def avoid_controller_committed(cur_pos, cur_vel, goal_pos, obstacle_rel_pos,
                               obstacle_radius, side, clearance=0.12,
                               angle_jitter=0.0):
  """avoid_controller와 동일하되, 우회 방향을 관측(perp 부호)으로 정하지 않고
  **에피소드 시작에 외부에서 정해진 side(+1/-1)로 고정**한다.

  왜 필요한가 (2026-07-21 실측 근거):
  (1) 원본 avoid_controller는 perp≈0 능선에서 sign이 불연속으로 뒤집혀 퇴화한다.
      노이즈 0으로 굴리면 그 근방에서 100~130스텝씩 기어갔다(재롤아웃 실측:
      노이즈를 넣으면 오히려 중앙값 49~57로 빨라짐). 방향을 미리 커밋하면
      이 불안정이 사라진다.
  (2) 더 중요하게, 원본은 전체 관측의 결정론적 함수라 p(a|s)가 구성상 단봉이다.
      side를 관측에 없는 난수로 두면 같은 관측에 좌·우 액션이 공존해
      **진짜 within-state 다봉성**이 생긴다(믹스처 헤드/다봉 탐지 검증용).
      단, cur_vel이 관측에 있으므로 몇 스텝 뒤엔 속도가 선택을 드러내
      애매함은 에피소드 극초반에만 존재한다(의도된 성질).

  angle_jitter>0이면 접선 방향을 ±jitter(라디안) 내에서 흔들어 각 모드 안에
  자연스러운 폭을 준다(모드 자체는 side로 이진 분리 유지).
  """
  pos = np.asarray(cur_pos, dtype=np.float32)
  goal = np.asarray(goal_pos, dtype=np.float32)
  rel = np.asarray(obstacle_rel_pos, dtype=np.float32)
  d = float(np.linalg.norm(rel))
  radius = float(np.asarray(obstacle_radius).reshape(-1)[0])
  R = radius + clearance

  to_goal = goal - pos
  L = float(np.linalg.norm(to_goal))
  if L < 1e-8:
    return pd_controller(pos, cur_vel, goal)

  if d <= R:
    out = -rel / max(d, 1e-8)
    return pd_controller(pos, cur_vel, pos + out * (R - d + 0.15))

  u = to_goal / L
  proj = float(np.dot(rel, u))
  perp = float(u[0] * rel[1] - u[1] * rel[0])
  blocking = (0.0 < proj < L) and (abs(perp) < R)
  if not blocking:
    return pd_controller(pos, cur_vel, goal)

  theta = float(np.arcsin(min(R / d, 1.0)))
  if angle_jitter > 0.0:
    theta = float(np.clip(theta + np.random.uniform(-angle_jitter, angle_jitter),
                          0.0, np.pi / 2))
  # 원본과 달리 perp 부호를 쓰지 않고 주어진 side로 고정
  tangent_dir = _rot(rel / d, float(side) * theta)
  tangent_len = float(np.sqrt(max(d * d - R * R, 1e-6)))
  subgoal = pos + tangent_dir * tangent_len
  return pd_controller(pos, cur_vel, subgoal)


def demo_action_committed(obs: dict, side, angle_jitter=0.0) -> np.ndarray:
  return avoid_controller_committed(
      obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
      obs['obstacle_rel_pos'], obs['obstacle_radius'],
      side=side, angle_jitter=angle_jitter)


def demo_action(obs: dict) -> np.ndarray:
  """관측 dict를 받아 데모 액션을 내는 편의 래퍼 (데이터 생성/프로브용)."""
  return avoid_controller(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
                          obs['obstacle_rel_pos'], obs['obstacle_radius'])


def demo_action_pf(obs: dict, noise_std=1.5e-4) -> np.ndarray:
  return avoid_controller_pf(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
                             obs['obstacle_rel_pos'], obs['obstacle_radius'],
                             noise_std=noise_std)


# ============================================================= 2-obstacle env
# 2026-07-20 추가: "장애물 2개일 때도 불확실성이 진짜 갈림길에만 반응하는가"
# (예: 두 장애물 사이 좁은 문/게이트) 검증용. 1-obstacle 클래스/체크포인트는
# 건드리지 않고 그대로 additive하게 새 클래스로 추가한다.
class TwoObstacleAvoidPoint2D(EnhancedPoint2D):
  """랜덤 시작/골 + 장애물 항상 2개. 확률 gate_prob로 둘을 마주보게 배치해
  '통과해야 하는 문(게이트)' 형태를 의도적으로 섞는다."""

  def __init__(self,
               radius_range=(0.08, 0.20),
               path_frac_range=(0.30, 0.70),
               perp_jitter_frac=0.5,
               min_start_goal_dist=0.6,
               min_wall_gap=0.25,
               gate_prob=0.4,
               gate_radius_range=(0.08, 0.15),
               gate_gap_range=(0.18, 0.35)):
    super().__init__()
    self._radius_range = radius_range
    self._path_frac_range = path_frac_range
    self._perp_jitter_frac = perp_jitter_frac
    self._min_start_goal_dist = min_start_goal_dist
    self._min_wall_gap = min_wall_gap
    self._gate_prob = gate_prob
    self._gate_radius_range = gate_radius_range
    self._gate_gap_range = gate_gap_range  # 두 원 표면 사이 간격(통과 폭)
    self._cur_obstacles: list = []  # [(center(2,), radius), (center, radius)]

  # ------------------------------------------------------------------ reset
  def reset(self):
    ts = super().reset()
    while (np.linalg.norm(self._cur_pos - self._goal_pos)
           < self._min_start_goal_dist):
      ts = super().reset()

    self._obstacles = {}
    if np.random.uniform() < self._gate_prob:
      self._place_gate()
    else:
      self._place_two_independent()
    return ts._replace(observation=self._augment(ts.observation))

  def _wall_gap(self, center, radius):
    return min(center[0] - BOUNDS_X[0], BOUNDS_X[1] - center[0],
              center[1] - BOUNDS_Y[0], BOUNDS_Y[1] - center[1]) - radius

  def _valid_obstacle(self, center, radius, others):
    start, goal = self._cur_pos, self._goal_pos
    if np.linalg.norm(center - start) <= radius + 0.05:
      return False
    if np.linalg.norm(center - goal) <= radius + self._success_radius + 0.05:
      return False
    if self._wall_gap(center, radius) < self._min_wall_gap:
      return False
    for oc, orad in others:
      if np.linalg.norm(center - oc) < radius + orad + 0.10:  # 서로 안 겹침
        return False
    return True

  def _place_two_independent(self):
    """1-obstacle 배치 로직(ObstacleAvoidPoint2D._place_obstacle)을 두 번
    반복하되, 서로 겹치지 않도록 상호 거리도 확인한다."""
    start, goal = self._cur_pos, self._goal_pos
    seg = goal - start
    seg_len = float(np.linalg.norm(seg))
    direction = seg / seg_len
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    placed = []
    for _ in range(2):
      radius = float(np.random.uniform(*self._radius_range))
      for _ in range(20):
        frac = float(np.random.uniform(*self._path_frac_range))
        offset = (float(np.random.uniform(-1.0, 1.0))
                 * self._perp_jitter_frac * radius)
        center = (start + frac * seg + offset * perp).astype(np.float32)
        if self._valid_obstacle(center, radius, placed):
          break
        radius *= 0.8
      placed.append((center, radius))
    self._cur_obstacles = placed
    for c, r in placed:
      self.add_obstacle(c, r)

  def _place_gate(self):
    """직선 경로 위 한 지점에서 좌우로 원 두 개를 마주보게 배치해, 그 틈을
    통과해야 하는 '문' 형태를 만든다. 통과 폭(gap)은 표면 기준으로 고정."""
    start, goal = self._cur_pos, self._goal_pos
    seg = goal - start
    seg_len = float(np.linalg.norm(seg))
    direction = seg / seg_len
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    for _ in range(20):
      frac = float(np.random.uniform(*self._path_frac_range))
      point = start + frac * seg
      r0 = float(np.random.uniform(*self._gate_radius_range))
      r1 = float(np.random.uniform(*self._gate_radius_range))
      gap = float(np.random.uniform(*self._gate_gap_range))
      off0 = r0 + gap / 2.0
      off1 = r1 + gap / 2.0
      c0 = (point + off0 * perp).astype(np.float32)
      c1 = (point - off1 * perp).astype(np.float32)
      if (self._valid_obstacle(c0, r0, [(c1, r1)]) and
          self._valid_obstacle(c1, r1, [(c0, r0)])):
        self._cur_obstacles = [(c0, r0), (c1, r1)]
        self.add_obstacle(c0, r0)
        self.add_obstacle(c1, r1)
        return
    # 20회 안에 못 만들면(맵 구석 등) 독립 배치로 폴백
    self._place_two_independent()

  # ------------------------------------------------------------------ obs
  def _augment(self, obs: dict) -> dict:
    """장애물을 배치 시점에 정해진 순서 그대로 관측에 담는다(에피소드 내내
    고정, 매 스텝 재정렬하지 않음).

    2026-07-20 수정: 원래는 매 스텝 '현재 위치 기준 가까운 순'으로 재정렬
    했었는데, 두 장애물이 마주보는 게이트 장면에서 에이전트가 등거리선을
    지날 때마다 슬롯 배정이 뒤바뀌어 물리와 무관한 인위적 불연속을
    만드는 게 실측으로 확인됐다(장애물에서 먼 지점에서도 그 등거리선을
    넘는 순간 σ²가 튀었음 — 정렬 아티팩트였지 진짜 불확실성 신호가
    아니었다). 배치 시점에 고정하면 한 에피소드 안에서는 슬롯이 절대
    안 바뀌므로 이 불연속이 사라진다.
    """
    obs = dict(obs)
    pos = self._cur_pos
    rel = np.concatenate([(c - pos).astype(np.float32)
                          for c, _ in self._cur_obstacles])
    radii = np.array([r for _, r in self._cur_obstacles], dtype=np.float32)
    obs['obstacle_rel_pos'] = rel      # (4,) = [dx0,dy0,dx1,dy1]
    obs['obstacle_radius'] = radii     # (2,)
    return obs

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
    spec['obstacle_rel_pos'] = specs.Array((4,), dtype=np.float32)
    spec['obstacle_radius'] = specs.Array((2,), dtype=np.float32)
    return spec


def avoid_controller_pf_two(cur_pos, cur_vel, goal_pos, obstacle_rel_pos,
                            obstacle_radius,
                            k_rep=4e-4, influence_margin=0.30, k_tan=1.5,
                            noise_std=1.5e-4, min_margin=0.04):
  """avoid_controller_pf를 장애물 2개에 대해 합산 적용.

  게이트(두 장애물이 마주보는 좁은 틈)에서 두 영향권(각 0.30)이 겹치면 양쪽
  반발+접선 힘이 동시에 발동해 서로 부딪혀 제자리에서 엉키는 교착이
  실측됐다(단일 장애물 정면 교착과 같은 종류). 해법: 각 장애물의 유효
  영향권을 '다른 장애물 표면까지 남은 간격의 절반'으로 자동 제한한다 —
  틈이 넓으면 원래 margin 그대로, 좁으면 둘 다 줄어들어 한쪽씩만 우세하게
  작동하거나 약하게라도 공존한다.
  """
  act = pd_controller(cur_pos, cur_vel, goal_pos)
  rel = np.asarray(obstacle_rel_pos, dtype=np.float32).reshape(2, 2)
  radii = np.asarray(obstacle_radius, dtype=np.float32).reshape(2)

  centers = np.asarray(cur_pos, dtype=np.float32) + rel  # (2,2) 절대좌표
  gap_between = (float(np.linalg.norm(centers[0] - centers[1]))
                - radii[0] - radii[1])
  eff_margin = min(influence_margin, max(min_margin, gap_between / 2.0))

  for i in range(2):
    d_vec = -rel[i]  # 장애물중심 -> 나
    dist = float(np.linalg.norm(d_vec))
    radius = float(radii[i])
    gap = dist - radius
    if dist > 1e-8 and gap < eff_margin:
      radial = d_vec / dist
      ramp = 1.0 - max(gap, 0.0) / eff_margin
      to_goal = np.asarray(goal_pos, dtype=np.float32) - (
          np.asarray(cur_pos, dtype=np.float32) + rel[i])
      side = float(np.sign(to_goal[0] * d_vec[1] - to_goal[1] * d_vec[0])) \
          or 1.0
      tangential = np.array([-radial[1], radial[0]], dtype=np.float32) * side
      act = act + k_rep * ramp * (radial + k_tan * tangential)

  if noise_std > 0:
    act = act + np.random.normal(0.0, noise_std, size=2).astype(np.float32)
  return act


def demo_action_pf_two(obs: dict, noise_std=1.5e-4) -> np.ndarray:
  return avoid_controller_pf_two(obs['cur_pos'], obs['cur_vel'],
                                 obs['goal_pos'], obs['obstacle_rel_pos'],
                                 obs['obstacle_radius'], noise_std=noise_std)


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


def _render_demo_grid_two(action_fn, title, out_path,
                          n_episodes=8, max_steps=500, seed0=0):
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
    env = TwoObstacleAvoidPoint2D()
    ts = env.reset()
    traj = [env._cur_pos.copy()]
    step = 0
    while not env.success() and step < max_steps:
      ts = env.step(action_fn(ts.observation))
      traj.append(env._cur_pos.copy())
      step += 1
    traj = np.array(traj)
    lengths.append(step)

    straight = np.linalg.norm(traj[0] - env._goal_pos)
    for c, r in env._cur_obstacles:
      ax.add_patch(patches.Circle(c, r, facecolor='gray', edgecolor='black',
                                  alpha=0.5))
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


# ============================================================ partial-observ.
# ADDED 2026-07-21. 배경(실측 근거):
#   완전관측 + 결정론적 동역학 + 결정론적 전문가에서는 지속적 다봉성이
#   원리적으로 불가능하다 — 어느 쪽을 골랐는지가 내 위치/속도에 즉시 기록되기
#   때문. 실제로 랜덤 커밋을 주입해도 t=1에 속도가 선택을 96% 노출했고(t=0만
#   0.512), 애매 구간이 전체의 2.4%뿐이라 믹스처가 붕괴했다.
#   그래서 "관측이 원리적으로 해소할 수 없는" 불확실성이 필요하다: 장애물을
#   센싱 반경 안에서만 보이게 하고, 장애물이 경로를 막을지 여부 자체를
#   확률적으로 만든다. 그러면 초반 STG가 진짜 이봉(직진 vs 우회)이 되고,
#   그 불확실성은 관측에 정보가 없어 기하학으로 재구성할 수 없다.
class PartialObsObstacleAvoidPoint2D(ObstacleAvoidPoint2D):
  """센싱 반경 밖의 장애물은 관측되지 않는 부분관측 버전.

  관측 필드는 부모와 같되 obstacle_visible(1,)이 추가된다. 안 보일 때는
  obstacle_rel_pos=(0,0), obstacle_radius=0, obstacle_visible=0으로 마스킹한다
  (플래그가 있어야 '원점에 반경0 장애물'과 '안 보임'이 구분된다).

  block_prob 확률로만 장애물을 경로 위에 놓고, 나머지는 경로에서 확실히 벗어난
  곳에 놓는다 — "막힐까 안 막힐까"가 관측 불가능한 진짜 이봉을 만들기 위함.
  """

  def __init__(self, sensing_radius=0.40, block_prob=0.5, **kwargs):
    super().__init__(**kwargs)
    self._sensing_radius = sensing_radius
    self._block_prob = block_prob
    self._is_blocking_episode = True

  def _place_obstacle(self):
    self._is_blocking_episode = bool(np.random.uniform() < self._block_prob)
    if self._is_blocking_episode:
      return super()._place_obstacle()      # 기존 로직: 경로 위에 배치
    # 비차단: 시작-골 직선에서 확실히 벗어나게 (수직거리 > radius+clearance)
    start, goal = self._cur_pos, self._goal_pos
    seg = goal - start
    seg_len = float(np.linalg.norm(seg))
    direction = seg / seg_len
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    for _ in range(40):
      radius = float(np.random.uniform(*self._radius_range))
      frac = float(np.random.uniform(*self._path_frac_range))
      # 수직으로 크게 밀어 확실히 비차단
      offset = float(np.random.choice([-1.0, 1.0])) * float(
          np.random.uniform(radius + 0.20, radius + 0.55))
      center = (start + frac * seg + offset * perp).astype(np.float32)
      wall_gap = min(
          center[0] - BOUNDS_X[0], BOUNDS_X[1] - center[0],
          center[1] - BOUNDS_Y[0], BOUNDS_Y[1] - center[1]) - radius
      if (np.linalg.norm(center - start) > radius + 0.05 and
          np.linalg.norm(center - goal) > radius + self._success_radius + 0.05
          and wall_gap >= self._min_wall_gap):
        break
    self._cur_obstacle = (center, radius)
    self.add_obstacle(center, radius)

  def _visible(self) -> bool:
    center, radius = self._cur_obstacle
    return bool(np.linalg.norm(center - self._cur_pos) - radius
                < self._sensing_radius)

  def _augment(self, obs: dict) -> dict:
    center, radius = self._cur_obstacle
    obs = dict(obs)
    if self._visible():
      obs['obstacle_rel_pos'] = (center - self._cur_pos).astype(np.float32)
      obs['obstacle_radius'] = np.array([radius], dtype=np.float32)
      obs['obstacle_visible'] = np.array([1.0], dtype=np.float32)
    else:
      obs['obstacle_rel_pos'] = np.zeros(2, dtype=np.float32)
      obs['obstacle_radius'] = np.zeros(1, dtype=np.float32)
      obs['obstacle_visible'] = np.array([0.0], dtype=np.float32)
    return obs

  def observation_spec(self):
    spec = dict(super().observation_spec())
    spec['obstacle_visible'] = specs.Array((1,), dtype=np.float32)
    return spec


def demo_action_partial(obs: dict) -> np.ndarray:
  """부분관측용 전문가: 장애물이 보일 때만 회피, 안 보이면 목표로 직진.

  전문가도 학습자와 똑같이 관측된 정보만 쓴다(특권 정보 없음) — 그래야 BC가
  well-posed하다. 전문가가 안 보이는 장애물을 피해버리면 학습자는 관측에 없는
  근거로 행동해야 해서 문제가 성립하지 않는다.
  """
  if float(np.asarray(obs['obstacle_visible']).reshape(-1)[0]) < 0.5:
    return pd_controller(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'])
  return avoid_controller(obs['cur_pos'], obs['cur_vel'], obs['goal_pos'],
                          obs['obstacle_rel_pos'], obs['obstacle_radius'])
