r"""`GraspCarry2D` — 은닉 물성 기반 파지·운반 환경 (phase 3, step 2).

측면도 2D. 단위계는 `config.py`와 동일(mm / kg / s / mN, **+y가 아래**).

```
      [그리퍼]
        |
   ___  |          ___                _______________
  |   |[블록]     |   |              |               |
  |___|___________|   |              |               |   <- 타겟 박스(넓고 낮음)
  =========================================================  <- 바닥
   ^소스 박스(내폭 랜덤, 깊음)
  |                                                       |  <- 월드 양끝 벽
```

이 환경의 존재 이유는 **관측이 결과를 미결정**하게 만드는 것이다. 블록의 크기와
위치는 보이지만 **질량과 마찰계수는 보이지 않는다**(음료 캔 비유: 채움 정도와
표면 미끄러움은 들어봐야 안다). 따라서 질량·마찰은 관측 벡터에 절대 들어가지
않으며, `info`에만 진단용으로 실린다.

이전 시도(`archive/grasp_carry_env.py` v1~v3)에서 **실측으로** 확인돼 여기 반영된 것:

1. 동적 베이스는 목표를 오버슛한다. 목표 y만 제한하면 그리퍼가 박스 rim 아래로
   파고들어 벽에 낀다. 실제 위치에도 **하드 스톱**(로봇 기구학적 한계)이 필요하다.
2. 성공 판정에 블록 **중심** y를 쓰면 안 된다. 키 큰 블록은 박스에 놓여도 중심이
   rim 위에 온다. 블록 **바닥**이 박스 바닥에 내려앉았는지로 판정한다.
3. 월드 양끝에 벽이 없으면 물체가 화면 밖으로 탈출한다.

**설계상 성립하지 않는 것 하나** — "소스 박스가 좁아서 블록이 박스 안에서는
넘어질 수 없다"는 이 기하에서 불가능하다. 소스 박스 내폭은 깊은 파지를 위해
그리퍼 바깥폭(136mm)을 걸쳐야 하는데 블록 높이는 100~140mm라, 내폭이 블록보다
넓은 에피소드에서는 눕을 자리가 있다(실측: 세운 블록을 세게 치면 기울기 90deg
도달). 보장되는 것은 스폰 상태의 안정성이다 — 로봇이 건드리지 않으면 200스텝
동안 기울기 0.000deg. 박스 안에서 넘어진 것도 "타겟 박스 밖에서 넘어짐"이므로
다른 넘어짐과 동일하게 영구 실패로 처리한다.

정책/컨트롤러(step 3)와 렌더링(step 4)은 여기 없다.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pymunk

from .config import CarryConfig
from .gripper import Gripper

# 관측 한 프레임의 필드 이름 — **질량·마찰은 여기 없다**(그게 이 환경의 설계).
FRAME_FIELDS: Tuple[str, ...] = (
    'ee_x', 'ee_y', 'ee_cos', 'ee_sin', 'ee_gap',
    'block_x', 'block_y', 'block_cos', 'block_sin',
    'block_w', 'block_h',
    'src_inner_width', 'src_center_x',
    'tgt_inner_width', 'tgt_center_x',
)

# 그리퍼가 박스 개구부에 "들어간다"고 볼 때 요구하는 좌우 여유(mm).
_FIT_CLEARANCE = 2.0


@dataclass
class Box:
  """U자 박스(좌벽/바닥/우벽) 기하. 좌표는 전부 월드 mm, +y가 아래."""

  center_x: float
  inner_width: float
  wall_height: float
  wall_thickness: float
  inner_floor_y: float          # 박스 내부 바닥면(블록이 닿는 면)의 y

  @property
  def left_inner(self) -> float:
    return self.center_x - self.inner_width / 2.0

  @property
  def right_inner(self) -> float:
    return self.center_x + self.inner_width / 2.0

  @property
  def left_outer(self) -> float:
    return self.left_inner - self.wall_thickness

  @property
  def right_outer(self) -> float:
    return self.right_inner + self.wall_thickness

  @property
  def rim_y(self) -> float:
    """벽 윗면(rim)의 y. +y가 아래이므로 내부 바닥보다 작다."""
    return self.inner_floor_y - self.wall_height

  def contains_x(self, x: float) -> bool:
    return self.left_inner <= x <= self.right_inner


def _wrap_to_upright(angle: float) -> float:
  """직사각형의 수직축 대비 기울기(rad, 0~pi/2).

  180deg 돌아간 블록도 서 있는 것이므로 주기 pi로 접는다.
  """
  return abs((angle + math.pi / 2.0) % math.pi - math.pi / 2.0)


class GraspCarry2D:
  """소스 박스의 블록을 집어 타겟 박스에 옮기는 파지·운반 환경.

  액션 `(x, y, theta, grip)`:
    x, y   — **절대** 목표 위치(mm, 월드 좌표). 임피던스 PD의 목표점이다.
    theta  — 절대 목표 자세(rad).
    grip   — > 0.5 이면 닫기, 아니면 열기(이진).

  관측은 `FRAME_FIELDS` 프레임을 `cfg.obs_history`개 스택한 1차원 float32
  벡터(오래된 프레임 → 최신 프레임 순)다. 길이는 `world_width`로 정규화한다.
  """

  def __init__(self, config: Optional[CarryConfig] = None):
    self.cfg = config or CarryConfig()
    self.n_substeps = int(round(
        1.0 / (self.cfg.physics_dt * self.cfg.control_hz)))
    self._rng = np.random.default_rng(self.cfg.seed)
    self._history: deque = deque(maxlen=self.cfg.obs_history)

    self.space: Optional[pymunk.Space] = None
    self.gripper: Optional[Gripper] = None
    self.block_body: Optional[pymunk.Body] = None
    self.block_shape: Optional[pymunk.Poly] = None
    self.src_box: Optional[Box] = None
    self.tgt_box: Optional[Box] = None
    self.block_w = 0.0
    self.block_h = 0.0
    # 은닉 물성 — 관측에 절대 넣지 않는다. `info`에만 진단용으로 싣는다.
    self._mass = 0.0
    self._mu = 0.0
    self._t = 0
    self._n_drops = 0
    self._was_held = False
    self._not_held_run = 0
    self._streak_had_grip = False
    self._outcome = 'running'

  # ================================================================ 에피소드
  def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """새 에피소드를 만든다. `seed`를 주면 그 시드로 RNG를 다시 만든다."""
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    cfg = self.cfg
    rng = self._rng

    # --- 에피소드 파라미터 샘플 -------------------------------------------
    self.block_w = float(rng.uniform(*cfg.object_width_range))
    self.block_h = float(rng.uniform(*cfg.object_height_range))
    # 은닉: 채움 정도(질량)와 표면 미끄러움(마찰)
    self._mass = float(rng.uniform(*cfg.object_mass_range))
    self._mu = float(rng.uniform(*cfg.object_friction_range))
    src_inner = float(rng.uniform(*cfg.src_box_width_range))

    self._build_world(src_inner)

    # 블록은 소스 박스 안 랜덤 위치에 세워서 스폰한다.
    lo = self.src_box.left_inner + self.block_w / 2.0 + 2.0
    hi = self.src_box.right_inner - self.block_w / 2.0 - 2.0
    block_x = float(rng.uniform(lo, hi))
    self._spawn_block(block_x)
    self._spawn_gripper()

    self._t = 0
    self._n_drops = 0
    self._was_held = False
    self._not_held_run = 0
    self._streak_had_grip = False
    self._outcome = 'running'

    frame = self.observe_frame()
    self._history.clear()
    for _ in range(cfg.obs_history):
      self._history.append(frame)
    return self._stacked_obs(), self._info()

  # ------------------------------------------------------------------ 월드
  def _build_world(self, src_inner: float) -> None:
    cfg = self.cfg
    space = pymunk.Space()
    space.gravity = (0.0, cfg.gravity)          # +y가 아래
    space.iterations = cfg.solver_iterations
    self.space = space

    inner_floor_y = cfg.floor_y - cfg.box_bottom_thickness
    # 소스 박스는 왼쪽 벽에, 타겟 박스는 오른쪽 벽에 붙인다. 그래야 두 박스
    # 사이 바닥이 최대한 넓어져, 떨어뜨린 블록이 누울 자리가 생긴다.
    src_cx = (cfg.world_margin + cfg.box_wall_thickness + src_inner / 2.0)
    tgt_inner = cfg.tgt_box_width
    tgt_cx = (cfg.world_width - cfg.world_margin - cfg.box_wall_thickness
              - tgt_inner / 2.0)
    self.src_box = Box(src_cx, src_inner, cfg.src_box_wall_height,
                       cfg.box_wall_thickness, inner_floor_y)
    self.tgt_box = Box(tgt_cx, tgt_inner, cfg.tgt_box_wall_height,
                       cfg.box_wall_thickness, inner_floor_y)

    # 바닥 + 월드 양끝 벽. 벽이 없으면 물체가 화면 밖으로 탈출한다.
    # Segment가 아니라 Poly를 쓰는 이유: Segment의 radius가 표면을 안쪽으로
    # 밀어 바닥면 y가 설정값과 어긋난다.
    self._add_static(_rect(0.0, cfg.floor_y, cfg.world_width,
                           cfg.floor_y + 40.0))
    self._add_static(_rect(-40.0, -200.0, 0.0, cfg.floor_y))
    self._add_static(_rect(cfg.world_width, -200.0,
                           cfg.world_width + 40.0, cfg.floor_y))

    for box in (self.src_box, self.tgt_box):
      self._add_static(_rect(box.left_outer, box.inner_floor_y,
                             box.right_outer, cfg.floor_y))          # 바닥판
      self._add_static(_rect(box.left_outer, box.rim_y,
                             box.left_inner, box.inner_floor_y))     # 좌벽
      self._add_static(_rect(box.right_inner, box.rim_y,
                             box.right_outer, box.inner_floor_y))    # 우벽

  def _add_static(self, verts: List[Tuple[float, float]]) -> None:
    shape = pymunk.Poly(self.space.static_body, verts)
    shape.friction = 0.7
    shape.elasticity = 0.0
    self.space.add(shape)

  def _spawn_block(self, x: float) -> None:
    w, h = self.block_w, self.block_h
    moment = pymunk.moment_for_box(self._mass, (w, h))
    body = pymunk.Body(self._mass, moment)
    # 바닥에 닿기 직전 높이에서 세워서 스폰 (angle=0 = 수직)
    body.position = (x, self.src_box.inner_floor_y - h / 2.0 - 0.5)
    shape = pymunk.Poly.create_box(body, (w, h))
    # 은닉 마찰을 물체 shape에 심는다(패드 마찰은 1.0이라 접촉 마찰을 물체가
    # 지배한다 — `cfg.pad_friction` 주석 참고).
    shape.friction = self._mu
    shape.elasticity = 0.0
    self.space.add(body, shape)
    self.block_body = body
    self.block_shape = shape

  def _spawn_gripper(self) -> None:
    cfg = self.cfg
    # 손가락 끝이 rim과 블록 윗면 **둘 다**보다 20mm 위에 오도록 시작한다.
    # (열린 손가락이 블록 옆구리를 치는 것을 막는다 — 실측된 실패 모드)
    block_top = self.src_box.inner_floor_y - self.block_h
    tip_y = min(self.src_box.rim_y, block_top) - 20.0
    base_y = max(20.0, tip_y - cfg.finger_length)
    self.gripper = Gripper(self.space, cfg, (self.src_box.center_x, base_y))

  # ==================================================================== 스텝
  def step(self, action: Sequence[float]) -> Tuple[
      np.ndarray, float, bool, bool, dict]:
    """제어 스텝 1회 = `n_substeps` 물리 substep."""
    cfg = self.cfg
    x, y, theta, grip = (float(v) for v in action)
    closing = grip > 0.5
    half = cfg.gripper_outer_width / 2.0
    # 작업공간 한계(로봇 기구학) — 그리퍼가 월드 벽을 밀지 않게 한다.
    tx = float(np.clip(x, half, cfg.world_width - half))

    for _ in range(self.n_substeps):
      base = self.gripper.base
      # 1) 목표 y를 하강 한계로 클립. 현재 위치와 목표 위치 양쪽의 한계 중
      #    엄한 쪽(작은 y)을 쓴다 — 박스로 진입/이탈하는 중에도 rim 아래로
      #    파고들지 않는다.
      limit = min(self.max_descend_y(base.position.x), self.max_descend_y(tx))
      ty = float(np.clip(y, 0.0, limit))

      # 2) 제어력 인가
      self.gripper.apply_pose_control((tx, ty), theta)
      self.gripper.apply_grip(closing)

      # 3) 물리
      self.space.step(cfg.physics_dt)

      # 4) 하드 스톱: 동적 베이스는 목표를 오버슛하므로, 목표 클립만으로는
      #    rim 아래로 파고드는 것을 못 막는다. 실제 위치를 되돌린다.
      hard = self.max_descend_y(base.position.x)
      if base.position.y > hard:
        base.position = (base.position.x, hard)
        if base.velocity.y > 0.0:
          base.velocity = (base.velocity.x, 0.0)

      # 5) 손가락 대칭 구속(단일 액추에이터) + 속도 제한
      self.gripper.enforce_symmetry()
      self.gripper.clamp_finger_speed()

    self._track_drop()
    self._t += 1
    self._history.append(self.observe_frame())

    terminated = False
    reward = 0.0
    if self._is_success():
      terminated, reward, self._outcome = True, 1.0, 'success'
    elif self.is_tipped() and not self.tgt_box.contains_x(
        float(self.block_body.position.x)):
      # 타겟 박스 **밖**에서 넘어지면 영구 실패. 안에서 넘어진 것은 위의
      # 성공 조건이 먼저 잡는다(자세 무관 성공).
      terminated, self._outcome = True, 'tipped'
    truncated = (not terminated) and self._t >= cfg.max_steps
    if truncated:
      self._outcome = 'timeout'
    return self._stacked_obs(), reward, terminated, truncated, self._info()

  def _track_drop(self) -> None:
    """공중에서 파지가 풀린 횟수를 센다(진단용).

    **제어 스텝 경계에서 2번 연속 안 잡혀야** 센다(스트릭의 두 번째 스텝에서
    1회만 카운트). 1회 시도(제어 스텝당 1회
    평가, substep 루프 밖으로 옮김)로는 부족했다: `enforce_symmetry()`가
    substep마다 손가락 위치를 직접 되밀기 때문에, 그 substep에서 접촉
    재검출이 반대쪽 패드만 놓치는 경우가 실측된다(시드 3: t=29 held →
    t=30 안 잡힘 → t=31 다시 held, 그 사이 블록은 0.005mm도 안 움직였다).
    반면 진짜 낙하는 다음 스텝에도 이어진다(시드 11: t=9~10 연속으로 안
    잡히고 블록이 자유낙하 속도로 떨어진다). 정책의 `_watch_drop`도 이미
    같은 "2연속 제어 스텝" 기준으로 반응(감속·재접근)하므로, 진단 카운터를
    거기 맞추면 정책이 실제로 반응하는 사건과 정확히 같은 것을 센다.
    """
    held = self.is_held()
    if held:
      self._not_held_run = 0
      self._was_held = True
      return
    if self._not_held_run == 0:
      # 안 잡힌 스트릭이 막 시작됐다 — 카운트 여부는 스트릭 시작 **직전**에
      # 실제로 잡고 있었는지로 정한다(처음부터 못 잡은 것과 구분).
      self._streak_had_grip = self._was_held
    self._not_held_run += 1
    if self._not_held_run == 2 and self._streak_had_grip:
      bottom = self._block_bottom_y()
      if bottom < self._support_y(float(self.block_body.position.x)) - 5.0:
        self._n_drops += 1
    self._was_held = False

  # ================================================================== 기구학
  def max_descend_y(self, x: Optional[float] = None) -> float:
    """주어진 x에서 베이스가 내려갈 수 있는 최저(=가장 큰) y.

    손가락 끝(`base_y + finger_length`)이 어떤 바닥면도 뚫지 않는 것이 기준이다.
    그리퍼 바깥폭이 박스 개구부에 **들어가면** 박스 안 바닥까지, 아니면 rim
    위까지만 내려갈 수 있다. 그리퍼의 좌우 span으로 판정하므로, 베이스가 박스
    위에 완전히 오기 전부터 한계가 바뀐다(손가락이 먼저 걸리기 때문).

    `x`가 None이면 현재 베이스 x를 쓴다.
    """
    cfg = self.cfg
    if x is None:
      x = float(self.gripper.base.position.x)
    half = cfg.gripper_outer_width / 2.0
    lo, hi = x - half, x + half
    limit = cfg.floor_y - cfg.finger_length
    for box in (self.src_box, self.tgt_box):
      if hi < box.left_outer or lo > box.right_outer:
        continue
      fits = (box.inner_width >= cfg.gripper_outer_width + 2.0 * _FIT_CLEARANCE
              and lo >= box.left_inner + _FIT_CLEARANCE
              and hi <= box.right_inner - _FIT_CLEARANCE)
      surface = box.inner_floor_y if fits else box.rim_y
      limit = min(limit, surface - cfg.finger_length)
    return float(limit)

  def _support_y(self, x: float) -> float:
    """x 아래의 지지면 y(박스 안이면 박스 바닥, 아니면 월드 바닥)."""
    for box in (self.src_box, self.tgt_box):
      if box.contains_x(x):
        return box.inner_floor_y
    return self.cfg.floor_y

  # ================================================================== 접촉
  def is_held(self) -> bool:
    """양쪽 패드가 블록과 **실제로 접촉 중**인지 pymunk arbiter로 판정한다.

    주의: pymunk `Body`에 `.arbiters` 속성은 없다(실측된 에러). 반드시
    `body.each_arbiter(fn)`을 써야 한다.
    """
    if self.block_body is None:
      return False
    touching = [False, False]
    finger_shapes = self.gripper.finger_shapes

    def visit(arb: pymunk.Arbiter) -> None:
      a, b = arb.shapes
      other = None
      if a is self.block_shape:
        other = b
      elif b is self.block_shape:
        other = a
      if other is None:
        return
      for i, fs in enumerate(finger_shapes):
        if other is fs:
          touching[i] = True

    self.block_body.each_arbiter(visit)
    return touching[0] and touching[1]

  def contact_length(self) -> float:
    """패드와 블록 옆면이 세로로 겹친 길이(mm). 얕은 파지의 정량화에 쓴다.

    패드 span(양 손가락 평균)과 블록의 세로 범위의 교집합 길이다. 블록이
    기울어져 있으면 근사값이다.
    """
    if self.block_body is None:
      return 0.0
    top, bottom = self.gripper.pad_span_y()
    ys = [v.y for v in self._block_world_vertices()]
    overlap = min(bottom, max(ys)) - max(top, min(ys))
    return float(max(0.0, overlap))

  # ================================================================== 종료
  def is_tipped(self) -> bool:
    """블록이 수직에서 `cfg.tip_angle_deg` 이상 기운 채 정지했는지.

    위치는 보지 않는다 — 타겟 박스 **밖**에서 넘어진 것만 영구 실패로 치는
    조합은 `step()`이 한다(박스 안에서 누운 것은 성공).
    """
    if self.block_body is None:
      return False
    tilt = _wrap_to_upright(float(self.block_body.angle))
    return tilt > math.radians(self.cfg.tip_angle_deg) and self._block_at_rest()

  def _is_success(self) -> bool:
    """블록이 타겟 박스 안에 **바닥이 내려앉은 채** 정지 + 미파지.

    **자세는 무관하다** — 누워 있어도 성공이다. 블록 중심 y가 아니라
    **바닥(월드 정점 중 가장 아래)** 으로 판정한다: 키 큰 블록은 박스에
    제대로 놓여도 중심이 rim 위에 온다.
    """
    if self.block_body is None:
      return False
    if not self.tgt_box.contains_x(float(self.block_body.position.x)):
      return False
    if self._block_bottom_y() < self.tgt_box.inner_floor_y - self.cfg.settle_tol:
      return False
    return self._block_at_rest() and not self.is_held()

  def _block_at_rest(self) -> bool:
    cfg = self.cfg
    return (self.block_body.velocity.length < cfg.rest_speed_eps
            and abs(self.block_body.angular_velocity) < cfg.rest_omega_eps)

  def _block_world_vertices(self) -> List[pymunk.Vec2d]:
    return [self.block_body.local_to_world(v)
            for v in self.block_shape.get_vertices()]

  def _block_bottom_y(self) -> float:
    """블록의 가장 아래 점의 y(+y가 아래이므로 최대값)."""
    return float(max(v.y for v in self._block_world_vertices()))

  # ================================================================== 관측
  @property
  def obs_dim(self) -> int:
    return len(FRAME_FIELDS) * self.cfg.obs_history

  def observe_frame(self) -> np.ndarray:
    """관측 한 프레임(`FRAME_FIELDS` 순서, 길이는 world_width로 정규화).

    **질량·마찰은 들어가지 않는다.** 관측이 결과를 미결정하는 것이 이 환경의
    존재 이유이므로, 은닉 물성을 여기 넣으면 연구 전체가 무의미해진다.
    """
    cfg = self.cfg
    L = cfg.world_width
    ex, ey, eth = self.gripper.pose
    bx = float(self.block_body.position.x)
    by = float(self.block_body.position.y)
    ba = float(self.block_body.angle)
    return np.array([
        ex / L, ey / L, math.cos(eth), math.sin(eth),
        self.gripper.gap / cfg.finger_opening_max,
        bx / L, by / L, math.cos(ba), math.sin(ba),
        self.block_w / L, self.block_h / L,
        self.src_box.inner_width / L, self.src_box.center_x / L,
        self.tgt_box.inner_width / L, self.tgt_box.center_x / L,
    ], dtype=np.float32)

  def _stacked_obs(self) -> np.ndarray:
    return np.concatenate(list(self._history)).astype(np.float32)

  def _info(self) -> dict:
    """**진단용** 정보. 학습 입력이 아니다.

    `mass`, `friction`은 은닉 물성이므로 정책에 넣으면 안 된다 — 분석/디버깅과
    특권(privileged) 컨트롤러 전용이다.
    """
    return {
        'mass': self._mass,
        'friction': self._mu,
        'contact_length': self.contact_length(),
        'is_held': self.is_held(),
        'n_drops': self._n_drops,
        'outcome': self._outcome,
        'steps': self._t,
    }


def _rect(x0: float, y0: float, x1: float,
          y1: float) -> List[Tuple[float, float]]:
  return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
