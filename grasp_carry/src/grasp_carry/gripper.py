r"""ALOHA식 2지 평행 그리퍼의 pymunk 강체 모델 (phase 3, step 1).

측면도 2D 단순화. 단위계는 `config.py`와 동일(mm / kg / s / mN, +y가 아래):

```
        [스템/마운트]        <- 베이스에 붙은 세로 바
     ====[레일 플레이트]====  <- 가로 바 (베이스 강체)
       |             |
      /|             |\      <- 삼각 손가락 2개 (안쪽면은 수직=평평)
     / |             | \
```

이전 시도(`src/grasp_carry_env.py` v1~v3, `experiments/2026-07-31_grasp-carry-env.md`)
에서 **실측으로** 확인된 제약을 그대로 반영한다:

1. 베이스·손가락은 전부 **DYNAMIC**이다. KINEMATIC(무한 질량)이면 물체의
   반작용을 전혀 받지 못해 물체가 그리퍼와 따로 논다.
2. 파지는 **패드-물체 마찰 접촉**으로만 성립한다. 용접/PivotJoint로 붙이면
   "한계 넘으면 즉시 낙하"가 되어 점진적 미끄러짐(= 센서 없는 설정의 유일한
   경고 신호)이 사라진다.
3. 중력이 +y(아래)이므로 중력 보상력은 **-y(위)** 다. 부호를 틀리면 중력이
   두 배가 되어 그리퍼가 위로 올라가지 못한다.
4. 손가락 닫힘 속도를 `cfg.finger_speed_max`로 제한한다. 그립력을 가벼운
   손가락에 그대로 걸면 가속도가 과도해 물체를 때려 튕겨낸다.
5. 손목 회전은 병진과 **별도 게인**(`k_p_ang`, `k_v_ang`)으로 제어한다.
   회전 강성이 낮으면 하중 토크에 그리퍼가 통째로 돌아간다.

**물체 질량은 중력 보상하지 않는다.** 하중에 의한 처짐이 은닉 질량을 드러내는
유일한 관측 단서이기 때문이다(이 환경의 핵심 설계).

`space`는 호출자가 만든 것을 받는다. 이 모듈은 Space를 만들지 않는다.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np
import pymunk

from .config import CarryConfig


def _rect(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
  return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _finger_verts(cfg: CarryConfig, sign: float) -> List[Tuple[float, float]]:
  """삼각 손가락의 로컬 정점 3개. `sign`=-1 왼손가락, +1 오른손가락.

  로컬 원점 = **패드 안쪽 면의 위쪽 끝**(= 베이스에 물리는 지점). +y는 아래.
  안쪽 면은 `(0,0)-(0,finger_length)`의 **수직 직선**이라 물체 옆면과 면접촉해
  회전을 저항한다. 바깥쪽은 위쪽만 `finger_thickness`만큼 두껍고 끝으로 갈수록
  좁아지는 테이퍼다. 삼각형은 그 자체로 볼록이라 `pymunk.Poly` 하나로 족하다.
  """
  return [(0.0, 0.0),
          (sign * cfg.finger_thickness, 0.0),
          (0.0, cfg.finger_length)]


def _wrap_angle(a: float) -> float:
  return (a + math.pi) % (2.0 * math.pi) - math.pi


class Gripper:
  """ALOHA식 2지 평행 그리퍼. 베이스 DYNAMIC + 손가락 2개 DYNAMIC.

  베이스 로컬 원점은 **레일 플레이트 아랫면의 중심**이고, 손가락은 거기(y=0)에
  물려 좌우로만 미끄러진다(`GrooveJoint`). 자세는 `GearJoint`로 베이스에
  고정되므로 손가락은 베이스에 대해 1자유도(개도)만 갖는다.
  """

  # 그리퍼 부품끼리는 서로 충돌하지 않아야 한다(같은 group). 한 Space에 여러
  # 그리퍼를 둘 수도 있으므로 인스턴스마다 다른 group 번호를 쓴다.
  _next_group = 1

  def __init__(self, space: pymunk.Space, cfg: CarryConfig,
               position: Sequence[float]):
    self.space = space
    self.cfg = cfg
    self._group = Gripper._next_group
    Gripper._next_group += 1
    self._signs = (-1.0, 1.0)          # 0=왼손가락, 1=오른손가락
    self._build(position)

  # ------------------------------------------------------------------ 구성
  def _build(self, position: Sequence[float]) -> None:
    cfg = self.cfg
    filt = pymunk.ShapeFilter(group=self._group)

    # --- 베이스: 레일 플레이트 + 스템 (볼록 다각형 2개, 강체 1개) ---
    # 어셈블리 총질량이 cfg.assembly_mass가 되도록 손가락 몫을 뺀다.
    base_mass = cfg.assembly_mass - 2.0 * cfg.finger_mass
    if base_mass <= 0.0:
      raise ValueError('assembly_mass must exceed 2 * finger_mass')
    plate = _rect(-cfg.gripper_outer_width / 2.0, -cfg.rail_plate_thickness,
                  cfg.gripper_outer_width / 2.0, 0.0)
    stem = _rect(-cfg.stem_width / 2.0,
                 -cfg.rail_plate_thickness - cfg.stem_length,
                 cfg.stem_width / 2.0, -cfg.rail_plate_thickness)
    a_plate = cfg.gripper_outer_width * cfg.rail_plate_thickness
    a_stem = cfg.stem_width * cfg.stem_length
    m_plate = base_mass * a_plate / (a_plate + a_stem)
    m_stem = base_mass - m_plate
    # 무게중심은 로컬 원점(0,0)에 둔다. 관성모멘트도 같은 점 기준으로 더해야
    # 일관된다(pymunk.moment_for_poly는 정점 좌표계 원점 기준).
    base_moment = (pymunk.moment_for_poly(m_plate, plate)
                   + pymunk.moment_for_poly(m_stem, stem))
    base = pymunk.Body(base_mass, base_moment)
    base.position = (float(position[0]), float(position[1]))
    self.base_shapes = [pymunk.Poly(base, plate), pymunk.Poly(base, stem)]
    for s in self.base_shapes:
      s.friction = 0.5
      s.elasticity = 0.0
      s.filter = filt
    self.space.add(base, *self.base_shapes)
    self.base = base

    # --- 손가락 2개: 좌우로만 미끄러지는 삼각 강체 ---
    half_min = cfg.finger_opening_min / 2.0
    half_max = cfg.finger_opening_max / 2.0
    half_start = 0.5 * (half_min + half_max)     # 중간 개도에서 시작
    self.fingers: List[pymunk.Body] = []
    self.finger_shapes: List[pymunk.Poly] = []
    finger_moment = 0.0
    for sign in self._signs:
      verts = _finger_verts(cfg, sign)
      finger_moment = pymunk.moment_for_poly(cfg.finger_mass, verts)
      body = pymunk.Body(cfg.finger_mass, finger_moment)
      body.angle = base.angle
      body.position = base.local_to_world((sign * half_start, 0.0))
      shape = pymunk.Poly(body, verts)
      shape.friction = cfg.pad_friction     # 접촉 마찰은 물체 쪽 mu가 지배
      shape.elasticity = 0.0
      shape.filter = filt
      self.space.add(body, shape)

      # 프리즘 구속: 손가락 원점(= 패드 안쪽 면)이 베이스 로컬 수평선 위의
      # [개도 하한/2, 개도 상한/2] 구간에만 있을 수 있다.
      groove = pymunk.GrooveJoint(base, body, (sign * half_min, 0.0),
                                  (sign * half_max, 0.0), (0.0, 0.0))
      # 자세를 베이스에 고정(비 = 1). max_force는 기본값 무한대 —
      # 손가락은 베이스에 대해 회전 자유도가 아예 없어야 한다.
      gear = pymunk.GearJoint(base, body, 0.0, 1.0)
      self.space.add(groove, gear)
      self.fingers.append(body)
      self.finger_shapes.append(shape)

    # 회전 PD가 목표 각가속도를 내려면 어셈블리 전체의 관성모멘트가 필요하다
    # (손가락은 GearJoint로 베이스와 함께 돈다). 손가락 오프셋은 중간 개도 기준.
    self._assembly_moment = base_moment + 2.0 * (
        finger_moment + cfg.finger_mass * half_start ** 2)

  # ------------------------------------------------------------------ 관측
  @property
  def gap(self) -> float:
    """현재 패드 안쪽 면 사이 간격(mm). 관측 가능한 값.

    손가락 로컬 원점이 곧 안쪽 면이므로, 베이스 좌표계에서의 두 원점 x
    차이가 그대로 개도가 된다.
    """
    left = self.base.world_to_local(self.fingers[0].position)
    right = self.base.world_to_local(self.fingers[1].position)
    return float(right.x - left.x)

  @property
  def pose(self) -> Tuple[float, float, float]:
    """(x, y, angle)"""
    return (float(self.base.position.x), float(self.base.position.y),
            float(self.base.angle))

  # ------------------------------------------------------------------ 제어
  def apply_grip(self, closing: bool) -> None:
    """손가락에 그립력을 인가한다(닫힘/열림 모두 같은 크기 = 같은 모터).

    매 물리 substep마다 호출되는 것을 전제로 한다(pymunk는 step마다 힘을
    비운다). 조인트로 붙이지 않고 힘만 걸므로, 파지는 패드-물체 마찰 접촉으로만
    성립하고 한계를 넘으면 **점진적으로** 미끄러진다.
    """
    f = self.cfg.grip_force
    for sign, body in zip(self._signs, self.fingers):
      fx = -sign * f if closing else sign * f
      # 로컬 좌표 힘 → 베이스가 기울어도 항상 개폐 축 방향이다.
      body.apply_force_at_local_point((fx, 0.0), (0.0, 0.0))

  def apply_pose_control(self, target_xy: Sequence[float],
                         target_angle: float) -> None:
    """절대 목표 pose로 임피던스 PD 제어. 매 substep 호출 전제.

    병진: `F = M*(k_p*(target-pos) - k_v*vel) - M*g_vec`. `M`은 그리퍼
    어셈블리 질량이고 중력 보상은 **-y(위)** 다. **물체 질량은 보상하지
    않는다** — 하중에 의한 처짐(`sag = m_obj*g/(M*k_p)`)이 은닉 질량을
    드러내는 관측 단서이기 때문이다.

    회전: 병진과 별도 게인(`k_p_ang`, `k_v_ang`)을 쓴다. 힘·토크는 모터의
    물리 한계(`ee_force_max`, `ee_torque_max`)로 클립한다.
    """
    cfg = self.cfg
    m = cfg.assembly_mass
    pos = np.array([self.base.position.x, self.base.position.y])
    vel = np.array([self.base.velocity.x, self.base.velocity.y])
    tgt = np.asarray(target_xy, dtype=np.float64).reshape(2)

    acc = cfg.k_p * (tgt - pos) - cfg.k_v * vel
    # 중력은 +y(아래)이므로 보상력은 -y(위). 부호를 틀리면 중력이 두 배가 된다.
    force = m * acc - np.array([0.0, m * cfg.gravity])
    norm = float(np.linalg.norm(force))
    if norm > cfg.ee_force_max:
      force = force * (cfg.ee_force_max / norm)
    self.base.force = (float(force[0]), float(force[1]))

    err = _wrap_angle(float(target_angle) - self.base.angle)
    torque = self._assembly_moment * (
        cfg.k_p_ang * err - cfg.k_v_ang * self.base.angular_velocity)
    self.base.torque = float(
        np.clip(torque, -cfg.ee_torque_max, cfg.ee_torque_max))

  def enforce_symmetry(self) -> None:
    """두 손가락을 개폐 축에서 **좌우 대칭**으로 되돌린다. 매 substep 직후 호출.

    실제 평행죠 그리퍼는 단일 액추에이터(랙&피니언/리드스크류)라 좌우 손가락이
    기구적으로 묶여 있어 `x_L = -x_R`가 항상 성립한다. 그런데 여기서는 손가락이
    각각 독립 DYNAMIC 바디이고 `GrooveJoint`는 베이스에만 묶으므로, 좌우를
    이어주는 구속이 없다. 그 상태에서는 블록이 한쪽 패드를 밀면 **그쪽만**
    미끄러져 손가락 쌍 전체가 옆으로 밀린다(실측: 베이스 로컬 x의 합이 0이 아닌
    40~48mm까지 벌어짐). 기구학적으로 불가능한 상태다.

    구현은 공통모드 투영이다. 베이스 로컬 개폐 축에서 두 손가락의 **평균**
    (= 공통모드)을 위치·속도 모두에서 제거하고, 차동모드(개도)는 건드리지
    않는다. 두 손가락 질량이 같으므로 속도 공통모드를 없애면 축 방향 운동량
    `2*m*c`가 남는데, 이것을 **베이스에 넘겨준다** — 랙&피니언 하우징이 받는
    반작용에 해당하며, 이렇게 해야 "블록이 패드를 밀면 그리퍼 전체가 밀린다"는
    올바른 반응이 나온다.
    """
    axis = pymunk.Vec2d(math.cos(self.base.angle), math.sin(self.base.angle))
    fl, fr = self.fingers

    # --- 위치: 로컬 x의 평균만큼 두 손가락을 함께 되민다(구속 드리프트 보정) ---
    xl = self.base.world_to_local(fl.position).x
    xr = self.base.world_to_local(fr.position).x
    common = 0.5 * (xl + xr)
    if common != 0.0:
      fl.position = fl.position - axis * common
      fr.position = fr.position - axis * common

    # --- 속도: 축 방향 공통모드를 제거하고 그 운동량을 베이스로 넘긴다 ---
    vl = (fl.velocity - self.base.velocity).dot(axis)
    vr = (fr.velocity - self.base.velocity).dot(axis)
    cv = 0.5 * (vl + vr)
    if cv != 0.0:
      fl.velocity = fl.velocity - axis * cv
      fr.velocity = fr.velocity - axis * cv
      # 손가락 2개가 잃은 축 방향 운동량을 베이스가 그대로 받는다.
      self.base.velocity = self.base.velocity + axis * (
          2.0 * self.cfg.finger_mass * cv / self.base.mass)

  def clamp_finger_speed(self) -> None:
    """손가락의 베이스 대비 상대 속도를 `cfg.finger_speed_max`로 제한한다.

    매 물리 substep **직후** 호출 전제. 힘 자체는 유지되므로 접촉 시 그립력은
    그대로이고, 빈 공간에서 가속해 물체를 때리는 것만 막는다. 개폐 축(베이스
    로컬 x) 성분만 제한하고 나머지 성분은 건드리지 않는다.
    """
    vmax = self.cfg.finger_speed_max
    axis = pymunk.Vec2d(math.cos(self.base.angle), math.sin(self.base.angle))
    base_vel = self.base.velocity
    for body in self.fingers:
      along = (body.velocity - base_vel).dot(axis)
      if abs(along) > vmax:
        excess = along - math.copysign(vmax, along)
        body.velocity = body.velocity - axis * excess

  # ------------------------------------------------------------------ 기하
  def polygons(self) -> List[np.ndarray]:
    """렌더링용 월드 좌표 다각형 리스트 — 현재 개도를 반영한다.

    순서: [레일 플레이트, 스템, 왼손가락, 오른손가락].
    """
    out = []
    pairs = [(self.base, s) for s in self.base_shapes]
    pairs += list(zip(self.fingers, self.finger_shapes))
    for body, shape in pairs:
      out.append(np.array([[body.local_to_world(v).x,
                            body.local_to_world(v).y]
                           for v in shape.get_vertices()], dtype=np.float64))
    return out

  def pad_span_y(self) -> Tuple[float, float]:
    """패드의 (위, 아래) 월드 y 좌표. 접촉 길이 계산에 쓴다.

    '위'는 베이스에 가까운 쪽(`finger_length - pad_length`), '아래'는 손가락
    끝(`finger_length`)이다. 두 손가락의 평균을 쓴다.
    """
    cfg = self.cfg
    tops, bottoms = [], []
    for body in self.fingers:
      tops.append(body.local_to_world(
          (0.0, cfg.finger_length - cfg.pad_length)).y)
      bottoms.append(body.local_to_world((0.0, cfg.finger_length)).y)
    return (float(np.mean(tops)), float(np.mean(bottoms)))
