"""GraspCarry2D — 2D 블록 파지·운반 환경 (은닉 물성 + 실제 접촉 물리).

설계 의도 (2026-07-31 논의):
  기존 2D 장애물 환경은 **전관측 + 결정론**이라 steps-to-go 분포의 분산이
  기댓값의 재표현에 불과했다(독립 감사 결론). 이 환경은 불확실성의 출처를
  "인위적 노이즈"나 "인위적 마스킹"이 아니라 **물리적으로 볼 수 없는 물성**에
  둔다 — 음료 캔을 옮길 때 크기는 보이지만 **채움 정도와 표면 미끄러움은
  들어봐야 아는** 상황이다.

  관측 가능 : EE pose/개도, 블록 pose, 블록 h·w(기하), 박스 위치
  은닉      : 채움 정도 fill(-> 질량 m), 마찰계수 mu

센서 가정 (사용자 지정):
  접촉 센서 없음, 그리퍼 전류 센서 없음. **최대 그립력(=최대 토크/손가락반경)만
  정해져 있다.** 은닉 물성은 오직 **운동학적 단서**로만 드러난다:
    - 무거우면 그리퍼가 **처진다**(목표 대비 위치 오차) — 하중이 실제로 걸리므로
    - 미끄러우면 패드에서 **밀리거나 기운다**

물리 (v3 — 전부 물리엔진이 담당, 제약으로 붙이지 않는다):
  * EE 베이스: **DYNAMIC** 강체. 목표점으로 **힘 PD**로 구동하되 **자기 무게만**
    중력 보상한다. 블록 무게는 보상하지 않으므로 하중이 그대로 처짐으로 나타난다.
    KINEMATIC(무한 질량)이면 물체 힘을 전혀 받지 못해 "따로 노는" 문제가 생긴다.
  * 손가락 2개: **DYNAMIC**. GrooveJoint로 베이스에 대해 좌우로만 미끄러지고,
    GearJoint로 자세가 베이스에 고정된다. 그립은 **손가락에 힘을 가해** 닫는다.
  * 파지: **패드-블록 마찰 접촉**. 붙이는 조인트가 없다. 들 수 있는지, 미끄러지는지,
    기우는지는 전부 접촉 마찰(mu)과 그립력이 정한다.

  손가락은 UMI 그리퍼처럼 **곡선 아치 + 평평한 접촉 패드** 형상이다.

소스 박스를 **좁게** 만든 이유:
  그리퍼 바깥 폭이 박스 개구부보다 크면 rim 아래로 내려갈 수 없다. 그래서
  소스 박스에서는 대개 **얕게밖에 못 잡고**(패드가 블록 윗부분만 물어 무게중심이
  아래로 멀다 -> 기울어짐), 블록이 벽에 가까우면 더 얕아진다.
  타겟 박스는 넓어서 내려놓기는 쉽다.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymunk


# ----------------------------------------------------------------- config
@dataclass
class CarryConfig:
  """모든 수치는 여기에 노출한다(하드코딩 금지)."""
  # --- 월드 (512 픽셀 좌표계, y는 아래 방향) ---
  world: float = 512.0
  floor_y: float = 470.0
  gravity: float = 900.0
  dt: float = 0.005                 # 접촉 파지는 작은 스텝이 안정적
  control_hz: float = 10.0
  solver_iterations: int = 30       # 마찰 파지 안정화

  # --- 박스 (소스는 좁고, 타겟은 넓다) ---
  src_box_w: float = 96.0
  tgt_box_w: float = 150.0
  box_rim_h: float = 46.0
  src_cx: float = 110.0
  tgt_cx: float = 380.0
  wall_t: float = 5.0

  # --- 블록: 기하는 **관측 가능**, 물성은 은닉 ---
  block_w_range: tuple = (26.0, 36.0)
  block_h_range: tuple = (62.0, 88.0)
  density: float = 0.010            # m = density * (w*h) * fill
  fill_range: tuple = (0.60, 1.0)   # 은닉
  mu_range: tuple = (0.55, 1.05)    # 은닉 (패드-블록 마찰)

  # --- 그리퍼 형상 (UMI식 곡선 손가락 + 평평한 패드) ---
  finger_len: float = 62.0          # 손가락 전체 길이(아치 포함)
  pad_len: float = 28.0             # 평평한 접촉 패드 길이
  finger_t: float = 7.0             # 손가락 두께
  finger_curve: float = 15.0        # 아치가 바깥으로 휘는 정도
  gap_open: float = 56.0            # 열림 시 패드 안쪽 간격
  gap_closed: float = 12.0          # 완전히 닫힐 때 간격(빈손)
  palm_w: float = 74.0
  palm_t: float = 14.0
  stem_w: float = 16.0
  stem_h: float = 30.0

  # --- 그리퍼 힘/질량 ---
  tau_max: float = 520_000.0        # 최대 토크(고정) -> 그립력 = tau_max/r_finger
  r_finger: float = 12.0
  pad_friction: float = 1.35        # 패드 표면(블록 mu와 결합)
  base_mass: float = 26.0
  finger_mass: float = 4.0
  ee_force_max: float = 2.4e6       # 베이스 구동력 상한
  k_p: float = 110.0                # 힘 PD (가속도 차원)
  k_v: float = 21.0
  k_p_ang: float = 1400.0           # 손목 회전 강성. 낮으면 블록 하중
  k_v_ang: float = 90.0             # 토크에 그리퍼가 통째로 돌아간다
  ee_torque_max: float = 6.0e7
  finger_speed_max: float = 25.0   # 손가락 닫힘 속도 상한(실제 그리퍼처럼)
  ee_cmd_max: float = 60.0
  ee_vel_max: float = 900.0

  # --- 에피소드 ---
  max_steps: int = 400
  settle_speed: float = 8.0
  seed: Optional[int] = None
  obs_history: int = 4


# ----------------------------------------------------------------- helpers
def _rect(x0, y0, x1, y1):
  return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _finger_polys(cfg, sign):
  """UMI식 손가락: 바깥으로 휜 곡선 아치 + 안쪽 끝의 평평한 패드.

  로컬 원점 = 손가락이 베이스에 붙는 지점, +y가 아래.
  sign=-1 왼손가락, +1 오른손가락. 반환: 볼록 다각형 리스트(pymunk Poly용).
  """
  arch_len = cfg.finger_len - cfg.pad_len
  n = 5                                    # 아치를 n개 볼록 조각으로 근사
  polys = []
  ys = np.linspace(0.0, arch_len, n + 1)
  for i in range(n):
    y0, y1 = ys[i], ys[i + 1]
    c0 = sign * cfg.finger_curve * np.sin(np.pi * y0 / arch_len) ** 0.9
    c1 = sign * cfg.finger_curve * np.sin(np.pi * y1 / arch_len) ** 0.9
    h = cfg.finger_t / 2
    polys.append([(c0 - h, y0), (c0 + h, y0), (c1 + h, y1), (c1 - h, y1)])
  # 평평한 패드 (안쪽 면이 직선이라 물체와 면접촉한다)
  inner, outer = 0.0, -sign * cfg.finger_t
  polys.append(_rect(min(inner, outer), arch_len,
                     max(inner, outer), cfg.finger_len))
  return polys


# ----------------------------------------------------------------- env
class GraspCarry2D:
  """소스 박스의 블록을 그리퍼로 집어 타겟 박스로 옮긴다.

  action = (dx, dy, dtheta, grip)
    dx, dy   : EE 목표 변위 (픽셀). 크게 주면 가속도가 커져 미끄러질 위험↑
    dtheta   : EE 회전 변위 (rad)
    grip     : >0.5 이면 손가락을 닫는다(그립력 인가), 아니면 연다
  """

  def __init__(self, config: Optional[CarryConfig] = None):
    self.cfg = config or CarryConfig()
    self._rng = np.random.default_rng(self.cfg.seed)
    self._hist: list = []

  # ------------------------------------------------------------- setup
  def _boxes(self):
    cfg = self.cfg
    return ((cfg.src_cx, cfg.src_box_w), (cfg.tgt_cx, cfg.tgt_box_w))

  def _build_space(self):
    cfg = self.cfg
    space = pymunk.Space()
    space.gravity = (0.0, cfg.gravity)
    space.iterations = cfg.solver_iterations
    statics = [pymunk.Segment(space.static_body, (0, cfg.floor_y),
                              (cfg.world, cfg.floor_y), cfg.wall_t)]
    for x in (6.0, cfg.world - 6.0):
      statics.append(pymunk.Segment(space.static_body, (x, cfg.floor_y),
                                    (x, cfg.floor_y - 300), cfg.wall_t))
    for cx, bw in self._boxes():
      for sx in (-1, 1):
        x = cx + sx * bw / 2
        statics.append(pymunk.Segment(space.static_body, (x, cfg.floor_y),
                                      (x, cfg.floor_y - cfg.box_rim_h),
                                      cfg.wall_t))
    for s in statics:
      s.friction = 0.9
      s.elasticity = 0.0
    space.add(*statics)
    return space

  def _spawn_block(self):
    cfg = self.cfg
    w = float(self._rng.uniform(*cfg.block_w_range))
    h = float(self._rng.uniform(*cfg.block_h_range))
    fill = float(self._rng.uniform(*cfg.fill_range))     # 은닉
    mu = float(self._rng.uniform(*cfg.mu_range))          # 은닉
    mass = cfg.density * (w * h) * fill

    body = pymunk.Body(mass, pymunk.moment_for_box(mass, (w, h)))
    room = max(0.0, (cfg.src_box_w - w) / 2 - cfg.wall_t - 3)
    body.position = (cfg.src_cx + float(self._rng.uniform(-room, room)),
                     cfg.floor_y - h / 2 - cfg.wall_t)
    shape = pymunk.Poly.create_box(body, (w, h))
    shape.friction = mu
    shape.elasticity = 0.0
    self.space.add(body, shape)

    self.block, self.block_shape = body, shape
    self.block_w, self.block_h = w, h
    self.fill, self.mu, self.mass = fill, mu, mass

  def _spawn_ee(self):
    """베이스(DYNAMIC) + 손가락 2개(DYNAMIC, 좌우로만 미끄러짐)."""
    cfg = self.cfg
    y0 = cfg.floor_y - cfg.box_rim_h - cfg.finger_len - 40
    x0 = cfg.src_cx

    base = pymunk.Body(cfg.base_mass,
                       pymunk.moment_for_box(cfg.base_mass,
                                             (cfg.palm_w, cfg.palm_t)))
    base.position = (x0, y0)
    bs = [pymunk.Poly(base, _rect(-cfg.palm_w / 2, -cfg.palm_t,
                                  cfg.palm_w / 2, 0.0)),
          pymunk.Poly(base, _rect(-cfg.stem_w / 2, -cfg.palm_t - cfg.stem_h,
                                  cfg.stem_w / 2, -cfg.palm_t))]
    for s in bs:
      s.friction = 0.5
      s.elasticity = 0.0
      s.filter = pymunk.ShapeFilter(group=7)   # 그리퍼 부품끼리는 충돌 안 함
    self.space.add(base, *bs)

    self.fingers, self.finger_shapes = [], []
    half_open, half_closed = cfg.gap_open / 2, cfg.gap_closed / 2
    for sign in (-1.0, 1.0):
      fb = pymunk.Body(cfg.finger_mass,
                       pymunk.moment_for_box(cfg.finger_mass,
                                             (cfg.finger_t, cfg.finger_len)))
      fb.position = (x0 + sign * half_open, y0)
      shapes = [pymunk.Poly(fb, p) for p in _finger_polys(cfg, sign)]
      for s in shapes:
        s.friction = cfg.pad_friction
        s.elasticity = 0.0
        s.filter = pymunk.ShapeFilter(group=7)
      self.space.add(fb, *shapes)
      # 좌우로만 미끄러지는 프리즘 구속 + 자세 고정
      groove = pymunk.GrooveJoint(base, fb, (sign * half_closed, 0.0),
                                  (sign * (half_open + 6.0), 0.0), (0.0, 0.0))
      gear = pymunk.GearJoint(base, fb, 0.0, 1.0)
      gear.max_force = 5e7
      self.space.add(groove, gear)
      self.fingers.append(fb)
      self.finger_shapes.append(shapes)

    self.base, self.base_shapes = base, bs

  # ------------------------------------------------------- gripper geometry
  @property
  def ee(self):
    """호환용: 그리퍼 기준점(베이스)."""
    return self.base

  @property
  def finger_gap_cur(self):
    """현재 패드 안쪽 간격 (관측 가능 — 물린 물체 폭을 알려준다)."""
    return float(abs(self.fingers[1].position.x - self.fingers[0].position.x))

  def gripper_polys(self):
    """렌더링용 월드 좌표 다각형 리스트 (베이스 + 손가락들)."""
    out = []
    for body, shapes in ([(self.base, self.base_shapes)]
                         + list(zip(self.fingers, self.finger_shapes))):
      for s in shapes:
        out.append(np.array(
            [[(v.rotated(body.angle) + body.position).x,
              (v.rotated(body.angle) + body.position).y]
             for v in s.get_vertices()]))
    return out

  def max_descend_y(self, ex=None):
    """ex에서 베이스가 내려갈 수 있는 최저 y (손가락이 벽에 걸리는 한계).
    물리 충돌로도 막히지만, 목표점 자체를 제한해 교착을 피한다."""
    cfg = self.cfg
    ex = self.base.position.x if ex is None else ex
    half_out = cfg.gap_open / 2 + cfg.finger_t   # 박스로 들어가는 건 패드
    for cx, bw in self._boxes():
      inner_l, inner_r = cx - bw / 2 + cfg.wall_t, cx + bw / 2 - cfg.wall_t
      if (ex + half_out > inner_l - 2) and (ex - half_out < inner_r + 2):
        if (ex - half_out >= inner_l) and (ex + half_out <= inner_r):
          return cfg.floor_y - cfg.finger_len - 4
        return cfg.floor_y - cfg.box_rim_h - cfg.finger_len - 4
    return cfg.floor_y - cfg.finger_len - 4

  def contact_length(self):
    """패드와 블록 옆면이 겹친 길이. 접촉 '면적'(2D)이자 기울어짐 저항의 근거."""
    cfg = self.cfg
    if abs(self.block.position.x - self.base.position.x) > \
            (cfg.gap_open + self.block_w) / 2:
      return 0.0
    pad_top = self.base.position.y + (cfg.finger_len - cfg.pad_len)
    pad_bot = self.base.position.y + cfg.finger_len
    b_top = self.block.position.y - self.block_h / 2
    b_bot = self.block.position.y + self.block_h / 2
    return float(max(0.0, min(pad_bot, b_bot) - max(pad_top, b_top)))

  @property
  def grip_force(self):
    return self.cfg.tau_max / self.cfg.r_finger

  # ------------------------------------------------------------- 파지 판정
  def is_held(self):
    """양쪽 패드가 블록과 **실제로 접촉 중**인지 (물리엔진 arbiter 기준)."""
    touching = [False, False]

    def _visit(arb):
      for i, shapes in enumerate(self.finger_shapes):
        if arb.shapes[0] in shapes or arb.shapes[1] in shapes:
          touching[i] = True

    self.block.each_arbiter(_visit)
    return bool(touching[0] and touching[1])

  def block_airborne(self):
    return (self.block.position.y + self.block_h / 2) < self.cfg.floor_y - 14

  # ------------------------------------------------------------- obs
  def _raw_obs(self):
    cfg = self.cfg
    return np.array([
        self.base.position.x / cfg.world, self.base.position.y / cfg.world,
        np.cos(self.base.angle), np.sin(self.base.angle),
        self.finger_gap_cur / cfg.world,          # 개도: 물린 폭 단서
        self.block.position.x / cfg.world, self.block.position.y / cfg.world,
        np.cos(self.block.angle), np.sin(self.block.angle),
        self.block_w / cfg.world, self.block_h / cfg.world,
        cfg.src_cx / cfg.world, cfg.tgt_cx / cfg.world,
    ], dtype=np.float32)
    # 주: fill / mu / mass 는 **포함하지 않는다**(은닉).

  def _stacked_obs(self):
    return np.concatenate(self._hist, axis=0)

  @property
  def obs_dim(self):
    return 13 * self.cfg.obs_history

  # ------------------------------------------------------------- api
  def reset(self, seed: Optional[int] = None):
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    self.space = self._build_space()
    self.gripping = False
    self._spawn_block()
    self._spawn_ee()
    self.t = 0
    self.n_drops = 0
    self._was_held_air = False
    self.grasp_contact = 0.0
    for _ in range(40):
      self.space.step(self.cfg.dt)
    o = self._raw_obs()
    self._hist = [o.copy() for _ in range(self.cfg.obs_history)]
    return self._stacked_obs(), self._info()

  def step(self, action):
    cfg = self.cfg
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    dx, dy, dth, grip = float(a[0]), float(a[1]), float(a[2]), float(a[3])

    d = np.array([dx, dy], dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm > cfg.ee_cmd_max:
      d = d / nrm * cfg.ee_cmd_max
    target = np.array([self.base.position.x, self.base.position.y]) + d
    target[1] = min(target[1], self.max_descend_y(target[0]))
    target[0] = float(np.clip(target[0], 30, cfg.world - 30))
    target_th = self.base.angle + float(np.clip(dth, -0.25, 0.25))

    self.gripping = grip > 0.5
    n_sub = int(1.0 / (cfg.dt * cfg.control_hz))
    assembly_m = cfg.base_mass + 2 * cfg.finger_mass

    for _ in range(n_sub):
      # --- 베이스: 힘 PD + **자기 무게만** 중력 보상.
      # 블록 무게는 보상하지 않으므로 하중이 처짐으로 그대로 나타난다.
      pos = np.array([self.base.position.x, self.base.position.y])
      vel = np.array([self.base.velocity.x, self.base.velocity.y])
      acc = cfg.k_p * (target - pos) - cfg.k_v * vel
      # 중력 보상은 **위쪽(-y)** 이다. 부호를 틀리면 중력이 두 배가 된다.
      F = assembly_m * acc - np.array([0.0, assembly_m * cfg.gravity])
      fn = float(np.linalg.norm(F))
      if fn > cfg.ee_force_max:
        F = F / fn * cfg.ee_force_max
      self.base.force = (float(F[0]), float(F[1]))
      tq = (self.base.moment
            * (cfg.k_p_ang * (target_th - self.base.angle)
               - cfg.k_v_ang * self.base.angular_velocity))
      self.base.torque = float(np.clip(tq, -cfg.ee_torque_max, cfg.ee_torque_max))

      # --- 손가락: 그립력을 **힘으로** 인가한다(닫힘/열림 모두)
      for sign, fb in zip((-1.0, 1.0), self.fingers):
        fx = (-sign * self.grip_force if self.gripping
              else sign * self.grip_force * 0.25)
        fb.force = (fx, -cfg.finger_mass * cfg.gravity)

      sp = float(np.linalg.norm(vel))
      if sp > cfg.ee_vel_max:
        v = vel / sp * cfg.ee_vel_max
        self.base.velocity = (float(v[0]), float(v[1]))
      self.space.step(cfg.dt)
      # 로봇 기구학적 하드 스톱: 목표만 제한하면 동적 베이스가 오버슛해
      # 패드가 rim 아래로 들어가 벽에 끼인다(실측). 실제 위치도 막는다.
      lim = self.max_descend_y(self.base.position.x)
      if self.base.position.y > lim:
        self.base.position = (self.base.position.x, lim)
        if self.base.velocity.y > 0:
          self.base.velocity = (self.base.velocity.x, 0.0)
      # 손가락 닫힘 속도 제한: 힘은 그대로 전달되지만(접촉 시 그립력 유지)
      # 빈 공간에서 가속해 물체를 때려 튕겨내는 일은 막는다.
      for fb in self.fingers:
        rel = fb.velocity.x - self.base.velocity.x
        if abs(rel) > cfg.finger_speed_max:
          fb.velocity = (self.base.velocity.x
                         + np.sign(rel) * cfg.finger_speed_max, fb.velocity.y)

    # --- 낙하 집계: 공중에서 물고 있다가 놓친 경우
    held = self.is_held()
    if held and self.block_airborne():
      self._was_held_air = True
      self.grasp_contact = self.contact_length()
    elif self._was_held_air and not held:
      self.n_drops += 1
      self._was_held_air = False

    self.t += 1
    o = self._raw_obs()
    self._hist = self._hist[1:] + [o]
    term = self.success()
    trunc = self.t >= cfg.max_steps
    return self._stacked_obs(), float(term), term, trunc, self._info()

  def success(self):
    cfg = self.cfg
    inside = (abs(self.block.position.x - cfg.tgt_cx) < cfg.tgt_box_w / 2 - 4
              and (self.block.position.y + self.block_h / 2)
              > cfg.floor_y - cfg.wall_t - 14)
    still = self.block.velocity.length < cfg.settle_speed
    return bool(inside and still and not self.is_held())

  @property
  def clear_y(self):
    return (self.cfg.floor_y - self.cfg.box_rim_h - self.block_h
            - self.cfg.finger_len * 0.25 - 14.0)

  def _info(self):
    return dict(t=self.t, gripping=self.gripping, held=self.is_held(),
                n_drops=self.n_drops, contact=self.contact_length(),
                gap=self.finger_gap_cur,
                hidden=dict(fill=self.fill, mu=self.mu, mass=self.mass),
                block=dict(w=self.block_w, h=self.block_h,
                           x=float(self.block.position.x),
                           y=float(self.block.position.y)),
                ee=dict(x=float(self.base.position.x),
                        y=float(self.base.position.y)))


# ------------------------------------------------------------- 데모 컨트롤러
class ScriptedCarryPolicy:
  """상태기계 스크립트 정책: 정렬 -> 하강 -> 파지 -> 들기 -> 이동 -> 내려놓기.

  speed가 크면 가속도가 커져 빨리 가지만, 패드 마찰이 부족하면 미끄러진다.
  낙하하면 더 조심해서(느리게) 재시도한다.
  """

  def __init__(self, speed=40.0, privileged=False,
               fast=58.0, slow=26.0, mass_thresh=18.0,
               a_reduce=0.65, min_speed=14.0):
    self.speed = speed
    self.privileged = privileged
    self.fast, self.slow, self.thresh = fast, slow, mass_thresh
    self.a_reduce, self.min_speed = a_reduce, min_speed
    self.reset()

  def reset(self):
    self.phase = 'align'
    self._cur = None
    self._seen_drops = 0
    self._grip_wait = 0

  def _speed_for(self, env):
    if self._cur is None:
      self._cur = ((self.slow if env.mass >= self.thresh else self.fast)
                   if self.privileged else self.speed)
    if env.n_drops > self._seen_drops:
      self._seen_drops = env.n_drops
      self._cur = max(self.min_speed, self._cur * self.a_reduce)
    return self._cur

  def __call__(self, env):
    cfg = env.cfg
    ex, ey = env.base.position.x, env.base.position.y
    bx, by = env.block.position.x, env.block.position.y
    v = self._speed_for(env)
    care = min(v, 20.0)
    # 접근 높이: 열린 손가락 끝이 **블록 위**를 완전히 지나가야 한다.
    # 낮으면 이동 중 패드가 블록 옆구리를 쳐서 정렬이 끝나지 않는다(실측).
    safe_y = min(cfg.floor_y - cfg.box_rim_h - cfg.finger_len - 18,
                 (by - env.block_h / 2) - cfg.finger_len - 10)

    def act(dx, dy, g):
      return np.array([dx, dy, 0.0, g], np.float32)

    held = env.is_held()
    if not held and self.phase in ('lift', 'traverse', 'lower'):
      self.phase = 'align'
      self._grip_wait = 0

    if self.phase == 'align':
      if abs(ex - bx) < 2.5 and ey <= safe_y + 5:
        self.phase = 'descend'
      else:
        return act(np.clip(bx - ex, -care, care),
                   np.clip(safe_y - ey, -care, care), 0.0)

    if self.phase == 'descend':
      # 패드가 블록 옆면을 충분히 물도록 내려간다
      want_y = (by - env.block_h / 2 - (cfg.finger_len - cfg.pad_len)
                + min(cfg.pad_len, env.block_h) * 0.5)
      deep = min(env.max_descend_y(ex), want_y)
      if ey >= deep - 3:
        self.phase = 'grip'
        self._grip_wait = 0
      else:
        return act(np.clip(bx - ex, -6, 6), min(care, deep - ey), 0.0)

    if self.phase == 'grip':                # 손가락이 닫혀 물릴 때까지 대기
      self._grip_wait += 1
      if held:
        self.phase = 'lift'
      elif self._grip_wait > 10:
        self.phase = 'align'
        return act(0.0, -care, 0.0)
      else:
        return act(0.0, 0.0, 1.0)

    if self.phase == 'lift':
      if ey <= env.clear_y:
        self.phase = 'traverse'
      else:
        return act(0.0, -v, 1.0)

    if self.phase == 'traverse':
      if abs(ex - cfg.tgt_cx) < 8:
        self.phase = 'lower'
      else:
        return act(np.clip(cfg.tgt_cx - ex, -v, v),
                   np.clip(env.clear_y - ey, -v, v), 1.0)

    if self.phase == 'lower':
      limit = env.max_descend_y(ex)
      if ey >= limit - 5 or by >= cfg.floor_y - env.block_h * 0.5 - 16:
        self.phase = 'release'
      else:
        return act(np.clip(cfg.tgt_cx - ex, -care, care), care, 1.0)

    if self.phase == 'release':
      return act(0.0, -10.0, 0.0)

    return act(0.0, 0.0, 0.0)


def rollout(env, policy, seed, max_steps=None):
  """한 에피소드 실행. 반환: dict(success, steps, n_drops, contact, mass)."""
  max_steps = max_steps or env.cfg.max_steps
  env.reset(seed=seed)
  policy.reset()
  for t in range(max_steps):
    obs, r, term, trunc, info = env.step(policy(env))
    if term:
      return dict(success=True, steps=t + 1, n_drops=env.n_drops,
                  contact=env.grasp_contact, mass=env.mass)
    if trunc:
      break
  return dict(success=False, steps=max_steps, n_drops=env.n_drops,
              contact=env.grasp_contact, mass=env.mass)
