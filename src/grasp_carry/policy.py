r"""`GraspCarry2D`용 스크립트 정책 (phase 3, step 3).

이 환경의 난이도는 **소스 박스가 좁다**는 데서 온다. 박스가 좁으면 그리퍼가
rim 아래로 못 내려가 블록 **윗부분만** 물게 되고, 파지점과 무게중심 사이에
지렛대(`arm`)가 생긴다. 수평 가속 `a`는 그 지렛대에서 `m*a*arm`의 토크를
만들고, 이것이 패드의 회전 저항을 넘으면 블록이 그립 안에서 돌아 빠진다.

따라서 정책이 반드시 갖춰야 하는 것 두 가지:

1. **속도 선택** — 액션이 절대 목표 위치이므로 "목표를 현재 위치에서 얼마나
   멀리 두는가"(`s`, mm)가 곧 명령 가속도다(`a = k_p * s`). 파지 상태의
   회전 토크 여유에서 `s`의 상한을 유도한다(`_speed_cap`).
2. **재파지** — 지렛대가 커서 안전 속도가 너무 낮으면, 블록을 그리퍼가 끝까지
   내려갈 수 있는 자리(박스 밖 바닥 또는 넓은 박스 안)로 옮겨 놓고 **무게중심
   높이에서** 다시 잡는다. 시간 비용을 치르는 대신 이후 운반이 빨라진다.
   재파지 여부는 임의 임계값이 아니라 **완료까지 예상 스텝 수 비교**로 정한다.

이전 시도(`experiments/2026-07-31_grasp-carry-env.md`)에서 실측으로 확인돼 여기
반영된 것:

- 접근 높이가 낮으면 **열린 손가락이 블록 옆구리를 쳐서** 정렬이 끝나지 않는다.
  수평 이동은 손가락 끝이 블록 상단과 모든 rim 위를 지나는 높이에서만 한다.
- 낙하 후 감속 적응이 없으면 감당 못 하는 블록에서 무한히 놓쳐 에피소드가
  끝나지 않는다.

**은닉 물성(질량·마찰)은 `privileged=True`일 때만 본다.** 기본값에서는
`config`의 설계 범위(에피소드별 표본이 아니라 사전 지식)에서 위험 분위수를
취해 쓴다.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from .config import CarryConfig

# 손가락 끝/블록 바닥이 rim 위를 지날 때 남길 여유(mm). 물리적 여유값이라
# 태스크 난이도와 무관하다.
_CLEARANCE = 12.0
# 위치 도달 판정 허용오차(mm).
_REACH_TOL = 3.0
# 파지 시 그리퍼 중심과 블록 중심의 허용 x 오프셋을 정할 때 남길 여유(mm).
# 열린 손가락 안쪽 면이 블록 옆면 바깥에 있어야 하강할 수 있다.
_SIDE_MARGIN = 6.0

_OPEN, _CLOSE = 0.0, 1.0


def _rect_half_extent(w: float, h: float, angle: float) -> Tuple[float, float]:
  """각도 `angle`로 돌아간 w x h 직사각형의 월드 (x, y) 반지름."""
  c, s = abs(math.cos(angle)), abs(math.sin(angle))
  return (c * w / 2.0 + s * h / 2.0, s * w / 2.0 + c * h / 2.0)


class ScriptedCarryPolicy:
  """소스 박스의 블록을 타겟 박스로 옮기는 상태기계 정책.

  `__call__(env)`가 절대 목표 pose 액션 `(x, y, theta, grip)`을 돌려준다.

  파라미터
  --------
  speed:
      한 스텝에 목표를 현재 위치에서 얼마나 멀리 둘지의 상한(mm). 목표를
      항상 `s`만큼 앞에 두면 PD가 정상상태 속도 `k_p*s/k_v = s mm/step`으로
      수렴하므로 사실상 순항 속도이기도 하다. 기본값 `None`이면
      `max_accel / k_p`(= 로봇 최대 가속도를 그대로 요구하는 명령 거리)를 쓴다.
  allow_regrasp:
      `False`면 재파지 phase에 진입하지 않는다(직접 운반만).
  privileged:
      `True`면 `env._info()`의 은닉 물성(질량·마찰)을 실제값으로 읽어 속도를
      고른다. `False`(기본)면 `config`의 설계 범위에서 위험 분위수를 취한다.
      같은 관측에서 다른 행동을 만들어 다봉성 데이터를 얻기 위한 스위치다.
  risk_alpha:
      비특권 모드의 위험 분위수. `alpha=0.1`이면 "질량 90분위 / 마찰 10분위"
      라는 뜻이다(불리한 쪽 10% 지점).
  torque_margin:
      `cfg.tip_torque_capacity`는 접촉 길이 전체가 최대 지렛대에서 마찰을
      내는 **상한**(`mu*F*L`)이다. 압력이 패드에 고르게 분포하면 실제 회전
      저항은 `mu*F*L/4`라 최소 4가 필요하고, 접촉이 완전한 면접촉이 아닌
      점을 감안해 여기서 더 키운다. 유일한 실측 보정 상수다.
  drop_backoff:
      낙하 1회마다 속도 상한에 곱하는 계수. 없으면 감당 못 하는 블록에서
      무한히 놓친다(실측된 교착 모드).
  """

  #: 정책이 거치는 phase. `transport`는 하위 상태(`_sub`)로 세분된다.
  PHASES = ('approach', 'descend', 'grasp', 'assess', 'transport', 'verify')

  def __init__(self, speed: Optional[float] = None, allow_regrasp: bool = True,
               privileged: bool = False, risk_alpha: float = 0.1,
               torque_margin: float = 4.0, drop_backoff: float = 0.55,
               config: Optional[CarryConfig] = None):
    self.cfg = config or CarryConfig()
    self.speed = float(speed) if speed is not None \
        else self.cfg.max_accel / self.cfg.k_p
    self.allow_regrasp = bool(allow_regrasp)
    self.privileged = bool(privileged)
    self.risk_alpha = float(risk_alpha)
    self.torque_margin = float(torque_margin)
    self.drop_backoff = float(drop_backoff)
    self.reset()

  # =================================================================== 상태
  def reset(self) -> None:
    """에피소드 시작 상태로 되돌린다."""
    self.phase = 'approach'
    self._sub = 'lift'
    self._dest_x = 0.0
    self._dest_floor = 0.0
    self._after = 'verify'
    self._wait = 0
    self._lost = 0
    self._stall = 0
    self._prev_y = None
    self._hold_y = None
    self._grasp_x_cache = None
    self._speed_hold = self.speed
    self._backoff = 1.0
    # --- 진단용 통계 (eval 스크립트가 읽는다) ---
    self.n_regrasps = 0
    self.n_drops = 0
    self.regrasped = False
    self.grasp_contacts: List[float] = []
    self.grasp_arms: List[float] = []
    self.grasp_speeds: List[float] = []

  # ================================================================= 진입점
  def __call__(self, env) -> np.ndarray:
    """`env`의 현재 상태를 보고 절대 위치 액션 `(x, y, theta, grip)`을 낸다."""
    self._sense(env)
    self._watch_drop(env)
    # phase 전이는 액션 없이 연쇄될 수 있다(예: descend -> grasp -> assess).
    for _ in range(len(self.PHASES) + 2):
      action = getattr(self, '_phase_' + self.phase)(env)
      if action is not None:
        return action
    return self._act(self.ex, self.ey, _OPEN)

  # ----------------------------------------------------------------- 관측
  def _sense(self, env) -> None:
    """이번 스텝에 쓸 기하량을 캐시한다. **전부 관측 가능한 값만** 쓴다."""
    self.ex, self.ey, _ = env.gripper.pose
    body = env.block_body
    self.bx = float(body.position.x)
    self.by = float(body.position.y)
    hx, hy = _rect_half_extent(env.block_w, env.block_h, float(body.angle))
    self.block_half_x = hx
    self.block_top = self.by - hy
    self.block_bottom = self.by + hy
    pad_top, pad_bottom = env.gripper.pad_span_y()
    self.pad_center = 0.5 * (pad_top + pad_bottom)

  def _watch_drop(self, env) -> None:
    """운반 중 파지가 풀렸으면 감속하고 접근부터 다시 한다.

    한 스텝짜리 접촉 깜빡임으로 오판하지 않도록 2스텝 연속일 때만 낙하로 본다.
    """
    if self.phase != 'transport' or self._sub not in ('lift', 'across', 'lower'):
      self._lost = 0
      return
    if env.is_held():
      self._lost = 0
      return
    self._lost += 1
    if self._lost >= 2:
      self.n_drops += 1
      self._backoff *= self.drop_backoff
      self._lost = 0
      self._enter('approach')

  # ================================================================== phase
  def _enter(self, phase: str) -> None:
    self.phase = phase
    self._wait = 0
    self._stall = 0
    self._prev_y = None
    self._grasp_x_cache = None

  def _phase_approach(self, env):
    """열린 손가락으로 블록 위 안전 높이에 정렬한다."""
    gx = self._grasp_x(env)
    ty = self._travel_y_free(env)
    if self.ey > ty + _REACH_TOL:
      # 먼저 **위로**. 낮은 높이에서 수평 이동하면 열린 손가락이 블록 옆구리를
      # 쳐서 정렬이 끝나지 않는다(실측된 교착 모드).
      return self._goto(self.ex, ty, self.speed, _OPEN)
    if abs(self.ex - gx) > 2.0:
      return self._goto(gx, ty, self.speed, _OPEN)
    self._enter('descend')
    return None

  def _phase_descend(self, env):
    """벽이 허용하는 만큼(최대 무게중심 높이까지) 내려간다."""
    gx = self._grasp_x(env)
    goal = self._grasp_base_y(env, gx)
    if self.ey >= goal - 2.0 or self._stalled():
      self._hold_y = self.ey
      self._enter('grasp')
      return None
    return self._goto(gx, goal, self.speed, _OPEN)

  def _phase_grasp(self, env):
    """손가락을 닫고 양쪽 패드가 물릴 때까지 기다린다."""
    self._wait += 1
    if env.is_held() and self._wait >= 3:
      self._enter('assess')
      return None
    if self._wait > 20:
      # 못 물었다 — 다시 접근한다(정렬이 어긋났거나 손가락이 얹혔다).
      self._enter('approach')
      return None
    return self._act(self._grasp_x(env), self._hold_y, _CLOSE)

  def _phase_assess(self, env):
    """접촉 길이와 지렛대로 안전 속도를 정하고, 재파지 여부를 결정한다."""
    contact = env.contact_length()
    arm = abs(self.by - self.pad_center)
    speed = self._speed_cap(env, contact, arm)
    self.grasp_contacts.append(contact)
    self.grasp_arms.append(arm)
    self.grasp_speeds.append(speed)
    self._speed_hold = speed

    spot = self._regrasp_spot(env) if (
        self.allow_regrasp and not self.regrasped) else None
    if spot is not None and self._regrasp_pays_off(env, spot, speed):
      self.regrasped = True
      self.n_regrasps += 1
      self._start_transport(spot, env._support_y(spot), 'approach')
    else:
      self._start_transport(env.tgt_box.center_x,
                            env.tgt_box.inner_floor_y, 'verify')
    return None

  def _start_transport(self, dest_x: float, dest_floor: float,
                       after: str) -> None:
    self._enter('transport')
    self._sub = 'lift'
    self._dest_x = float(dest_x)
    self._dest_floor = float(dest_floor)
    self._after = after

  def _phase_transport(self, env):
    """들어올림 -> 수평 이동 -> 내려놓음 -> 놓기. 목적지만 바꿔 재파지에도 쓴다."""
    s = self._speed_hold
    if self._sub == 'lift':
      ty = self._travel_y_hold(env)
      if self.ey <= ty + _REACH_TOL:
        self._sub = 'across'
        return None
      return self._goto(self.ex, ty, s, _CLOSE)

    if self._sub == 'across':
      ty = self._travel_y_hold(env)
      if abs(self.ex - self._dest_x) <= _REACH_TOL:
        self._sub = 'lower'
        self._prev_y = None
        self._stall = 0
        return None
      return self._goto(self._dest_x, ty, s, _CLOSE)

    if self._sub == 'lower':
      if self.block_bottom >= self._dest_floor - _REACH_TOL or self._stalled():
        self._sub = 'release'
        self._wait = 0
        return None
      ty = self.ey + (self._dest_floor - self.block_bottom)
      return self._goto(self._dest_x, ty, s, _CLOSE)

    # release: 손가락을 열고, 블록이 자리를 잡으면 위로 물러난다.
    self._wait += 1
    if self._wait <= 4:
      return self._act(self._dest_x, self.ey, _OPEN)
    ty = self._travel_y_free(env)
    if self.ey > ty + _REACH_TOL:
      return self._goto(self.ex, ty, self.speed, _OPEN)
    self._enter(self._after)
    return None

  def _phase_verify(self, env):
    """타겟 박스에 놓였는지 확인하고, 아니면 처음부터 다시 한다.

    성공했다면 `env.step()`이 이 스텝에서 종료를 알리므로 이 대기 동작은
    버려진다. 확인 없이 끝내면 튕겨 나간 경우에 타임아웃으로 교착한다.
    """
    self._wait += 1
    settled = (env.tgt_box.contains_x(self.bx)
               and self.block_bottom >= env.tgt_box.inner_floor_y
               - env.cfg.settle_tol)
    if settled:
      return self._act(self.ex, self.ey, _OPEN)
    if self._wait > 5:
      self._enter('approach')
      return None
    return self._act(self.ex, self.ey, _OPEN)

  # ================================================================ 액션 생성
  def _act(self, tx: float, ty: float, grip: float) -> np.ndarray:
    return np.array([tx, ty, 0.0, grip], dtype=np.float32)

  def _goto(self, tx: float, ty: float, s: float, grip: float) -> np.ndarray:
    """목표를 현재 위치에서 축별로 최대 `s`(mm)만큼 앞에 둔 액션.

    액션이 **절대 위치**이므로 목표를 멀리 둘수록 PD가 큰 힘을 내고 가속도가
    커진다(`a = k_p * lead`). 목표를 항상 `lead`만큼 앞에 두면 PD는 정상상태
    속도 `k_p*lead/k_v = 10*lead mm/s`, 즉 제어 스텝당 `lead` mm로 수렴하므로
    `lead`는 곧 순항 속도이기도 하다.

    축을 나누는 이유: 회전 토크를 만드는 것은 **수평** 가속뿐이고(지렛대가
    연직 방향이라 연직 관성력은 모멘트를 만들지 않는다), 수직은 미끄러짐
    한계가 따로 있다. 그래서 잡은 상태에서는 `s`를 축별로 다르게 준다.

    `grip`이 닫힘이면 **하중 처짐**을 앞먹임으로 보상한다. 그리퍼는 제 무게만
    중력 보상하므로(은닉 질량의 관측 단서를 남기기 위한 설계) 블록을 들면
    베이스가 목표보다 `m_obj*g/(M*k_p)`만큼 아래로 처진 채 평형이 된다. 이걸
    보상하지 않으면 명령 리드를 처짐이 다 먹어 **들어올릴 수가 없다**(실측:
    리드 5.9mm로 초당 15mm밖에 못 올라가 200스텝 타임아웃).
    """
    dx = float(np.clip(tx - self.ex, -s, s))
    dy = float(np.clip(ty - self.ey, -self._s_lift, self._s_lift))
    return self._act(self.ex + dx, self.ey + dy - self._sag(grip), grip)

  def _sag(self, grip: float) -> float:
    """잡은 상태에서 예상되는 하중 처짐(mm). 열려 있으면 0이다."""
    return self._sag_ff if grip > 0.5 else 0.0

  def _stalled(self) -> bool:
    """하강이 더 이상 진행되지 않는지(벽/바닥에 걸렸는지) 본다."""
    y = self.ey
    if self._prev_y is not None and abs(y - self._prev_y) < 0.3:
      self._stall += 1
    else:
      self._stall = 0
    self._prev_y = y
    return self._stall >= 4

  # ================================================================== 기하
  def _travel_y_free(self, env) -> float:
    """열린 손가락으로 수평 이동할 때의 베이스 y.

    손가락 끝이 블록 상단과 두 박스 rim **모두**보다 위를 지나야 한다.
    """
    cfg = env.cfg
    top = min(self.block_top, env.src_box.rim_y, env.tgt_box.rim_y)
    return top - _CLEARANCE - cfg.finger_length

  def _travel_y_hold(self, env) -> float:
    """블록을 든 채 수평 이동할 때의 베이스 y — 블록 바닥이 rim 위를 지난다."""
    hang = self.block_bottom - self.ey
    return min(env.src_box.rim_y, env.tgt_box.rim_y) - _CLEARANCE - hang

  def _com_base_y(self, env) -> float:
    """패드 중심을 블록 무게중심 높이에 맞추는 베이스 y.

    지렛대(`arm`)가 0이 되는 이상적인 파지 높이다. 벽에 막혀 여기까지 못
    내려가는 것이 곧 "얕은 파지"다.
    """
    cfg = env.cfg
    return self.by - (cfg.finger_length - cfg.pad_length / 2.0)

  def _can_reach_com(self, env, x: float) -> bool:
    """베이스 x가 `x`일 때 무게중심 높이까지 내려갈 수 있는지."""
    return env.max_descend_y(x) >= self._com_base_y(env)

  def _grasp_x(self, env) -> float:
    """파지할 베이스 x.

    기본은 블록 중심이다. 다만 블록 중심에서는 벽에 막혀 얕게밖에 못 무는데
    **옆으로 조금 비키면** 깊게 내려갈 수 있는 경우가 있다(넓은 박스인데 블록이
    벽 쪽에 붙어 있는 경우). 손가락이 블록 위에 얹히지 않는 범위
    (`_max_offset`) 안에서만 비킨다.
    """
    if self._grasp_x_cache is not None:
      return self._grasp_x_cache
    x = self.bx
    if not self._can_reach_com(env, x):
      lim = self._max_offset(env)
      for d in np.linspace(0.0, lim, 9)[1:]:
        for cand in (x - d, x + d):
          if self._can_reach_com(env, cand):
            x = cand
            break
        else:
          continue
        break
    self._grasp_x_cache = float(x)
    return self._grasp_x_cache

  def _max_offset(self, env) -> float:
    """열린 손가락이 블록 위에 얹히지 않는 최대 x 오프셋(mm)."""
    return max(0.0, env.cfg.finger_opening_max / 2.0 - self.block_half_x
               - _SIDE_MARGIN)

  def _grasp_base_y(self, env, x: float) -> float:
    """하강 목표 베이스 y — 무게중심 높이까지, 벽이 막으면 막히는 데까지."""
    return min(self._com_base_y(env), env.max_descend_y(x))

  # ================================================================ 속도 선택
  def _hidden(self, env) -> Tuple[float, float]:
    """속도 계산에 쓸 (질량, 마찰).

    `privileged=True`면 실제 은닉값을, 아니면 `config`의 **설계 범위**에서
    위험 분위수를 쓴다. 설계 범위는 에피소드별 표본이 아니라 사전 지식이므로
    비특권 모드에서 참조해도 관측-행동 관계가 오염되지 않는다.
    """
    cfg = self.cfg
    if self.privileged:
      info = env._info()
      return float(info['mass']), float(info['friction'])
    a = self.risk_alpha
    m_lo, m_hi = cfg.object_mass_range
    mu_lo, mu_hi = cfg.object_friction_range
    return (m_lo + (1.0 - a) * (m_hi - m_lo), mu_lo + a * (mu_hi - mu_lo))

  def _speed_cap(self, env, contact_len: float, arm: float) -> float:
    """현재 파지에서 안전한 명령 거리 `s`(mm) 상한.

    회전(= 그립 안에서 블록이 돌아 빠짐):
        `m * (k_p*s) * arm <= tip_torque_capacity(L, mu) / torque_margin`
    수직 미끄러짐(= 아래로 빠짐):
        `m * (gravity + k_p*s) <= 2 * mu * grip_force`

    `arm`의 하한을 `pad_length/2`로 둔다 — 접촉력이 패드 전체에 분포하므로
    패드 반길이보다 짧은 지렛대는 물리적으로 의미가 없고, 두지 않으면
    `arm -> 0`에서 상한이 발산한다.
    """
    cfg = self.cfg
    mass, mu = self._hidden(env)
    d = max(arm, cfg.pad_length / 2.0)
    tau = cfg.tip_torque_capacity(max(contact_len, 1e-3), mu)
    s_tip = tau / (self.torque_margin * cfg.k_p * mass * d)
    s_slip = (2.0 * mu * cfg.grip_force / mass - cfg.gravity) / cfg.k_p
    return float(max(0.3, min(self.speed, s_tip, s_slip) * self._backoff))

  # ================================================================== 재파지
  def _regrasp_spot(self, env) -> Optional[float]:
    """무게중심 높이까지 내려갈 수 있는 재파지 자리(블록을 놓을 x).

    후보는 (1) 두 박스 사이 바닥, (2) 소스/타겟 박스 안쪽이다. 어느 쪽이든
    **그리퍼가 끝까지 내려갈 수 있는 x**여야 하고, 블록이 벽에 걸치지 않고
    설 자리가 있어야 한다. 타겟 박스에 가까운 자리를 고른다(이후 운반이 짧다).
    """
    cfg = env.cfg
    pad = self.block_half_x + 3.0
    spans = [(env.src_box.left_inner + pad, env.src_box.right_inner - pad),
             (env.tgt_box.left_inner + pad, env.tgt_box.right_inner - pad),
             (env.src_box.right_outer + pad, env.tgt_box.left_outer - pad)]
    half = cfg.gripper_outer_width / 2.0
    best = None
    for lo, hi in spans:
      if hi <= lo:
        continue
      for x in np.linspace(hi, lo, 24):
        x = float(np.clip(x, half, cfg.world_width - half))
        if lo <= x <= hi and self._can_reach_com(env, x):
          if best is None or x > best:
            best = x
          break
    return best

  def _regrasp_pays_off(self, env, spot: float, speed: float) -> bool:
    """재파지가 이득인지 **완료까지 예상 스텝 수**로 판단한다.

    직접 운반은 지금의 낮은 안전 속도로 끝까지 가야 하고, 재파지는 옮기는
    비용을 치른 뒤 무게중심 파지의 빠른 속도로 간다. 남은 스텝 예산을
    넘기면(= 타임아웃) 직접 운반은 불가능한 것으로 본다.
    """
    cfg = env.cfg
    tgt = env.tgt_box.center_x
    fast = self._deep_speed(env)
    direct = self._eta(env, self.ex, tgt, speed)
    via = (self._eta(env, self.ex, spot, speed)
           + self._eta(env, spot, tgt, fast) + self._pick_cost(env, fast))
    budget = cfg.max_steps - env._t
    return via < direct or direct > budget

  def _deep_speed(self, env) -> float:
    """무게중심 높이에서 다시 잡았을 때 기대되는 안전 속도."""
    return self._speed_cap(env, env.cfg.pad_length, 0.0)

  def _eta(self, env, x0: float, x1: float, speed: float) -> float:
    """`x0`에서 집어 `x1`에 내려놓기까지의 예상 스텝 수.

    올림/내림 높이는 rim 여유에서 나오고, 상수 항은 파지·해제 대기(약 8스텝)다.
    """
    rise = abs(self._travel_y_hold(env) - self.ey) + abs(
        self._travel_y_hold(env) - self.ey)
    return (abs(x1 - x0) + rise) / max(speed, 0.1) + 8.0

  def _pick_cost(self, env, speed: float) -> float:
    """재파지 자리에서 다시 접근·하강·파지하는 데 드는 예상 스텝 수."""
    reach = abs(self._travel_y_free(env) - self._com_base_y(env))
    return 2.0 * reach / max(speed, 0.1) + 8.0
