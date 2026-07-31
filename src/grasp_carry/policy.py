r"""`GraspCarry2D`용 스크립트 정책 (phase 3, step 3).

이 태스크의 난이도는 **소스 박스가 좁다**는 데서 온다. 박스가 좁으면 그리퍼가
rim 아래로 못 내려가 블록 **윗부분만** 물게 되고, 패드 접촉 중심과 무게중심
사이에 지렛대(`arm`)가 남는다. 수평 가속 `a_h`는 그 지렛대에서 `m*a_h*arm`의
모멘트를 만들고, 그것이 패드의 회전 저항을 넘으면 블록이 그립 안에서 돌아 빠진다.

그래서 정책은 두 가지를 한다.

1. **속도 선택.** 액션이 **절대 목표 위치**이므로 "목표를 현재 위치에서 얼마나
   멀리 두는가"(리드 `s`, mm)가 곧 명령 가속도다(`a = k_p * s`). 아래
   `_speed_cap()`이 파지의 회전 토크 여유에서 `s`의 상한을 유도한다.
2. **재파지.** 회전 여유가 명목 가속도조차 못 버티면(= 얕은 파지), 블록을
   **그리퍼가 무게중심 높이까지 내려갈 수 있는 자리**(주로 두 박스 사이 바닥,
   때로는 소스 박스 안의 넓은 지점)로 옮겨 놓고 다시 깊게 잡는다.

## 회전 저항의 유도 (임의 상수를 쓰지 않기 위한 근거)

패드 1개가 접촉 길이 `L`에 압력을 고르게 걸고 있다고 보면, 접선(=연직·회전
방향) 트랙션 밀도의 상한은 `mu*N/L`이다. 그중 일부는 블록을 **중력에 대해
붙들고 있는 데** 이미 쓰인다(패드당 `m*(g+a_v)/2`). 남는 밀도로 낼 수 있는
회전 모멘트는 패드 2개를 합쳐

    T_resist = ( mu*N - m*(g+a_v)/2 ) * L / 2                    ... (1)

이다(각 패드가 `∫|y| * 밀도 dy = 밀도 * L^2/4`를 내고, 두 개를 더한 값).
`cfg.tip_torque_capacity(L, mu) = mu*N*L`은 접촉력 **전부**가 최대 지렛대
`L`에서 작용하는 상한이므로, (1)은 그 상한을 압력 분포와 중력 소모분으로
보정한 값이다. 여기에 걸리는 부하 모멘트는 `m*(k_p*s)*arm`이므로

    s <= T_resist / ( safety * k_p * m * arm )                   ... (2)

가 명령 리드의 상한이다. **"얕다"의 정의도 여기서 나온다** — 임의의 접촉 길이
임계값이 아니라, (2)의 상한이 로봇의 명목 리드(`max_accel/k_p`)보다 작으면
그 파지는 얕은 것이다.

## 이전 시도에서 실측으로 확인돼 반영된 것

(`experiments/2026-07-31_grasp-carry-env.md`)

- **접근 높이가 낮으면 열린 손가락이 블록 옆구리를 쳐서** 정렬이 끝나지 않는다.
  수평 이동은 손가락 끝이 블록 상단과 두 rim **모두** 위를 지나는 높이에서만 한다.
- **낙하 후 감속 적응이 없으면** 감당 못 하는 블록에서 무한히 놓쳐 에피소드가
  끝나지 않는다.
- 그리퍼는 **제 무게만** 중력 보상한다(은닉 질량의 관측 단서를 남기기 위한
  설계). 그래서 블록을 들면 베이스가 목표보다 `m*g/(M*k_p)`(최대 28.6mm)만큼
  처진 채 평형이 된다. 이걸 보상하지 않으면 명령 리드를 처짐이 다 먹어 **들어
  올릴 수가 없다**. 은닉 질량을 모르므로 관측되는 높이 오차로 적분 보상한다
  (`_sag_ff`).

**은닉 물성(질량·마찰)은 `privileged=True`일 때만 본다.** 기본값에서는 `config`의
설계 범위(에피소드별 표본이 아니라 사전 지식)에서 불리한 쪽 분위수를 쓴다.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from .config import CarryConfig

# 손가락 끝/블록 바닥이 rim 위를 지날 때 남길 기하 여유(mm). 태스크 난이도와
# 무관한 순수 충돌 회피 값이다.
_CLEARANCE = 12.0
# 위치 도달 판정 허용오차(mm).
_REACH_TOL = 3.0
# 열린 손가락 안쪽 면과 블록 옆면 사이에 남길 여유(mm). 파지 x를 옆으로
# 비킬 수 있는 한계(`_max_offset`)와 가능 구간(`_feasible_x`)을 함께 정한다.
_FINGER_CLEAR = 3.0
# 하중 처짐 적분 보상의 이득. 1보다 작아야 진동하지 않고, 너무 작으면
# 수렴 전에 rim에 걸린다. 0.6이면 3스텝 안에 90% 수렴한다.
_SAG_GAIN = 0.6
# 베이스가 올라갈 수 있는 최고 높이(mm) — 월드 밖으로 나가지 않게 한다.
_CEILING_Y = 20.0
# 월드 양끝 벽에서 띄울 여유(mm). `_feasible_x` 참고 — 활짝 벌린 손가락이
# 벽에 닿으면 그리퍼가 마찰로 고착된다.
_WALL_MARGIN = 3.0
# 박스 안쪽 벽에서 띄울 여유(mm). 월드 벽보다 크게 잡을 수 있다 — 박스 안으로
# 들어갈지 말지는 선택의 문제지만, 월드 벽 쪽은 블록이 거기 스폰되면 피할 길이
# 없어 여유를 최소로 써야 한다(`_feasible_x`).
_BOX_MARGIN = 5.0
# 블록 위치 오차를 베이스 목표로 옮길 때의 이득(`_base_x_for_block`). 1보다
# 작아야 잡은 블록이 손가락 홈 안에서 출렁일 때 오버슛하지 않는다.
_TRACK_GAIN = 0.5

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
      명목 명령 리드(mm/스텝)의 상한. 액션이 절대 위치이므로 리드 `s`는 곧
      명령 가속도 `k_p*s`이자(PD 정상상태에서) 순항 속도 `s mm/스텝`이다.
      `None`이면 `cfg.max_accel / cfg.k_p`(로봇 최대 가속도에 해당하는 리드).
  allow_regrasp:
      `False`면 `relocate` phase에 절대 진입하지 않는다(직접 운반만).
  privileged:
      `True`면 `info`의 은닉 물성(질량·마찰)을 실제값으로 읽어 속도를 고른다.
      `False`(기본)면 `config`의 설계 범위에서 불리한 쪽 분위수를 쓴다. 같은
      관측에서 다른 행동을 만들어(다봉성) 데이터를 수집하기 위한 스위치다.
  risk_alpha:
      비특권 모드의 위험 분위수. 0.1이면 "질량 90분위 / 마찰 10분위"다.
  torque_safety:
      식 (2)의 안전계수. (1)은 압력이 접촉 길이에 **고르게** 분포한다고 보지만
      실제 접촉은 블록 모서리 쪽에 몰리고, 접촉 길이 자체도 기울어진 블록에서는
      근사값이다. 1.0이면 유도식 그대로다.
  drop_backoff:
      낙하 1회마다 속도 상한에 곱하는 계수. 없으면 감당 못 하는 블록에서
      무한히 놓친다(실측된 교착 모드).
  """

  #: 정책이 거치는 phase. `carry`/`place`/`relocate`는 `_sub`로 세분된다.
  PHASES = ('approach', 'descend', 'grasp', 'assess',
            'relocate', 'carry', 'place', 'verify')

  #: 운반 이외의 동작(들어올림·내려놓음·파지·해제·정착)에 남겨둘 스텝 예산.
  #: 타임아웃 대신 (허용된 실패인) 넘어짐을 택하기 위한 마감 시한 계산에 쓴다.
  _RESERVE = 32

  def __init__(self, speed: Optional[float] = None, allow_regrasp: bool = True,
               privileged: bool = False, risk_alpha: float = 0.1,
               torque_safety: float = 1.0, drop_backoff: float = 0.6,
               config: Optional[CarryConfig] = None):
    self.cfg = config or CarryConfig()
    cfg = self.cfg
    self.speed = float(speed) if speed is not None else cfg.max_accel / cfg.k_p
    self.allow_regrasp = bool(allow_regrasp)
    self.privileged = bool(privileged)
    self.risk_alpha = float(risk_alpha)
    self.torque_safety = float(torque_safety)
    self.drop_backoff = float(drop_backoff)

    # 설계 범위의 가장 무거운 블록에서 생기는 처짐(mm). 연직 리드는 이보다
    # 커야 어떤 블록이든 들어올릴 수 있다.
    self._sag_max = (cfg.object_mass_range[1] * cfg.gravity
                     / (cfg.assembly_mass * cfg.k_p))
    # 연직 리드 = 최악 처짐 + 명목 리드. 앞의 항이 하중을 상쇄하고 뒤의 항이
    # 실제 상승 가속도를 만든다.
    self._lift_lead = self._sag_max + self.speed
    # 패드 중심이 베이스 원점에서 얼마나 아래인지(mm).
    self._pad_drop = cfg.finger_length - cfg.pad_length / 2.0
    self.reset()

  # =================================================================== 상태
  def reset(self) -> None:
    """에피소드 시작 상태로 되돌린다."""
    self.phase = 'approach'
    self._sub = 'lift'
    self._step_i = 0
    self._dest_x = 0.0
    self._dest_floor = 0.0
    self._dest_reserve = self._RESERVE
    self._wait = 0
    self._lost = 0
    self._stall = 0
    self._retry = 0
    self._prev_y: Optional[float] = None
    self._hold_y = 0.0
    self._hold_x = 0.0
    self._grasp_x_cache: Optional[float] = None
    self._speed_hold = self.speed
    self._sag_ff = 0.0
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
    """`env`의 현재 상태를 보고 절대 목표 pose 액션을 낸다."""
    self._sense(env)
    self._watch_drop(env)
    # phase 전이는 액션 없이 연쇄될 수 있다(예: descend -> grasp -> assess).
    for _ in range(len(self.PHASES) + 2):
      action = getattr(self, '_phase_' + self.phase)(env)
      if action is not None:
        self._step_i += 1
        return action
    self._step_i += 1
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
    carrying = (self._sub != 'release'
                and self.phase in ('carry', 'relocate', 'place'))
    if not carrying or env.is_held():
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
    self._sub = 'lift'
    self._wait = 0
    self._stall = 0
    self._prev_y = None
    self._sag_ff = 0.0
    if phase == 'approach':
      self._grasp_x_cache = None

  def _phase_approach(self, env):
    """열린 손가락으로 블록 위 안전 높이에 정렬한다."""
    ty = self._travel_y_free(env)
    if self.ey > ty + _REACH_TOL:
      # 먼저 **위로**. 낮은 높이에서 수평 이동하면 열린 손가락이 블록 옆구리를
      # 쳐서 정렬이 끝나지 않는다(실측된 교착 모드).
      return self._goto(env, self.ex, ty, self.speed, _OPEN)
    gx = self._grasp_x(env)
    if abs(self.ex - gx) > _REACH_TOL:
      return self._goto(env, gx, ty, self.speed, _OPEN)
    self._enter('descend')
    return None

  def _phase_descend(self, env):
    """벽이 허용하는 만큼(최대 무게중심 높이까지) 내려간다."""
    gx = self._grasp_x(env)
    # 손가락이 아직 다 안 벌어졌으면 내려가지 않는다 — 반쯤 닫힌 채 내려가면
    # 손가락이 블록 옆구리를 쳐서 블록을 넘어뜨린다(실측된 실패 모드).
    # 좌우 여유 자체는 `_grasp_x`가 이미 확보해 두었으므로, 여기서는 개도가
    # 그 계산의 전제(완전 개방)에 도달했는지만 본다. 기다리는 동안에도 x는
    # 계속 목표로 몰아야 한다 — 제자리에 얼어붙으면 그대로 교착한다.
    if env.gripper.gap < env.cfg.finger_opening_max - 1.0:
      return self._goto(env, gx, self.ey, self.speed, _OPEN)
    goal = self._grasp_base_y(env, gx)
    if self.ey >= goal - 2.0 or self._stalled():
      self._hold_y = self.ey
      self._enter('grasp')
      return None
    return self._goto(env, gx, goal, self.speed, _OPEN)

  def _phase_grasp(self, env):
    """손가락을 닫고 양쪽 패드가 물릴 때까지 기다린다."""
    self._wait += 1
    if env.is_held() and self._wait >= 4:
      self._enter('assess')
      return None
    if self._wait > 18:
      # 못 물었다 — 정렬이 어긋났거나 손가락이 블록 위에 얹혔다. 파지 x를
      # 조금 흔들어 대칭을 깨고 다시 접근한다.
      self._retry += 1
      self._enter('approach')
      return None
    return self._act(self._grasp_x(env), self._hold_y, _CLOSE)

  def _phase_assess(self, env):
    """접촉 길이·지렛대로 안전 속도를 정하고, 재파지 여부를 결정한다."""
    contact = env.contact_length()
    arm = abs(self.by - self.pad_center)
    speed = self._speed_cap(env, contact, arm)
    self.grasp_contacts.append(contact)
    self.grasp_arms.append(arm)
    self.grasp_speeds.append(speed)
    self._speed_hold = speed

    if self._wants_regrasp(env, speed):
      spot = self._regrasp_spot(env)
      if spot is not None and self._regrasp_fits_budget(env, spot):
        self.regrasped = True
        self.n_regrasps += 1
        self._start_transport('relocate', spot, env._support_y(spot))
        return None
    self._start_transport('carry', env.tgt_box.center_x,
                          env.tgt_box.inner_floor_y)
    return None

  def _start_transport(self, phase: str, dest_x: float,
                       dest_floor: float) -> None:
    # 재파지는 옮기고 끝이 아니라 그 뒤에 "다시 집어 타겟까지 운반"이 남으므로
    # 마감 시한 계산에 남길 예산이 두 배다.
    reserve = self._RESERVE * (2 if phase == 'relocate' else 1)
    self._enter(phase)
    self._dest_x = float(dest_x)
    self._dest_floor = float(dest_floor)
    self._dest_reserve = reserve

  # ------------------------------------------------------- 운반 (공통 동작)
  def _phase_carry(self, env):
    """들어올려 타겟 박스 위까지 옮긴다. 도착하면 `place`."""
    return self._transit(env, then='place')

  def _phase_relocate(self, env):
    """재파지 자리로 옮겨 내려놓는다. 끝나면 다시 `approach`부터 한다."""
    if self._sub in ('lift', 'across'):
      return self._transit(env, then=None)
    return self._settle(env, then='approach')

  def _phase_place(self, env):
    """타겟 박스에 내려놓고 놓는다. 끝나면 `verify`."""
    return self._settle(env, then='verify')

  def _transit(self, env, then: Optional[str]):
    """`lift` -> `across`. 도착하면 `then`(None이면 같은 phase의 `lower`)로."""
    s = self._safe_speed(env)
    ty = self._travel_y_hold(env)
    if self._sub == 'lift':
      if self.ey <= ty + _REACH_TOL:
        self._sub = 'across'
        return None
      return self._goto(env, self._base_x_for_block(env, self.bx), ty, s,
                        _CLOSE)
    tx = self._base_x_for_block(env, self._dest_x)
    # 블록이 목적지에 닿았거나, **베이스가 이미 낼 수 있는 데까지 다 낸** 경우
    # 다음 단계로 간다. 뒤 조건이 없으면, 블록이 그립 안에서 비켜 잡힌 채로
    # 목적지가 작업공간 밖으로 밀려나 영원히 도달 판정이 안 된다(실측된
    # 타임아웃 원인).
    if (abs(self.bx - self._dest_x) > _REACH_TOL
        and abs(self.ex - tx) > _REACH_TOL):
      return self._goto(env, tx, ty, s, _CLOSE)
    dest = (self._dest_x, self._dest_floor, self._dest_reserve)
    if then is not None:
      self._enter(then)
      self._dest_x, self._dest_floor, self._dest_reserve = dest
    self._sub = 'lower'
    self._prev_y, self._stall = None, 0
    return None

  def _settle(self, env, then: str):
    """`lower` -> `release`. 놓고 물러나면 `then` phase로 넘어간다."""
    if self._sub == 'lower':
      if (self.block_bottom >= self._dest_floor - _REACH_TOL
          or self._stalled()):
        self._sub = 'release'
        self._wait = 0
        self._hold_x = self.ex
        return None
      ty = self.ey + (self._dest_floor - self.block_bottom)
      return self._goto(env, self._base_x_for_block(env, self._dest_x), ty,
                        self._safe_speed(env), _CLOSE)

    # release: 손가락을 열고, 블록이 자리를 잡으면 위로 물러난다. x는 **고정된**
    # 값으로 잡는다 — 현재 위치를 그대로 되돌려주면, 놓다가 기울어진 블록이
    # 손가락을 밀 때 그리퍼가 밀리는 대로 흘러가 월드 벽에 고착된다(실측).
    self._wait += 1
    if self._wait <= 4:
      return self._act(self._hold_x, self.ey, _OPEN)
    ty = self._travel_y_free(env)
    if self.ey > ty + _REACH_TOL:
      return self._goto(env, self._hold_x, ty, self.speed, _OPEN)
    self._enter(then)
    return None

  def _base_x_for_block(self, env, block_x: float) -> float:
    """블록을 `block_x`에 두려면 베이스를 어디에 둬야 하는지.

    손가락은 `GrooveJoint`로 베이스에 대해 **각자 독립적으로** 미끄러지므로,
    잡은 블록과 두 손가락은 베이스에 대해 좌우로 미끄러지는 하나의 덩어리가
    된다. 그래서 블록 중심이 베이스 중심과 어긋난 채로 유지되고(실측: 20mm
    이상), 목적지를 베이스 기준으로 주면 그만큼 어긋난 곳에 내려놓게 된다.

    그렇다고 목표를 `block_x + (ee_x - block_x)`로 잡으면 **안 된다**: 그 오프셋
    자체가 상태(덩어리의 미끄러진 정도)이고 가속에 따라 출렁이므로, 목표가
    출렁임을 그대로 되먹여 수렴하지 않는 한계고리에 빠진다(실측: 블록이 목적지
    좌우로 ±16mm를 계속 왕복하며 200스텝 타임아웃). 대신 **블록의 위치 오차만큼
    베이스를 옮기는** 형태로 쓴다 — 오프셋을 추정하지 않고, 오차가 줄면 명령도
    저절로 멈춘다.
    """
    cfg = env.cfg
    half = cfg.gripper_outer_width / 2.0 + _WALL_MARGIN
    return float(np.clip(self.ex + _TRACK_GAIN * (block_x - self.bx),
                         half, cfg.world_width - half))

  def _phase_verify(self, env):
    """타겟 박스에 놓였는지 확인하고, 아니면 처음부터 다시 한다.

    성공했다면 `env.step()`이 종료를 알리므로 이 대기 동작은 버려진다. 확인
    없이 끝내면 튕겨 나간 경우에 타임아웃으로 교착한다.
    """
    self._wait += 1
    settled = (env.tgt_box.contains_x(self.bx)
               and self.block_bottom >= env.tgt_box.inner_floor_y
               - env.cfg.settle_tol)
    if settled or self._wait <= 4:
      return self._act(self.ex, self.ey, _OPEN)
    self._enter('approach')
    return None

  # ================================================================ 액션 생성
  def _act(self, tx: float, ty: float, grip: float) -> np.ndarray:
    return np.array([tx, ty, 0.0, grip], dtype=np.float32)

  def _goto(self, env, tx: float, ty: float, s: float,
            grip: float) -> np.ndarray:
    """목표를 현재 위치에서 축별로 제한한 리드만큼 앞에 둔 액션.

    수평 리드는 `s`(회전 토크 여유에서 유도된 값), 연직 리드는 `_lift_lead`로
    따로 제한한다. 회전 모멘트를 만드는 것은 **수평** 가속뿐이고(지렛대가 연직
    방향이라 연직 관성력은 모멘트를 만들지 않는다), 연직에는 대신 하중 처짐이라는
    다른 제약이 있기 때문이다.

    `grip`이 닫힘이면 관측되는 높이 오차로 처짐을 적분 보상한다.
    """
    if grip > 0.5:
      # 목표보다 아래에 있으면(ey > ty) 보상을 키운다. 은닉 질량을 모르므로
      # 관측 가능한 오차만으로 추정하는 적분기다.
      self._sag_ff = float(np.clip(
          self._sag_ff + _SAG_GAIN * (self.ey - ty), 0.0, self._sag_max))
      ty = ty - self._sag_ff
    ty = max(ty, _CEILING_Y)
    dx = float(np.clip(tx - self.ex, -s, s))
    dy = float(np.clip(ty - self.ey, -self._lift_lead, self._lift_lead))
    return self._act(self.ex + dx, self.ey + dy, grip)

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
    top = min(self.block_top, env.src_box.rim_y, env.tgt_box.rim_y)
    return max(_CEILING_Y, top - _CLEARANCE - env.cfg.finger_length)

  def _travel_y_hold(self, env) -> float:
    """블록을 든 채 수평 이동할 때의 베이스 y — 블록 바닥이 rim 위를 지난다."""
    hang = self.block_bottom - self.ey
    top = min(env.src_box.rim_y, env.tgt_box.rim_y)
    return max(_CEILING_Y, top - _CLEARANCE - hang)

  def _com_base_y(self, env, com_y: Optional[float] = None) -> float:
    """패드 중심을 블록 무게중심 높이에 맞추는 베이스 y.

    지렛대(`arm`)가 0이 되는 이상적인 파지 높이다. 벽에 막혀 여기까지 못
    내려가는 것이 곧 "얕은 파지"다.
    """
    return (self.by if com_y is None else com_y) - self._pad_drop

  def _can_reach_com(self, env, x: float,
                     com_y: Optional[float] = None) -> bool:
    """베이스 x가 `x`일 때 무게중심 높이까지 내려갈 수 있는지."""
    return self._descend_limit(env, x) >= self._com_base_y(env, com_y)

  def _descend_limit(self, env, x: float) -> float:
    """정책이 쓰는 하강 한계 — `env.max_descend_y`에 **벽 여유**를 더한 것.

    환경의 기구학 한계는 그리퍼 바깥폭이 박스 개구부에 `_FIT_CLEARANCE`(2mm)만
    들어가면 박스 안까지 내려갈 수 있다고 본다. 그런데 손가락은 열릴 때
    `grip_force`(12N)로 바깥을 밀기 때문에, 2mm 여유로 들어가면 손가락 바깥
    면이 박스 벽에 눌리고 그 마찰이 EE 힘 상한을 넘어 **그리퍼가 벽에 고착
    된다**(실측된 타임아웃 원인 — 벌린 채로 다시 못 올라온다). 여유가 모자라는
    박스에는 아예 들어가지 않고 rim 위까지만 내려간다.
    """
    cfg = env.cfg
    half = cfg.gripper_outer_width / 2.0
    lo, hi = x - half, x + half
    limit = env.max_descend_y(x)
    for box in (env.src_box, env.tgt_box):
      if hi < box.left_outer or lo > box.right_outer:
        continue
      if not (lo >= box.left_inner + _BOX_MARGIN
              and hi <= box.right_inner - _BOX_MARGIN):
        limit = min(limit, box.rim_y - cfg.finger_length)
    return float(limit)

  def _grasp_x(self, env) -> float:
    """파지할 베이스 x.

    기본은 블록 중심이다. 다만 블록 중심에서는 벽에 막혀 얕게밖에 못 무는데
    **옆으로 조금 비키면** 깊게 내려갈 수 있는 경우가 있다(박스가 넓은데 블록이
    벽 쪽에 붙어 있는 경우). 손가락이 블록 위에 얹히지 않는 범위(`_max_offset`)
    안에서만 비킨다. 파지에 실패해 다시 올 때마다 조금씩 흔들어(`_retry`)
    같은 자리에서 무한히 실패하는 것을 막는다.
    """
    if self._grasp_x_cache is not None:
      return self._grasp_x_cache
    lim = self._max_offset(env)
    best, best_depth = self.bx, self._descend_limit(env, self.bx)
    goal = self._com_base_y(env)
    if best_depth < goal:
      for d in np.linspace(0.0, lim, 13)[1:]:
        for cand in (self.bx - d, self.bx + d):
          depth = self._descend_limit(env, cand)
          if depth > best_depth:
            best, best_depth = float(cand), depth
        if best_depth >= goal:
          break
    if self._retry:
      jitter = ((-1.0) ** self._retry) * min(lim, 3.0 * self._retry)
      best = float(np.clip(best + jitter, self.bx - lim, self.bx + lim))
    self._grasp_x_cache = self._feasible_x(env, best)
    return self._grasp_x_cache

  def _feasible_x(self, env, x: float) -> float:
    """파지 가능한 베이스 x 구간 안으로 `x`를 되돌린다.

    구간은 서로 다른 두 물리 제약의 교집합이다.

    1. **월드 벽 여유.** `env.step`의 액션 클립 한계
       (`gripper_outer_width/2`)에 정확히 붙이면 활짝 벌린 손가락 바깥 면이
       월드 벽에 닿는다. 손가락은 `grip_force`(12N)로 벽을 밀고 있어 그 마찰이
       EE 힘 상한을 넘고, 그리퍼가 벽에 **마찰로 고착된다**(실측된 타임아웃
       원인). 그래서 몇 mm 더 띄운다.
    2. **손가락-블록 여유.** 열린 손가락 안쪽 면이 블록 옆면 바깥에 있어야
       내려갈 수 있다.

    소스 박스가 왼쪽 월드 벽에 붙어 있어서 블록이 그리퍼 중심의 도달 한계보다
    더 왼쪽에 스폰되면 두 구간이 **비어 있을 수** 있다. 그때는 양쪽 여유를
    절반씩 나눠 갖는 중점을 쓴다(둘 다 mm 단위 여유만 남지만, 벽에 붙이거나
    블록을 치는 것보다는 낫다). 정책이 이 클립을 아예 하지 않으면 도달 판정이
    영원히 참이 되지 않아 교착한다(실측된 타임아웃 원인).
    """
    cfg = env.cfg
    half_g = cfg.gripper_outer_width / 2.0
    half_open = cfg.finger_opening_max / 2.0
    lo = max(half_g + _WALL_MARGIN,
             self.bx + self.block_half_x - half_open + _FINGER_CLEAR)
    hi = min(cfg.world_width - half_g - _WALL_MARGIN,
             self.bx - self.block_half_x + half_open - _FINGER_CLEAR)
    if lo <= hi:
      return float(np.clip(x, lo, hi))
    return float(0.5 * (lo + hi))

  def _max_offset(self, env) -> float:
    """열린 손가락이 블록 위에 얹히지 않는 최대 x 오프셋(mm)."""
    return max(0.0, env.cfg.finger_opening_max / 2.0 - self.block_half_x
               - _FINGER_CLEAR)

  def _grasp_base_y(self, env, x: float) -> float:
    """하강 목표 베이스 y — 무게중심 높이까지, 벽이 막으면 막히는 데까지."""
    return min(self._com_base_y(env), self._descend_limit(env, x))

  # ================================================================ 속도 선택
  def _hidden(self, env) -> Tuple[float, float]:
    """속도 계산에 쓸 (질량, 마찰).

    `privileged=True`면 실제 은닉값을, 아니면 `config`의 **설계 범위**에서
    불리한 쪽 분위수를 쓴다. 설계 범위는 에피소드별 표본이 아니라 사전 지식이므로
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
    """이 파지에서 안전한 수평 명령 리드 `s`(mm)의 상한 — 모듈 문서의 식 (2).

    `contact_len`이 0이거나 마찰 트랙션이 전부 중력을 붙드는 데 쓰이면 회전
    여유가 없으므로 최소값을 준다.
    """
    cfg = self.cfg
    mass, mu = self._hidden(env)
    if contact_len <= 0.0:
      return self._min_lead()
    # (1) 패드 2개의 회전 저항 모멘트. 출발점은 config의 회전 저항 상한
    # `mu*N*L`(접촉력 **전부**가 최대 지렛대 L에서 작용하는 경우)이고, 거기에
    # 두 가지 보정을 건다: 압력이 접촉 길이에 고르게 분포한다는 가정(계수 1/2)과,
    # 접선 트랙션 중 블록을 중력에 대해 붙드는 데 이미 쓰이는 몫(패드당 m*g/2).
    t_resist = (0.5 * cfg.tip_torque_capacity(contact_len, mu)
                - 0.25 * mass * cfg.gravity * contact_len)
    if t_resist <= 0.0:
      return self._min_lead()
    if arm <= 1e-6:
      return self.speed * self._backoff
    s = t_resist / (self.torque_safety * cfg.k_p * mass * arm)
    return float(np.clip(s * self._backoff, self._min_lead(), self.speed))

  def _min_lead(self) -> float:
    """명령 리드의 하한(mm). 0이면 아예 움직이지 못해 교착한다."""
    return 0.4

  def _safe_speed(self, env) -> float:
    """이번 스텝에 쓸 수평 리드 — 안전 속도와 **마감 시한** 중 큰 쪽.

    안전 속도만 따르면 감당 못 하는 파지에서 200스텝을 넘겨 타임아웃이 난다.
    타임아웃은 "정책이 교착했다"는 뜻이라 넘어짐보다 나쁘다. 남은 예산으로
    목적지에 닿으려면 최소 얼마가 필요한지 계산해 그만큼은 낸다.
    """
    budget = self.cfg.max_steps - self._step_i - self._dest_reserve
    need = abs(self._dest_x - self.bx) / max(budget, 1.0)
    return float(np.clip(max(self._speed_hold, need),
                         self._min_lead(), self.speed))

  # ================================================================== 재파지
  def _wants_regrasp(self, env, speed: float) -> bool:
    """이 파지가 **얕은지** — 회전 여유가 명목 가속도조차 못 버티는지.

    임의의 접촉 길이 임계값이 아니라 식 (2)의 상한이 명목 리드에 못 미치는지로
    정의한다. `allow_regrasp=False`거나 이미 한 번 재파지했으면 하지 않는다.
    """
    if not self.allow_regrasp or self.regrasped:
      return False
    return speed < self.speed - 1e-9

  def _regrasp_spot(self, env) -> Optional[float]:
    """블록을 내려놓았을 때 **무게중심 높이까지 내려갈 수 있는** 자리(x).

    후보는 (1) 두 박스 사이 바닥, (2) 소스 박스 안의 넓은 지점이다. 타겟 박스
    안은 후보가 아니다 — 거기 놓는 것은 재파지가 아니라 태스크 완료다.
    블록이 벽에 걸치지 않고 설 자리가 있어야 하고, 그리퍼가 그 x에서 무게중심
    높이까지 내려갈 수 있어야 한다. 타겟에 가까운 자리를 고른다.
    """
    cfg = env.cfg
    # 블록 옆면과 박스 벽 사이 여유. 빠듯하게 잡으면 내려놓는 도중 블록이
    # 박스 벽 **바깥 모서리에 걸려** 기울고, 그리퍼는 그걸 끌다 교착한다
    # (실측: 여유 3mm에서 블록이 타겟 박스 벽에 얹혀 정지 -> 공중에서 해제 ->
    # 전도). 그래서 후보 구간 안에서도 **가운데에 가까운 x를 먼저** 고른다.
    half_b = self.block_half_x + _CLEARANCE
    half_g = cfg.gripper_outer_width / 2.0
    spans = [(env.src_box.right_outer + half_b, env.tgt_box.left_outer - half_b),
             (env.src_box.left_inner + half_b, env.src_box.right_inner - half_b)]
    for lo, hi in spans:
      if hi <= lo:
        continue
      mid = 0.5 * (lo + hi)
      for x in sorted(np.linspace(lo, hi, 41), key=lambda v: abs(v - mid)):
        x = float(x)
        if not (half_g <= x <= cfg.world_width - half_g):
          continue
        com_y = env._support_y(x) - env.block_h / 2.0
        if self._can_reach_com(env, x, com_y):
          return x
    return None

  def _regrasp_fits_budget(self, env, spot: float) -> bool:
    """재파지 계획이 남은 스텝 예산 안에 끝나는지 대략 확인한다.

    옮기기(현재 속도) + 다시 집기 + 타겟까지 운반(깊은 파지의 명목 속도)로
    나눠 센다. 예산을 넘으면 재파지를 포기하고 직접 운반한다 — 타임아웃보다는
    넘어질 위험을 감수하는 편이 낫다.
    """
    move = abs(spot - self.bx) / max(self._speed_hold, self._min_lead())
    back = abs(env.tgt_box.center_x - spot) / self.speed
    left = self.cfg.max_steps - self._step_i
    return move + back + 3.0 * self._RESERVE <= left
