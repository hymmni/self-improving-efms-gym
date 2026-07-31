"""`GraspCarry2D` 물리/태스크 설정값.

단위계 (모든 수치의 기준):
  길이 = mm  (렌더링 1px = 1mm)
  질량 = kg
  시간 = s
  => 힘   = kg·mm/s² = milli-Newton (mN).  12 N = 12_000
  => 중력 = 9.81 m/s² = 9810 mm/s²

이 모듈은 순수 설정값과, 다른 설정값으로부터 유도되는 계산(프로퍼티)만
담는다. pymunk Space/Body 등 물리 로직은 이후 step(1~2)의 범위다.

배경: 이전 시도(v1~v3, `src/grasp_carry_env.py`)는 픽셀 단위 파라미터를
임의 설정하고 반복 튜닝하다 폐기했다(`experiments/2026-07-31_grasp-carry-env.md`
"다음 단계" 이전 논의 참고). 이번 버전은 실제 SI 규격(ALOHA ViperX-300 그리퍼
스펙, 음료 캔 규격 등)에서 값을 유도해 근거 없는 튜닝을 배제한다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CarryConfig:
  """`GraspCarry2D`의 모든 물리/태스크 수치를 노출하는 설정.

  값은 하드코딩하지 않고, 근거를 필드별 주석으로 남긴다. 다른 필드에서
  기계적으로 유도되는 값(그리퍼 바깥 폭, EE 힘 상한 등)은 필드가 아니라
  아래 프로퍼티/메서드로 노출한다.
  """

  # --- 월드 --------------------------------------------------------------
  world_width: float = 512.0        # mm; 작업공간 약 0.5 m
  floor_y: float = 470.0            # mm; 바닥 높이
  gravity: float = 9810.0           # mm/s^2 = 9.81 m/s^2

  # --- 물체 (음료 캔 규격) -------------------------------------------------
  object_width_range: Tuple[float, float] = (60.0, 70.0)     # mm; 캔 지름 규격
  object_height_range: Tuple[float, float] = (100.0, 140.0)  # mm; 캔 높이 규격
  # 은닉 — 빈 캔(~30g) ~ 가득 찬 캔(250~500ml급, ~350g)
  object_mass_range: Tuple[float, float] = (0.030, 0.350)    # kg
  # 은닉 — 결로(젖음)로 미끄러운 알루미늄 ~ 마른 알루미늄-고무 패드 마찰
  object_friction_range: Tuple[float, float] = (0.25, 0.65)

  # --- 그리퍼 (ALOHA ViperX-300 스펙 기반) ---------------------------------
  # ALOHA 가반하중 750g 역산: mu=0.5일 때 N >= mg/(2*mu) = 7.4N, 여유 포함
  grip_force: float = 12_000.0        # mN = 12 N
  # ALOHA ViperX-300 공식 스펙 42~116mm에서 하한만 완화(사용자 결정) —
  # 얕은 파지를 강제할 수 있도록 더 좁은 개도까지 허용
  finger_opening_min: float = 20.0    # mm
  finger_opening_max: float = 116.0   # mm; ALOHA ViperX-300 공식 스펙 상한
  pad_length: float = 30.0            # mm; ALOHA 핑거 접촉면(패드) 길이
  finger_length: float = 90.0         # mm; ALOHA 핑거 전체 길이
  # 평행죠 핑거 단면 두께(개도 바깥으로 늘어나는 폭). ALOHA 핑거 어태치먼트의
  # 공식 CAD 치수는 비공개라, 평행죠 그리퍼 핑거의 통상적 단면 두께로 근사
  finger_thickness: float = 10.0      # mm
  # 손가락이 물리는 레일 플레이트 두께 / 위쪽 마운트 스템 치수. 공식 CAD가
  # 없어 Dynamixel XM430 하우징(28.5x46.5x34mm)과 마운트 브래킷 크기로 근사
  rail_plate_thickness: float = 14.0  # mm
  stem_width: float = 20.0            # mm
  stem_length: float = 40.0           # mm
  # 손가락 1개 질량 — 3D 프린트 핑거 어태치먼트 근사(어셈블리 질량에 포함)
  finger_mass: float = 0.030          # kg
  # 패드 표면 마찰. pymunk는 접촉 마찰을 sqrt(f_a*f_b)로 합성하므로, 패드를
  # 1.0으로 두면 접촉 마찰이 물체 쪽 shape 마찰만으로 결정된다(은닉 mu를
  # 물체 shape에 그대로 심을 수 있다).
  pad_friction: float = 1.0
  # 전 스트로크(개도 상한→하한)를 닫는 데 걸리는 시간. 평행죠 그리퍼의
  # 통상적 개폐 시간
  finger_close_time: float = 0.5      # s

  # --- EE 힘 상한 유도용 (가반하중 스펙) -----------------------------------
  payload_mass: float = 0.75          # kg; ALOHA 공식 가반하중 스펙
  # EE 어셈블리(그리퍼 서보 + 마운트 + 핑거) 근사 질량 — ALOHA 그리퍼
  # 서보(Dynamixel XM430, ~82g) + 3D 프린트 마운트/핑거 근사 총중량
  assembly_mass: float = 0.30         # kg
  # 협동로봇(cobot) 수준 최대 가속도 근사
  max_accel: float = 5000.0           # mm/s^2

  # --- 제어/물리 ------------------------------------------------------------
  # EE 임피던스 PD 강성(가속도 차원, 1/s^2). w_n = sqrt(k_p) = 20 rad/s
  # (~3.2Hz)로, 10Hz 명령을 추종할 만큼 빠르면서 물리 스텝보다 훨씬 느리다.
  # 이 값이 하중에 의한 **처짐**을 정한다: sag = m_obj*g/(assembly_mass*k_p)
  # → 30g에서 2.5mm, 350g에서 29mm. 이 처짐이 은닉 질량의 유일한 관측 단서다.
  k_p: float = 400.0
  # 손목 회전 강성. 병진보다 10배 강해야 하중 토크에 그리퍼가 통째로 돌아가지
  # 않는다(이전 시도에서 실측된 실패 모드)
  k_p_ang: float = 4000.0
  control_hz: float = 10.0
  physics_dt: float = 0.002           # s; 마찰 파지는 작은 스텝이 필요
  solver_iterations: int = 30         # 마찰 파지 안정화에 필요한 최소치

  seed: Optional[int] = None

  # ------------------------------------------------------------------ 유도값
  @property
  def gripper_outer_width(self) -> float:
    """개도 상한(`finger_opening_max`)일 때 그리퍼 전체 바깥 폭(mm).

    각 손가락이 개도 바깥으로 `finger_thickness`만큼 폭을 더한다. 소스 박스
    내폭의 랜덤 범위(`src_box_width_range`)는 이 값을 중심으로 걸쳐야
    하는데, 그래야 어떤 에피소드는 그리퍼가 박스에 들어가 깊은 파지가
    가능하고, 어떤 에피소드는 못 들어가 얕은 파지가 강제된다.
    """
    return self.finger_opening_max + 2.0 * self.finger_thickness

  @property
  def src_box_width_range(self) -> Tuple[float, float]:
    """소스 박스 내폭의 랜덤 범위 — `gripper_outer_width`의 0.85~1.25배.

    이 범위가 `gripper_outer_width`를 걸치지 않으면(항상 상회/하회) 모든
    에피소드가 항상 깊게 또는 항상 얕게만 잡혀 설계 의도(파지 깊이의
    에피소드 간 다양성)가 깨진다.
    """
    gow = self.gripper_outer_width
    return (0.85 * gow, 1.25 * gow)

  @property
  def tgt_box_width(self) -> float:
    """타겟 박스 내폭 — 그리퍼가 여유롭게 들어가도록 바깥 폭의 1.4배."""
    return 1.4 * self.gripper_outer_width

  @property
  def ee_force_max(self) -> float:
    """EE 힘 상한(mN) — 가반하중을 중력 하에서 지지하고 추가로
    `max_accel`으로 가속할 수 있는 힘.

    (assembly_mass + payload_mass) * (gravity + max_accel)
    """
    return (self.assembly_mass + self.payload_mass) * (
        self.gravity + self.max_accel)

  def max_hold_mass(self, mu: float) -> float:
    """주어진 마찰계수 `mu`에서 정적으로 들 수 있는 최대 질량(kg).

    2 * mu * grip_force / gravity. 이 값이 `object_mass_range`의 상한보다
    커야 태스크가 성립한다(가장 미끄러운 물체도 원리적으로 들 수 있어야
    함) — `mu` 최악값(범위 하한)에서도 성립하는지가 검증 포인트다.
    """
    return 2.0 * mu * self.grip_force / self.gravity

  def tip_torque_capacity(self, contact_len: float, mu: float) -> float:
    """패드 접촉 길이(`contact_len`, mm)에서의 회전 저항 토크 근사(mN·mm).

    mu * grip_force * contact_len. 얕은 파지(접촉 길이가 짧음)일수록
    회전 저항이 작아져 불안정해지는 것을 정량화하는 데 쓴다.
    """
    return mu * self.grip_force * contact_len
