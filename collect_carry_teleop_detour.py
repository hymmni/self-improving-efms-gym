r"""GraspCarry2D 비최적(non-optimal) 데모를 사람 개입 복구와 함께 수집한다.

디스플레이가 있는 PC에서 실행해야 한다(마우스/키보드 인터랙티브 창).

## 흐름
1. **자동 접근·파지**: `ScriptedCarryPolicy`가 평소처럼 블록에 접근해서 잡는다.
2. **자동 완전 무작위 방황**: 파지가 끝나면(운반 phase 진입) 박스 높이 같은
   안전장치 없이 작업공간 아무 데나(경유지)로 향한다 — 도착하면 바로 다음
   무작위 지점으로. 계속 이런 식이라 정말로 놓치는 일이 자주 생긴다(의도).
   지금 향하고 있는 경유지는 화면에 마젠타 별표로 항상 표시된다.
3. **사람 개입은 두 가지 경로로 들어간다**: (a) 자동 — 실제로 놓치면(2스텝
   연속 `is_held()=False`, 접촉 깜빡임 노이즈 아님) 자동으로 사람 모드로
   전환. (b) 수동 — 아무 때나 `h` 키를 누르면 즉시 사람 모드로/에서 전환된다
   (양방향 토글). 사람 모드에서는 마우스 커서 = 그리퍼 목표, 왼쪽 버튼
   누르는 동안 grip=닫힘. 다시 `h`를 누르면 방금 위치에서 이어서 자동
   방황으로(새 무작위 경유지 하나 새로 뽑아서) 넘어간다 — 원할 때
   자동/수동을 몇 번이고 오갈 수 있다.
4. 에피소드가 끝나면(성공/전도/타임아웃) 자동으로 다음 에피소드로 넘어간다.
   `--out`에 주기적으로 자동 저장하므로 중간에 창을 닫아도 그때까지 모은 성공
   에피소드는 남는다.

## 조작
- 마우스 이동: 그리퍼 목표 위치(사람 모드에서만 반영됨)
- 왼쪽 버튼 누르고 있기: 쥠(grip 닫힘) / 떼기: 열림(grip 열림)
- `h`: 자동 <-> 사람 모드 토글(양방향, 아무 때나)
- `r`: 지금 에피소드 버리고 새로 시작(실패로 안 셈, 저장도 안 함)
- `q`: 지금까지 모은 걸 저장하고 종료

## 실행
    python collect_carry_teleop_detour.py --out data/grasp_carry_demos_teleop_detour.pkl

`collect_carry_demos.py`와 **동일 스키마**로 저장한다(observation/action/
time_to_success/episode_id/is_success/meta) — `is_detour` 대신 사람이 조작한
스텝을 `is_human` 필드로 표시한다. 다른 데이터와 그대로 합칠 수 있다.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
# 인터랙티브 GUI 백엔드를 명시적으로 골라야 한다 — matplotlib.pyplot을 그냥
# import만 하면 환경에 따라(GUI 툴킷이 안 깔려 있는 등) 조용히 비인터랙티브
# 백엔드(Agg)로 떨어질 수 있고, 그러면 plt.ion()/plt.pause()가 에러 없이
# 그냥 아무 창도 안 띄운다. `matplotlib.use(name)` 자체는 해당 툴킷이 실제로
# import 가능한지 즉시 검증하지 않고 나중에(첫 Figure 생성 시) 실패할 수도
# 있어서(2026-08-24 실측 — try/except로 걸러지지 않고 그냥 조용히 넘어감),
# 후보마다 실제로 캔버스를 하나 만들어봐서 확실히 되는지까지 확인한다.
_INTERACTIVE_OK = False
for _backend in ('TkAgg', 'Qt5Agg', 'QtAgg', 'MacOSX'):
  try:
    matplotlib.use(_backend, force=True)
    import matplotlib.pyplot as plt
    _test_fig = plt.figure()
    plt.close(_test_fig)
    _INTERACTIVE_OK = True
    break
  except Exception:
    continue
if not _INTERACTIVE_OK:
  import matplotlib.pyplot as plt
  print(
      '!!! 인터랙티브 GUI 백엔드를 하나도 못 찾았다(TkAgg/Qt5Agg/QtAgg/MacOSX '
      '전부 실패) — 이대로 실행하면 창 없이 콘솔 로그만 찍히고 마우스 조작이 '
      '안 된다.\n'
      '    Linux: sudo apt install python3-tk  (또는 pip install PyQt5)\n'
      '    Windows/Mac: 보통 Tk가 기본 포함이니 pip install matplotlib --upgrade로도 '
      '해결 안 되면 pip install PyQt5\n'
      f'    (지금 남은 백엔드: {matplotlib.get_backend()})',
      file=sys.stderr)
  sys.exit(1)

from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import FRAME_FIELDS, GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy, _CEILING_Y
from record_carry import draw_env

_WAYPOINT_REACH_TOL = 5.0  # mm — 목표에 이 정도 가까우면 "도착"으로 보고 바로 다음 경유지
_WANDER_SPEED_MULT = 2.5   # 방황 중 PD 리드 배수(명목 속도의 몇 배) — 실측 근거는 아래 참고


class MouseState:
  """마우스 이벤트를 담는 그릇 — matplotlib 콜백은 클로저로 이 객체만 갱신한다."""

  def __init__(self):
    self.x = None
    self.y = None
    self.grip = 0.0  # 0.0=열림, 1.0=닫힘(왼쪽 버튼 누르는 동안)


def make_mouse_handlers(ms: MouseState):
  def on_move(event):
    if event.xdata is not None and event.ydata is not None:
      ms.x, ms.y = float(event.xdata), float(event.ydata)

  def on_press(event):
    if event.button == 1:
      ms.grip = 1.0

  def on_release(event):
    if event.button == 1:
      ms.grip = 0.0

  return on_move, on_press, on_release


def _sample_waypoint(env: GraspCarry2D, rng: np.random.Generator) -> tuple:
  """작업공간 아무 데나 — 박스 벽/rim 안전 높이 같은 건 신경 안 쓴다(의도).
  env.step의 하드스톱이 물리적으로 불가능한 목표는 어차피 막아주므로 폭주는
  안 하지만, 박스 벽 근처를 스칠 만큼 낮게도 뽑힐 수 있어 실제로 놓치는 일이
  생긴다 — 그게 이 함수의 목적이다."""
  cfg = env.cfg
  half = cfg.gripper_outer_width / 2.0
  tx = float(rng.uniform(half, cfg.world_width - half))
  ty = float(rng.uniform(_CEILING_Y, cfg.floor_y - 10.0))
  return tx, ty


def collect_one_episode(env: GraspCarry2D, cfg: CarryConfig, rng: np.random.Generator,
                        seed: int, ms: MouseState, fig, ax,
                        waypoint_min_steps: int, waypoint_max_steps: int,
                        pause_dt: float, discard: dict, toggle: dict):
  """에피소드 하나를 자동+사람 개입 섞어서 굴린다.

  `discard['flag']`가 True가 되면(‘r’ 키) 즉시 중단하고 (None, None, None,
  'discarded')를 반환한다. `toggle['flag']`가 True가 될 때마다(‘h’ 키)
  자동<->사람 모드가 즉시 뒤집힌다 — 놓쳤을 때의 자동 전환과 별개로 아무 때나
  직접 넘나들 수 있다.
  """
  obs, info = env.reset(seed=seed)
  policy = ScriptedCarryPolicy(config=cfg, allow_regrasp=True)
  policy.reset()

  obs_l, act_l, human_l = [obs], [], []
  # 2단계: auto_grasp(접근/파지, 여기까진 무작위로 만들 이유가 없다) ->
  # auto_wander(완전 무작위 경유지를 도착할 때마다/시간 초과 시마다 계속
  # 새로 뽑아 끝없이 방황 — 목적지로 스스로 마무리하는 단계가 없다, 사람이
  # `h`로 넘겨받아 직접 끝내거나 다시 자동으로 돌려보낸다).
  mode = 'auto_grasp'
  waypoint = None
  waypoint_ttl = 0
  was_held = False
  not_held_run = 0   # env._track_drop()과 동일 기준(2스텝 연속) — 접촉 깜빡임 오탐 방지

  terminated = truncated = False
  info_last = info
  for _ in range(cfg.max_steps):
    if discard['flag']:
      return None, None, None, 'discarded'

    if toggle['flag']:
      toggle['flag'] = False
      mode = 'human' if mode != 'human' else 'auto_wander'
      if mode == 'auto_wander':
        waypoint = None  # 자동으로 복귀하면 그 자리에서 새 경유지를 뽑는다
      print(f'  >> 수동 전환: {mode}')

    if mode == 'auto_grasp':
      a = policy(env)
      if policy.phase == 'carry':
        mode = 'auto_wander'
    elif mode == 'auto_wander':
      if waypoint is None:
        waypoint = _sample_waypoint(env, rng)
        waypoint_ttl = int(rng.integers(waypoint_min_steps, waypoint_max_steps + 1))
      # 목표를 그대로 즉시 명령하면(리드 제한 없이) PD 힘이 순간적으로 커져
      # 화면상 "순간이동"처럼 보일 만큼 빨라진다(사용자 지적, 2026-08-24) —
      # 그렇다고 정책의 명목 속도(policy.speed, 1배)로 완만히 다가가면 이번엔
      # 거의 절대 안 떨어뜨려서(0/15 실측) 방황의 의미가 없다. 절충으로
      # `_goto`(PD로 매 스텝 조금씩만 다가감, 여전히 물리적으로 연속적인
      # 움직임)를 쓰되 리드를 명목 속도의 배수로 키운다 — 실측(가짜 마우스로
      # 10회씩): 1.0x=0/10, 1.5x=2/10, 2.0x=3/10, 2.5x=4/10, 3.0x=5/10,
      # 4.0x=5/10(포화). 2.5배를 기본값으로 삼는다 — 대략 40%가 실제로
      # 놓치는 정도면 "위험하지만 매번은 아닌" 느낌에 맞는다고 판단.
      lead = policy.speed * _WANDER_SPEED_MULT
      vlead = policy._lift_lead * _WANDER_SPEED_MULT
      a = policy._goto(env, waypoint[0], waypoint[1], lead, 1.0, vlead=vlead)
      policy._step_i += 1  # __call__을 안 거치므로 내부 스텝 카운터를 직접 맞춘다
      waypoint_ttl -= 1
      ex, ey = float(env.gripper.pose[0]), float(env.gripper.pose[1])
      reached = (abs(ex - waypoint[0]) < _WAYPOINT_REACH_TOL
                and abs(ey - waypoint[1]) < _WAYPOINT_REACH_TOL)
      if reached or waypoint_ttl <= 0:
        waypoint = None   # 다음 스텝에 새로 뽑는다(도착이든 시간 초과든 바로 다음으로)
    else:  # 'human'
      tx = ms.x if ms.x is not None else float(env.gripper.pose[0])
      ty = ms.y if ms.y is not None else float(env.gripper.pose[1])
      a = np.array([tx, ty, 0.0, ms.grip], dtype=np.float32)

    is_human_step = (mode == 'human')
    act_l.append(np.asarray(a, dtype=np.float32))
    human_l.append(is_human_step)

    obs, _, terminated, truncated, info = env.step(a)
    info_last = info

    held_now = env.is_held()
    if held_now:
      not_held_run = 0
    elif was_held or not_held_run > 0:
      # was_held(방금까지 잡고 있었음) 또는 이미 연속 카운트 중이면 이어서 센다
      # — env.py _track_drop()과 동일하게 접촉 깜빡임(1스텝) 오탐을 거른다.
      not_held_run += 1
    if mode == 'auto_wander' and not_held_run >= 2:
      mode = 'human'
      print('  !! 놓침 -- 마우스로 이어받으세요 (누르고 있으면 쥠, h로 자동 복귀) !!')
    was_held = held_now

    draw_env(ax, env, action=a)
    tag = {'auto_grasp': '자동(접근/파지)', 'auto_wander': '자동(무작위 방황)',
          'human': '사람 개입 중'}[mode]
    ax.set_title(ax.get_title() + f'   [{tag}]', fontsize=9)
    if mode == 'auto_wander' and waypoint is not None:
      ax.plot(waypoint[0], waypoint[1], marker='*', markersize=16,
              color='magenta', markeredgecolor='k', zorder=10)
    fig.canvas.draw_idle()
    plt.pause(pause_dt)

    if terminated or truncated:
      break
    obs_l.append(obs)

  outcome = info_last['outcome']
  L = len(act_l)
  assert len(obs_l) == L, '관측/액션 스텝 수 불일치'
  return obs_l, act_l, human_l, outcome


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default='data/grasp_carry_demos_teleop_detour.pkl')
  ap.add_argument('--seed0', type=int, default=800000)
  ap.add_argument('--explore-seed', type=int, default=0)
  ap.add_argument('--waypoint-min-steps', type=int, default=10)
  ap.add_argument('--waypoint-max-steps', type=int, default=25)
  ap.add_argument('--pause-dt', type=float, default=0.03,
                  help='프레임 사이 대기(초) — 너무 빠르면 마우스로 못 쫓아감')
  ap.add_argument('--autosave-every', type=int, default=5,
                  help='이 개수만큼 성공 에피소드가 쌓일 때마다 --out에 저장')
  args = ap.parse_args()

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  rng = np.random.default_rng(args.explore_seed)

  print(f'[matplotlib 백엔드] {matplotlib.get_backend()} '
        f'(창이 안 뜨면 이 백엔드용 GUI 툴킷이 안 깔려있는 것 — '
        f'예: pip install pyqt5, 또는 Tk가 있다면 시스템 패키지로 python3-tk 설치)')
  plt.ion()
  fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
  try:
    fig.canvas.manager.set_window_title('GraspCarry2D 비최적 데모 수집')
  except Exception:
    pass
  plt.show(block=False)   # ion()+pause()만으로는 일부 환경에서 창이 안 떠서 명시적으로 띄운다
  ms = MouseState()
  on_move, on_press, on_release = make_mouse_handlers(ms)
  fig.canvas.mpl_connect('motion_notify_event', on_move)
  fig.canvas.mpl_connect('button_press_event', on_press)
  fig.canvas.mpl_connect('button_release_event', on_release)

  discard = {'flag': False}
  quit_flag = {'flag': False}
  toggle = {'flag': False}

  def on_key(event):
    if event.key == 'r':
      discard['flag'] = True
    elif event.key == 'q':
      quit_flag['flag'] = True
    elif event.key == 'h':
      toggle['flag'] = True

  fig.canvas.mpl_connect('key_press_event', on_key)

  obs_all, act_all, ttg_all, eid_all, succ_all, human_all = [], [], [], [], [], []
  outcomes = {}
  kept = 0
  seed = args.seed0
  last_saved = 0

  def save(final=False):
    data = {
        'observation': {'frame': (np.concatenate(obs_all) if obs_all
                                  else np.zeros((0, env.obs_dim), dtype=np.float32))},
        'action': (np.concatenate(act_all) if act_all else np.zeros((0, 4), dtype=np.float32)),
        'time_to_success': (np.concatenate(ttg_all) if ttg_all else np.zeros((0,), dtype=np.float32)),
        'episode_id': (np.concatenate(eid_all) if eid_all else np.zeros((0,), dtype=np.int32)),
        'is_success': (np.concatenate(succ_all) if succ_all else np.zeros((0,), dtype=bool)),
        'is_human': (np.concatenate(human_all) if human_all else np.zeros((0,), dtype=bool)),
        'meta': {
            'source': 'collect_carry_teleop_detour.py: auto random-waypoint carry '
                      '+ human mouse recovery on drop',
            'frame_fields': list(FRAME_FIELDS),
            'obs_history': cfg.obs_history,
            'action_fields': ('x', 'y', 'theta', 'grip'),
            'keep_failures': False,
            'failure_bin': cfg.max_steps,
            'requested_episodes': seed - args.seed0,
            'kept_episodes': kept,
            'outcomes': outcomes,
            'seed0': args.seed0,
            'control_hz': cfg.control_hz,
            'max_steps': cfg.max_steps,
            'waypoint_min_steps': args.waypoint_min_steps,
            'waypoint_max_steps': args.waypoint_max_steps,
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
      pickle.dump(data, fp)
    tag = '(최종)' if final else '(중간저장)'
    print(f'저장{tag}: 성공 {kept}개, {len(data["time_to_success"])}스텝 -> {args.out}')

  print(__doc__)
  print(f'시작 (seed0={args.seed0}). 창에서 h=자동/사람 전환, r=버리고 재시작, q=저장 후 종료.')

  try:
    while not quit_flag['flag']:
      discard['flag'] = False
      toggle['flag'] = False
      print(f'--- 에피소드 seed={seed} ---')
      obs_l, act_l, human_l, outcome = collect_one_episode(
          env, cfg, rng, seed, ms, fig, ax,
          args.waypoint_min_steps, args.waypoint_max_steps, args.pause_dt, discard, toggle)
      seed += 1

      if outcome == 'discarded':
        print('  버려짐 (재시작)')
        continue

      outcomes[outcome] = outcomes.get(outcome, 0) + 1
      is_success = outcome == 'success'
      print(f'  결과: {outcome}  (스텝 {len(act_l)})')
      if not is_success:
        continue

      L = len(act_l)
      ttg = (L - 1 - np.arange(L)).astype(np.float32)
      obs_all.append(np.array(obs_l, dtype=np.float32))
      act_all.append(np.array(act_l, dtype=np.float32))
      ttg_all.append(ttg)
      eid_all.append(np.full(L, kept, dtype=np.int32))
      succ_all.append(np.full(L, True, dtype=bool))
      human_all.append(np.array(human_l, dtype=bool))
      kept += 1

      if kept - last_saved >= args.autosave_every:
        save()
        last_saved = kept
  except KeyboardInterrupt:
    print('\n중단됨 (Ctrl+C) — 지금까지 모은 것 저장.')

  save(final=True)
  plt.close(fig)


if __name__ == '__main__':
  main()
