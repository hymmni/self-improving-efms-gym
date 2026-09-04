r"""GraspCarry2D 데모를 처음부터 끝까지 순수하게 마우스로 조작해 수집한다.

디스플레이가 있는 PC에서 실행해야 한다(마우스/키보드 인터랙티브 창).

`collect_carry_teleop_detour.py`(자동 접근/파지 + 자동 방황 + 실패 시 사람 개입)와
달리, 이 스크립트는 스크립트 정책을 아예 안 쓴다 — 접근·파지·운반 전부 매 스텝
사람이 마우스로 직접 목표를 준다. 매 에피소드 `h`를 눌러 모드를 전환할 필요가
없다(2026-08-30 사용자 피드백 — 매번 누르는 게 불편했음).

## 흐름
1. 에피소드가 리셋되면 화면에 초기 상태만 뜨고 **대기**한다(타임아웃 시계도
   안 간다) — 마우스를 원하는 위치에 갖다 놓고 준비되면 스페이스바를 눌러야
   실제로 스텝이 시작된다.
2. 스페이스바를 누르면 그 순간부터 매 스텝 마우스 위치 = 그리퍼 목표,
   휠 스크롤로 누적한 grip 값(0~1, 힘의 세기)으로 물리 스텝이 진행된다.
3. 마우스를 움직이면 목표가 그 즉시 그 위치로 점프하는 게 아니라(그러면 PD
   힘이 순간적으로 커져 불안정해짐 — 2026-08-30 사용자 리포트), 한 스텝에
   `--lead` mm만큼만 다가가도록 감쇠한다(자동 방황 모드의 `_goto`와 같은
   원리). 그리퍼가 마우스를 살짝 늦게, 하지만 부드럽게 따라온다.
4. **블록이 넘어져도**(outcome=tipped) 에피소드를 끝내지 않는다 —
   `env.py`는 넘어짐을 영구 실패로 판정하지만(스크립트 정책은 스스로
   못 일으키니 그게 맞는 판정), 사람은 그 자리에서 블록을 바로 세워 복구를
   시도할 수 있다. 그냥 계속 조작을 이어가면 되고(화면 타이틀에 `[TIPPED]`
   표시만 뜬다), 정 안 되겠으면 `r`로 버리고 새로 시작하면 된다(2026-08-30
   사용자 피드백 — "넘어짐 = 실패 판정" 자체를 안 하는 게 더 간단하다는
   지적으로, 재시도 여부를 매번 물어보던 이전 방식을 걷어냈다).
5. 에피소드가 성공/타임아웃으로 끝나면 자동으로 다음 에피소드로 넘어가고,
   다시 대기 상태(1번)로 돌아간다.
6. `--out`에 주기적으로 자동 저장하므로 중간에 창을 닫아도 그때까지 모은 성공
   에피소드는 남는다.

## 조작
- 마우스 이동: 그리퍼 목표 위치(스페이스바를 눌러 시작한 뒤에만 반영됨)
- 마우스 휠 스크롤: grip 조절 — 아래로 스크롤하면 `--grip-step`만큼 닫히는
  쪽으로, 위로 스크롤하면 열리는 쪽으로 누적된다(0~1 클립, 목표 개도로
  수렴하는 PD 위치제어 — `gripper.py::apply_grip` 참고). 왼쪽 버튼을 누른
  채 드래그하면 클릭이 위치 이동과 간섭해서 조작이 힘들다는 2026-08-30
  피드백으로 휠로 분리했다. 세밀하게 조금씩 눌러보며 좁은 박스 안 블록을
  조심스레 잡을 수 있다.
- 스페이스바: 대기 중인 에피소드를 시작
- `r`: 지금 에피소드(대기 중이든 조작 중이든, 넘어진 채든)를 버리고 새로
  시작(저장 안 됨)
- `q`: 지금까지 모은 걸 저장하고 종료

## 실행
    python -m grasp_carry.scripts.collect.collect_carry_teleop_demos \
        --out data/grasp_carry_demos_teleop.pkl

`collect_carry_demos.py`와 **동일 스키마**로 저장한다(observation/action/
time_to_success/episode_id/is_success/meta) — 사람이 조작한 스텝을 표시하는
`is_human` 필드는 전 스텝 True다(전부 수동이므로). 다른 데이터와 그대로
합칠 수 있다.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
# 인터랙티브 GUI 백엔드를 명시적으로 골라야 한다 — collect_carry_teleop_detour.py와
# 동일한 이유(2026-08-24 실측)로 실제 캔버스를 만들어봐서 확실히 되는지까지 확인한다.
_INTERACTIVE_OK = False
_INTERACTIVE_BACKEND = None
for _backend in ('TkAgg', 'Qt5Agg', 'QtAgg', 'MacOSX'):
  try:
    matplotlib.use(_backend, force=True)
    import matplotlib.pyplot as plt
    _test_fig = plt.figure()
    plt.close(_test_fig)
    _INTERACTIVE_OK = True
    _INTERACTIVE_BACKEND = _backend
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

from grasp_carry.config import CarryConfig
from grasp_carry.env import FRAME_FIELDS, GraspCarry2D
from grasp_carry.scripts.record.record_carry import draw_env

# record_carry.py는 자기 자신은 헤드리스 mp4 렌더링용이라 모듈 최상단에서
# matplotlib.use('Agg')를 무조건 호출한다 — 그 import(draw_env를 쓰려고 필요)가
# 위에서 애써 찾은 인터랙티브 백엔드를 조용히 Agg로 덮어써버린다(2026-08-30
# collect_carry_teleop_detour.py에서 실측한 것과 동일한 문제). 확인된 인터랙티브
# 백엔드를 다시 강제한다.
matplotlib.use(_INTERACTIVE_BACKEND, force=True)


class MouseState:
  """마우스 이벤트를 담는 그릇 — matplotlib 콜백은 클로저로 이 객체만 갱신한다."""

  def __init__(self):
    self.x = None
    self.y = None
    self.grip = 0.0  # 0.0=완전히 열림, 1.0=완전히 닫힘 — 휠 스크롤로 조금씩 누적
    self._last_scroll_t = 0.0


def make_mouse_handlers(ms: MouseState, grip_step: float, scroll_min_interval: float):
  def on_move(event):
    if event.xdata is not None and event.ydata is not None:
      ms.x, ms.y = float(event.xdata), float(event.ydata)

  def on_scroll(event):
    # 왼쪽 클릭-홀드는 드래그가 위치 이동과 간섭해서 뺐다 — 휠은 위치와
    # 완전히 독립된 입력이라 grip 전용으로 쓰기 좋다. gripper.py::apply_grip이
    # 힘의 세기를 grip 값으로 보간하도록 바뀌었으므로(2026-08-30), 한 번에
    # 완전히 열림/닫힘으로 점프하지 않고 `grip_step`만큼씩 누적한다 — 좁은
    # 박스 안 블록을 조심스레 눌러 잡을 수 있게.
    #
    # 트랙패드/일부 마우스는 스크롤 "한 칸"에도 이벤트가 여러 개(많으면
    # 10개 이상) 한꺼번에 들어온다 — 이벤트 하나당 grip_step씩 더하면 한
    # 번의 스크롤 제스처가 grip을 훨씬 크게 움직여버린다(2026-08-30 사용자
    # 리포트). `scroll_min_interval` 안에 들어온 이벤트는 무시해서 실제
    # 증가 속도를 시간 기준으로 제한한다(이벤트 개수와 무관하게 일정하다).
    now = time.monotonic()
    if now - ms._last_scroll_t < scroll_min_interval:
      return
    ms._last_scroll_t = now
    if event.button == 'down':
      ms.grip = min(1.0, ms.grip + grip_step)
    elif event.button == 'up':
      ms.grip = max(0.0, ms.grip - grip_step)

  return on_move, on_scroll


def _lead_limit(cur: float, target: float, lead: float) -> float:
  """`cur`에서 `target` 방향으로 최대 `lead`만큼만 다가간 값을 낸다.

  마우스를 그대로(리드 제한 없이) 명령하면 PD 힘이 순간적으로 커져 불안정해진다
  (2026-08-30 사용자 리포트) — `policy._goto`와 같은 원리로 목표를 완만하게
  뒤쫓게 한다."""
  d = target - cur
  if d > lead:
    d = lead
  elif d < -lead:
    d = -lead
  return cur + d


def _descend_hud(env: GraspCarry2D) -> str:
  """지금 x에서 왜 더 못 내려가는지 한눈에 보이게 하는 HUD 텍스트.

  '손가락이 안 닿았는데 왜 멈추냐'는 리포트(2026-08-30)의 원인이 그리퍼
  손가락이 아니라 **더 넓은 몸통(레일 플레이트)이 박스 개구부에 안 들어가서**
  인 경우가 많아, 눈에 보이는 손가락만으로는 이유를 알기 어렵다 — 그래서
  현재 하강 한계와 그 이유(바닥/rim)를 매 프레임 표시한다."""
  x = float(env.gripper.base.position.x)
  limit = env.max_descend_y(x)
  base_y = float(env.gripper.base.position.y)
  at_limit = base_y >= limit - 1.0
  reason = 'floor'
  for box in (env.src_box, env.tgt_box):
    if box.left_outer <= x <= box.right_outer and limit < env.cfg.floor_y - env.cfg.finger_length - 1.0:
      reason = 'rim (gripper body too wide for this box opening)'
      break
  tip_y = base_y + env.cfg.finger_length  # 손가락 패드가 실제로 닿을 수 있는 최하단
  block_top_y = float(env.block_body.position.y) - env.block_h / 2.0
  reach = 'reachable' if block_top_y <= tip_y else f'{block_top_y - tip_y:.0f}mm short'
  return (f'descend limit y={limit:.0f}{" [AT LIMIT: " + reason + "]" if at_limit else ""}'
         f'  |  finger tip y={tip_y:.0f}  block top y={block_top_y:.0f} ({reach})')


# record_carry.draw_env()가 내부에서 강제하는 view_limits()는 영상 프레이밍용
# 고정 크롭(위쪽 130mm까지만)이라, 그리퍼가 조금만 높이 들려도(불안정한 파지를
# 든 채 회복하려 할 때 특히) 화면/마우스 추적 범위 밖으로 나가버린다(2026-08-30
# 사용자 리포트 — 그리퍼가 실제로 갈 수 있는 위쪽 한계는 _CEILING_Y=20인데
# 화면은 130까지만 보여줌). teleop은 화면이 곧 마우스 조작 가능 범위이므로,
# draw_env가 그려놓은 뒤 그리퍼가 실제로 도달 가능한 전체 범위로 다시 넓힌다.
_TELEOP_TOP_Y = 5.0        # _CEILING_Y(20)보다 살짝 위 여유
_TELEOP_BOTTOM_MARGIN = 30.0  # record_carry._FLOOR_MARGIN과 동일


def _draw(ax, env, action=None):
  draw_env(ax, env, action=action)
  ax.set_xlim(0.0, env.cfg.world_width)
  ax.set_ylim(env.cfg.floor_y + _TELEOP_BOTTOM_MARGIN, _TELEOP_TOP_Y)


def wait_until_ready(env, fig, ax, pause_dt: float, discard: dict, ready: dict,
                     quit_flag: dict):
  """리셋 직후 초기 프레임만 보여주며 스페이스바를 기다린다(타임아웃 시계는
  안 감 — env.step()을 아예 안 부른다). `r`이면 'discarded', `q`면 'quit',
  스페이스바면 None을 반환한다."""
  while True:
    if quit_flag['flag']:
      return 'quit'
    if discard['flag']:
      return 'discarded'
    if ready['flag']:
      ready['flag'] = False
      return None
    _draw(ax, env)
    ax.set_title(ax.get_title() + '   [READY -- press space to start]', fontsize=9)
    fig.canvas.draw_idle()
    plt.pause(pause_dt)


def collect_one_episode(env: GraspCarry2D, cfg: CarryConfig, seed: int,
                        ms: MouseState, fig, ax, lead: float, pause_dt: float,
                        discard: dict, quit_flag: dict, ready: dict):
  """에피소드 하나를 순수 사람 조작으로 굴린다.

  `discard['flag']`가 True가 되면(‘r’ 키) 즉시 중단하고 (None, None, None,
  'discarded')를 반환한다. `q`로 종료하면 'quit'을 반환한다."""
  obs, info = env.reset(seed=seed)

  outcome = wait_until_ready(env, fig, ax, pause_dt, discard, ready, quit_flag)
  if outcome is not None:
    return None, None, None, outcome

  obs_l, act_l, human_l = [obs], [], []
  # 그리퍼의 현재 실제 위치에서 리드 제한 추적을 시작한다 — 마우스 커서가
  # 이미 멀리 있어도 첫 스텝부터 순간이동하지 않게.
  cur_tx, cur_ty = float(env.gripper.pose[0]), float(env.gripper.pose[1])

  terminated = truncated = False
  info_last = info
  for _ in range(cfg.max_steps):
    if discard['flag']:
      return None, None, None, 'discarded'
    if quit_flag['flag']:
      return None, None, None, 'quit'

    tx_raw = ms.x if ms.x is not None else cur_tx
    ty_raw = ms.y if ms.y is not None else cur_ty
    cur_tx = _lead_limit(cur_tx, tx_raw, lead)
    cur_ty = _lead_limit(cur_ty, ty_raw, lead)
    a = np.array([cur_tx, cur_ty, 0.0, ms.grip], dtype=np.float32)

    act_l.append(a)
    human_l.append(True)

    obs, _, terminated, truncated, info = env.step(a)
    info_last = info

    _draw(ax, env, action=a)
    tag = f"   [{info['outcome'].upper()}]" if info['outcome'] == 'tipped' else ''
    ax.set_title(ax.get_title() + tag + '   ' + _descend_hud(env), fontsize=8)
    fig.canvas.draw_idle()
    plt.pause(pause_dt)

    # 'tipped'는 여기서 끝내지 않는다 — env.py는 넘어짐을 영구 실패로 판정하지만
    # (스크립트 정책은 못 일으키니 그게 맞음), 사람은 그 자리에서 블록을 바로
    # 세워 복구를 시도할 수 있다. 그냥 계속 스텝을 밟게 두고 `r`(버리기)을
    # 탈출구로 남겨두면, 복구 성공 시 "실패 순간부터 사람이 이어받아 완성한"
    # 궤적 하나가 그대로 데이터가 된다 — 원래 요청(2026-08-30)이 이거였다.
    if (terminated or truncated) and info['outcome'] != 'tipped':
      break
    obs_l.append(obs)

  outcome = info_last['outcome']
  L = len(act_l)
  assert len(obs_l) == L, '관측/액션 스텝 수 불일치'
  return obs_l, act_l, human_l, outcome


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default='data/grasp_carry_demos_teleop.pkl')
  ap.add_argument('--seed0', type=int, default=800000)
  ap.add_argument('--lead', type=float, default=15.0,
                  help=('한 스텝에 그리퍼 목표가 마우스 커서 쪽으로 다가갈 수 '
                        '있는 최대 거리(mm). 너무 크면 순간이동처럼 불안정해지고 '
                        '(리드 제한이 없던 것과 같아짐), 너무 작으면 마우스를 '
                        '따라오는 게 굼떠진다. 기본값(15mm)은 정책 명목 속도'
                        '(12.5mm/스텝)보다 살짝 크게 잡은 값 — 필요하면 조절할 것.'))
  ap.add_argument('--max-steps', type=int, default=100000,
                  help=('에피소드 하나가 이 스텝을 넘기면 타임아웃 처리된다. '
                        '기본값(100000=10Hz에서 약 2.7시간)은 사실상 무제한 — '
                        '사람 조작은 스크립트 정책보다 훨씬 느리므로 기본 200스텝'
                        '(20초)으로는 하나도 못 모은다는 2026-08-30 리포트 반영.'))
  ap.add_argument('--pause-dt', type=float, default=None,
                  help=('프레임 사이 대기(초). 기본값은 --control-hz의 역수(실시간 '
                        '페이싱) — 너무 빠르면 마우스로 못 쫓아간다.'))
  ap.add_argument('--autosave-every', type=int, default=5,
                  help='이 개수만큼 성공 에피소드가 쌓일 때마다 --out에 저장')
  ap.add_argument('--grip-step', type=float, default=0.05,
                  help=('휠 한 노치당 grip이 누적되는 양(0~1 클립). 작을수록 '
                        '세밀하게 조절되지만 완전히 닫는 데 스크롤이 더 많이 '
                        '필요하다(기본 0.05 = 노치 20번이면 완전히 닫힘).'))
  ap.add_argument('--scroll-min-interval', type=float, default=0.05,
                  help=('스크롤 이벤트를 반영하는 최소 간격(초). 트랙패드 등에서 '
                        '스크롤 한 번에 이벤트가 여러 개 몰려 들어와도 이 간격 '
                        '안에 들어온 건 무시해서 grip이 한 번에 너무 많이 '
                        '움직이지 않게 한다(기본 0.05s = 최대 초당 20단계).'))
  args = ap.parse_args()

  cfg = CarryConfig(max_steps=args.max_steps)
  env = GraspCarry2D(cfg)
  pause_dt = args.pause_dt if args.pause_dt is not None else 1.0 / cfg.control_hz

  print(f'[matplotlib 백엔드] {matplotlib.get_backend()} '
        f'(창이 안 뜨면 이 백엔드용 GUI 툴킷이 안 깔려있는 것 — '
        f'예: pip install pyqt5, 또는 Tk가 있다면 시스템 패키지로 python3-tk 설치)')
  plt.ion()
  # 세로로 넓힌 뷰(_TELEOP_TOP_Y..floor+margin, record_carry의 기본 크롭보다
  # 약 1.34배 큼)에 맞춰 창도 세로로 키운다 — 안 그러면 그리퍼/블록이
  # 작아져서 마우스로 정밀 조작하기 더 어려워진다.
  fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
  try:
    fig.canvas.manager.set_window_title('GraspCarry2D pure teleop demo collection')
  except Exception:
    pass
  plt.show(block=False)
  ms = MouseState()
  on_move, on_scroll = make_mouse_handlers(ms, args.grip_step, args.scroll_min_interval)
  fig.canvas.mpl_connect('motion_notify_event', on_move)
  fig.canvas.mpl_connect('scroll_event', on_scroll)

  discard = {'flag': False}
  quit_flag = {'flag': False}
  ready = {'flag': False}

  def on_key(event):
    if event.key == 'r':
      discard['flag'] = True
    elif event.key == 'q':
      quit_flag['flag'] = True
    elif event.key == ' ':
      ready['flag'] = True

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
            'source': 'collect_carry_teleop_demos.py: pure mouse teleop, no scripted policy',
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
            'lead': args.lead,
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
      pickle.dump(data, fp)
    tag = '(최종)' if final else '(중간저장)'
    print(f'저장{tag}: 성공 {kept}개, {len(data["time_to_success"])}스텝 -> {args.out}')

  print(f'시작 (seed0={args.seed0}). 스페이스바=에피소드 시작, r=버리고 재시작, q=저장 후 종료.')

  try:
    while not quit_flag['flag']:
      discard['flag'] = False
      print(f'--- 에피소드 seed={seed} (스페이스바로 시작) ---')
      obs_l, act_l, human_l, outcome = collect_one_episode(
          env, cfg, seed, ms, fig, ax, args.lead, pause_dt, discard, quit_flag, ready)

      if outcome == 'quit':
        break
      if outcome == 'discarded':
        print('  버려짐 (재시작, 새 seed)')
        seed += 1
        continue

      outcomes[outcome] = outcomes.get(outcome, 0) + 1
      print(f'  결과: {outcome}  (스텝 {len(act_l)})')

      seed += 1
      is_success = outcome == 'success'
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
