r"""GraspCarry2D 비최적(non-optimal) 데모를 사람 개입 복구와 함께 수집한다.

디스플레이가 있는 PC에서 실행해야 한다(마우스/키보드 인터랙티브 창).

## 흐름
1. **자동 접근·파지**: `ScriptedCarryPolicy`가 평소처럼 블록에 접근해서 잡는다
   (여기까진 무작위로 만들 이유가 없다 — 실패하면 그냥 에피소드가 못 시작됨).
2. **자동 무작위 웨이포인트 운반**: 파지가 끝나면(운반 phase 진입) 목적지 대신
   매번 새로 뽑은 무작위 지점으로 향한다 — `ScriptedCarryPolicy._goto`와 같은
   완만한(안전 속도 제한) 이동이라 정상적으로는 안 떨어뜨리지만, 계속 방향을
   바꾸다 보면 실제로 놓치는 경우가 생긴다.
3. **사람 개입**: 놓치는 순간(`is_held()`가 True->False로 바뀌는 순간, 붙잡고
   있다가 놓친 것 — 원래 안 잡고 있던 것과 구분) 자동조종이 멈추고 마우스
   조작으로 넘어간다. 마우스 커서 위치 = 그리퍼 목표(x, y), 왼쪽 버튼을 누르고
   있으면 grip=닫힘(쥠), 떼면 열림 — 사람이 다시 집어서 타겟 박스까지 옮기면
   된다. 사람이 개입한 뒤에도 성공하면 그 에피소드는 그대로 저장된다.
4. 에피소드가 끝나면(성공/전도/타임아웃) 자동으로 다음 에피소드로 넘어간다.
   `--out`에 주기적으로 자동 저장하므로 중간에 창을 닫아도 그때까지 모은 성공
   에피소드는 남는다.

## 조작
- 마우스 이동: 그리퍼 목표 위치
- 왼쪽 버튼 누르고 있기: 쥠(grip 닫힘) / 떼기: 열림(grip 열림)
  (자동 구간에서는 무시된다 — 사람 개입 구간에서만 마우스가 액션을 낸다)
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

import numpy as np
import matplotlib.pyplot as plt

from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import FRAME_FIELDS, GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy
from record_carry import draw_env


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


def collect_one_episode(env: GraspCarry2D, cfg: CarryConfig, rng: np.random.Generator,
                        seed: int, ms: MouseState, fig, ax,
                        waypoint_min_steps: int, waypoint_max_steps: int,
                        pause_dt: float, discard: dict):
  """에피소드 하나를 자동+사람 개입 섞어서 굴린다.

  `discard['flag']`가 True가 되면(‘r’ 키) 즉시 중단하고 (None, None, None,
  'discarded')를 반환한다.
  """
  obs, info = env.reset(seed=seed)
  policy = ScriptedCarryPolicy(config=cfg, allow_regrasp=True)
  policy.reset()

  obs_l, act_l, human_l = [obs], [], []
  # 3단계: auto_grasp(접근/파지) -> auto_detour(무작위 경유지 n_detours개,
  # 여기서만 방황) -> auto_finish(정책에 제어권 그대로 돌려줘서 원래 목적지에
  # 정상적으로 내려놓기까지 마무리) — 어느 auto 단계에서든 실제로 놓치면
  # 그 순간 human으로 전환된다. auto_carry가 목적지로 안 가고 무한히 새
  # 경유지만 뽑으면 영원히 타임아웃만 나므로(2026-08-24 스모크 테스트로 발견),
  # 반드시 auto_finish로 빠지는 경로가 있어야 한다.
  mode = 'auto_grasp'
  n_detours = int(rng.integers(1, 4))     # 1~3개 경유지
  detours_done = 0
  waypoint = None
  waypoint_ttl = 0
  was_held = False
  not_held_run = 0   # env._track_drop()과 동일 기준(2스텝 연속) — 접촉 깜빡임 오탐 방지
  half = cfg.gripper_outer_width / 2.0

  terminated = truncated = False
  info_last = info
  for _ in range(cfg.max_steps):
    if discard['flag']:
      return None, None, None, 'discarded'

    if mode == 'auto_grasp':
      a = policy(env)
      if policy.phase == 'carry':
        mode = 'auto_detour'
    elif mode == 'auto_detour':
      if waypoint is None or waypoint_ttl <= 0:
        if detours_done >= n_detours:
          mode = 'auto_finish'
          a = policy(env)
        else:
          tx = float(rng.uniform(half, cfg.world_width - half))
          ty = float(policy._travel_y_hold(env))
          waypoint = (tx, ty)
          waypoint_ttl = int(rng.integers(waypoint_min_steps, waypoint_max_steps + 1))
          detours_done += 1
          a = policy._goto(env, waypoint[0], waypoint[1], policy._safe_speed(env), 1.0)
          waypoint_ttl -= 1
          policy._step_i += 1  # __call__을 안 거치므로 정책의 내부 스텝 카운터를
      else:                     # 직접 맞춰줘야 한다 — 안 그러면 마무리 단계(auto_finish)
        a = policy._goto(env, waypoint[0], waypoint[1], policy._safe_speed(env), 1.0)
        waypoint_ttl -= 1        # 에서 _safe_speed의 "남은 예산" 계산이 실제보다 넉넉하다고
        policy._step_i += 1      # 착각해 너무 느리게 움직이다 타임아웃난다(실측: 30/30 실패).
    elif mode == 'auto_finish':
      a = policy(env)
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
    if mode in ('auto_detour', 'auto_finish') and not_held_run >= 2:
      mode = 'human'
      print('  !! 놓침 -- 마우스로 이어받으세요 (누르고 있으면 쥠) !!')
    was_held = held_now

    draw_env(ax, env, action=a)
    tag = {'auto_grasp': '자동(접근/파지)', 'auto_detour': '자동(무작위 경유)',
          'auto_finish': '자동(마무리)', 'human': '사람 개입 중'}[mode]
    ax.set_title(ax.get_title() + f'   [{tag}]', fontsize=9)
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

  plt.ion()
  fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
  ms = MouseState()
  on_move, on_press, on_release = make_mouse_handlers(ms)
  fig.canvas.mpl_connect('motion_notify_event', on_move)
  fig.canvas.mpl_connect('button_press_event', on_press)
  fig.canvas.mpl_connect('button_release_event', on_release)

  discard = {'flag': False}
  quit_flag = {'flag': False}

  def on_key(event):
    if event.key == 'r':
      discard['flag'] = True
    elif event.key == 'q':
      quit_flag['flag'] = True

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
  print(f'시작 (seed0={args.seed0}). 창에서 r=버리고 재시작, q=저장 후 종료.')

  try:
    while not quit_flag['flag']:
      discard['flag'] = False
      print(f'--- 에피소드 seed={seed} ---')
      obs_l, act_l, human_l, outcome = collect_one_episode(
          env, cfg, rng, seed, ms, fig, ax,
          args.waypoint_min_steps, args.waypoint_max_steps, args.pause_dt, discard)
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
