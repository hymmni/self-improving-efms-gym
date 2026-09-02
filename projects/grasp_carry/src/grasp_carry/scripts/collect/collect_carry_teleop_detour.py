r"""GraspCarry2D **학습된 DP(diffusion) 정책** 롤아웃을 사람 개입 복구와 함께 수집한다.

디스플레이가 있는 PC에서 실행해야 한다(마우스/키보드 인터랙티브 창).

## 흐름
1. **자동**: `--diff-ckpt`로 지정한 학습된 diffusion BC 정책이 접근·파지·
   운반·놓기까지 전체 태스크를 수행한다(`collect_carry_bc_rollouts.py`와
   동일한 receding-horizon 청크 실행 — 청크를 한 번에 예측해 그중
   `exec_horizon`개만 실행하고 새 관측으로 재추론). 2026-08-31: 이전엔
   `ScriptedCarryPolicy`(규칙 기반 베이스라인)를 굴렸으나, "정책 롤아웃 중
   실패하면 인간 개입"의 진짜 의도는 **학습된 정책**의 실제 실패였다 — 스크립트
   정책은 이미 100% 가까운 성공률로 실패가 인위적이지 않으면 거의 안 나온다.
   이제 여기서 나는 실패는 DP가 데모에서 다 배우지 못한 부분에서 실제로 발생한
   것이다(SI-EFM Stage-2와 동일한 발상).
2. **사람 개입은 수동으로만** 들어간다: 아무 때나 `h` 키를 누르면 즉시 사람
   모드로/에서 전환된다(양방향 토글, 자동 전환 없음). 사람 모드에서는 마우스
   커서 = 그리퍼 목표, 왼쪽 버튼 누르는 동안 grip=닫힘. 다시 `h`를 누르면
   방금 위치에서 이어서 DP가 재개된다(그 시점 관측으로 새 청크를 다시
   예측한다) — 원할 때 자동/수동을 몇 번이고 오갈 수 있다.
3. 에피소드가 끝나면(성공/전도/타임아웃) 자동으로 다음 에피소드로 넘어간다.
   `--out`에 주기적으로 자동 저장하므로 중간에 창을 닫아도 그때까지 모은 성공
   에피소드는 남는다.

## 조작
- 마우스 이동: 그리퍼 목표 위치(사람 모드에서만 반영됨)
- 왼쪽 버튼 누르고 있기: 쥠(grip 닫힘) / 떼기: 열림(grip 열림)
- `h`: 자동 <-> 사람 모드 토글(양방향, 아무 때나)
- `r`: 지금 에피소드 버리고 새로 시작(실패로 안 셈, 저장도 안 함)
- `q`: 지금까지 모은 걸 저장하고 종료

## 실행
    python -m grasp_carry.scripts.collect.collect_carry_teleop_detour \
        --diff-ckpt checkpoints/grasp_carry_diff100_v5/predictor.pkl \
        --out data/grasp_carry_demos_teleop_detour.pkl

`collect_carry_demos.py`와 **동일 스키마**로 저장한다(observation/action/
time_to_success/episode_id/is_success/meta) — `is_detour` 대신 사람이 조작한
스텝을 `is_human` 필드로 표시한다. 다른 데이터와 그대로 합칠 수 있다.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
# 인터랙티브 GUI 백엔드를 명시적으로 골라야 한다 — matplotlib.pyplot을 그냥
# import만 하면 환경에 따라(GUI 툴킷이 안 깔려 있는 등) 조용히 비인터랙티브
# 백엔드(Agg)로 떨어질 수 있고, 그러면 plt.ion()/plt.pause()가 에러 없이
# 그냥 아무 창도 안 띄운다. `matplotlib.use(name)` 자체는 해당 툴킷이 실제로
# import 가능한지 즉시 검증하지 않고 나중에(첫 Figure 생성 시) 실패할 수도
# 있어서(2026-08-24 실측 — try/except로 걸러지지 않고 그냥 조용히 넘어감),
# 후보마다 실제로 캔버스를 하나 만들어봐서 확실히 되는지까지 확인한다.
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
from grasp_carry.scripts.analyze.rollout_carry_diff_stats import load_diff_policy
from grasp_carry.scripts.record.record_carry import draw_env
from grasp_carry.train_carry_predictor import concat_obs

# record_carry.py는 자기 자신은 헤드리스 mp4 렌더링용이라 모듈 최상단에서
# matplotlib.use('Agg')를 무조건 호출한다 — 그 import(draw_env를 쓰려고 필요)가
# 위에서 애써 찾은 인터랙티브 백엔드를 조용히 Agg로 덮어써버린다(2026-08-30 실측:
# plt.show(block=False)/plt.pause()가 "FigureCanvasAgg is non-interactive" 경고만
# 내고 창이 안 뜨며, 마우스 이벤트가 전혀 안 들어와 사람 모드가 사실상 무력화됨).
# 확인된 인터랙티브 백엔드를 다시 강제한다.
matplotlib.use(_INTERACTIVE_BACKEND, force=True)


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


def make_dp_infer(ck: dict, nets, normalize_obs, unnormalize_action,
                  horizon: int, act_dim: int, seed: int):
  """관측 -> (horizon, act_dim) 언정규화 액션 청크를 내는 클로저.

  호출마다 역확산으로 새 청크 전체를 샘플링한다 — 그중 몇 스텝을 실행할지는
  호출부(`collect_one_episode`)가 `exec_horizon`으로 정한다
  (`collect_carry_bc_rollouts.py`와 동일한 receding-horizon 실행 방식).
  """
  _sample = jax.jit(lambda p, c, k: nets.sample_chunk(p, c, k))
  key = [jax.random.PRNGKey(seed)]

  def infer(obs):
    o = {'frame': np.asarray(obs, np.float32)[None]}
    c = np.asarray(concat_obs(normalize_obs(o)))
    key[0], sub = jax.random.split(key[0])
    chunk_n = np.asarray(_sample(ck['params'], jnp.asarray(c), sub)
                         )[0].reshape(horizon, act_dim)
    return np.asarray(unnormalize_action(chunk_n), np.float32)

  return infer


def collect_one_episode(env: GraspCarry2D, cfg: CarryConfig, seed: int,
                        ms: MouseState, fig, ax, dp_infer, exec_horizon: int,
                        pause_dt: float, discard: dict, toggle: dict):
  """에피소드 하나를 자동(DP 롤아웃)+사람 개입 섞어서 굴린다.

  `discard['flag']`가 True가 되면(‘r’ 키) 즉시 중단하고 (None, None, None,
  'discarded')를 반환한다. `toggle['flag']`가 True가 될 때마다(‘h’ 키)
  자동<->사람 모드가 즉시 뒤집힌다 — 자동 전환은 없고 오직 이 수동 토글로만
  바뀐다(2026-08-31, 사용자 결정). 모드가 바뀌면 그때까지 남아있던 DP 청크는
  버린다 — 사람이 상태를 바꿔놨을 수 있으므로 복귀 시 그 시점 관측으로 다시
  예측해야 한다.
  """
  obs, info = env.reset(seed=seed)

  obs_l, act_l, human_l = [obs], [], []
  mode = 'auto'  # 'auto'(DP가 접근~놓기까지 전체 수행) 또는 'human'
  chunk, chunk_i = None, 0

  terminated = truncated = False
  info_last = info
  for _ in range(cfg.max_steps):
    if discard['flag']:
      return None, None, None, 'discarded'

    if toggle['flag']:
      toggle['flag'] = False
      mode = 'human' if mode != 'human' else 'auto'
      chunk = None
      print(f'  >> 수동 전환: {mode}')

    if mode == 'auto':
      if chunk is None or chunk_i >= exec_horizon:
        chunk = dp_infer(obs)
        chunk_i = 0
      a = chunk[chunk_i]
      chunk_i += 1
    else:  # 'human'
      tx = ms.x if ms.x is not None else float(env.gripper.pose[0])
      ty = ms.y if ms.y is not None else float(env.gripper.pose[1])
      a = np.array([tx, ty, 0.0, ms.grip], dtype=np.float32)

    is_human_step = (mode == 'human')
    act_l.append(np.asarray(a, dtype=np.float32))
    human_l.append(is_human_step)

    obs, _, terminated, truncated, info = env.step(a)
    info_last = info

    draw_env(ax, env, action=a)
    tag = {'auto': 'auto(DP rollout)', 'human': 'human control'}[mode]
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
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100_v5/predictor.pkl',
                  help='굴릴 학습된 diffusion BC 정책 체크포인트')
  ap.add_argument('--dp-seed', type=int, default=1234,
                  help='DP 역확산 샘플링 PRNG 시드')
  ap.add_argument('--out', default='data/grasp_carry_demos_teleop_detour.pkl')
  ap.add_argument('--seed0', type=int, default=800000)
  ap.add_argument('--pause-dt', type=float, default=0.03,
                  help='프레임 사이 대기(초) — 너무 빠르면 마우스로 못 쫓아감')
  ap.add_argument('--autosave-every', type=int, default=5,
                  help='이 개수만큼 성공 에피소드가 쌓일 때마다 --out에 저장')
  args = ap.parse_args()

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  ck, nets, normalize_obs, normalize_action, unnormalize_action = load_diff_policy(args.diff_ckpt)
  m = ck['meta']
  horizon = int(m.get('horizon', 1))
  exec_horizon = int(m.get('exec_horizon', horizon))
  act_dim = int(m.get('act_dim', len(ck['norm_stats']['act_mean'])))
  dp_infer = make_dp_infer(ck, nets, normalize_obs, unnormalize_action,
                           horizon, act_dim, args.dp_seed)
  print(f'[DP 체크포인트] {args.diff_ckpt}  (horizon={horizon}, exec_horizon={exec_horizon})')

  print(f'[matplotlib 백엔드] {matplotlib.get_backend()} '
        f'(창이 안 뜨면 이 백엔드용 GUI 툴킷이 안 깔려있는 것 — '
        f'예: pip install pyqt5, 또는 Tk가 있다면 시스템 패키지로 python3-tk 설치)')
  plt.ion()
  fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
  try:
    fig.canvas.manager.set_window_title('GraspCarry2D DP rollout + human recovery')
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
            'source': 'collect_carry_teleop_detour.py: DP (diffusion BC) policy rollout '
                      '+ manual (h-key) human mouse takeover',
            'diff_ckpt': args.diff_ckpt,
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
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
      pickle.dump(data, fp)
    tag = '(최종)' if final else '(중간저장)'
    print(f'저장{tag}: 성공 {kept}개, {len(data["time_to_success"])}스텝 -> {args.out}')

  print(f'시작 (seed0={args.seed0}). 창에서 h=자동/사람 전환, r=버리고 재시작, q=저장 후 종료.')

  try:
    while not quit_flag['flag']:
      discard['flag'] = False
      toggle['flag'] = False
      print(f'--- 에피소드 seed={seed} ---')
      obs_l, act_l, human_l, outcome = collect_one_episode(
          env, cfg, seed, ms, fig, ax, dp_infer, exec_horizon,
          args.pause_dt, discard, toggle)
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
