"""장애물 회피 환경 수동 조종 뷰어 — 마우스로 에이전트를 직접 몰면서
steps-to-go 분포가 실시간으로 어떻게 반응하는지 관찰한다.

  DISPLAY=:1 python drive_obstacle.py

조작:
  마우스 이동   에이전트가 커서를 향해 PD 제어로 끌려온다 (커서 = 당근).
                왼쪽 환경 패널 위에서만 동작. 관성이 있으니 급회전 불가.
  p            학습된 정책에게 조종을 넘김/뺏음 (토글)
  space        일시정지/재개
  n            새 에피소드 (장애물/골/시작 재배치)
  s            세션 저장 (results/manual_sessions/ 에 png 요약 + pkl 기록)
  q            종료

시간 제한 없음. 골에 도달하면 "성공!" 표시만 하고 계속 조종할 수 있다(n으로
새 에피소드). 분포 패널은 항상 '진짜 골' 기준 — 마우스로 골 반대편으로 끌고
가면 예측 분포가 어떻게 퍼지는지 직접 실험해볼 수 있다.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

CKPT_DEFAULT = 'checkpoints/obstacle/predictor.pkl'
SAVE_DIR = 'results/manual_sessions'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', default=CKPT_DEFAULT)
  ap.add_argument('--fps', type=float, default=30)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  if not os.environ.get('DISPLAY'):
    sys.exit('DISPLAY가 비어 있습니다. DISPLAY=:1 python drive_obstacle.py 로 실행하세요.')

  import matplotlib
  matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False
  # matplotlib 기본 단축키(s=저장 대화상자, p=팬, q=닫기)가 우리 키와 겹치므로 해제
  for km, k in [('keymap.save', 's'), ('keymap.pan', 'p'), ('keymap.quit', 'q')]:
    if k in plt.rcParams[km]:
      plt.rcParams[km].remove(k)

  import jax
  # From: pointmass_core.pd_controller — 마우스 추종도 동일 게인의 PD로 구동
  from pointmass_core import pd_controller
  from src.obstacle_env import ObstacleAvoidPoint2D

  probe = None
  if os.path.exists(args.checkpoint):
    from src.probe_generic import GenericSTGProbe
    probe = GenericSTGProbe(args.checkpoint)
    print(f'예측기 로드: {args.checkpoint}')
  else:
    print(f'체크포인트 없음({args.checkpoint}) — 분포 패널 없이 실행')

  # ---- 레이아웃
  if probe is not None:
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, 1])
    ax_ts2 = ax_ts.twinx()  # 반드시 1회만 생성 (누적 버그 방지)
  else:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax_dist = ax_ts = ax_ts2 = None
  fig.canvas.manager.set_window_title('obstacle-avoid manual drive')

  state = dict(mouse=None, paused=False, policy_drive=False, quit=False,
               new_episode=False, save=False)

  def on_move(event):
    if event.inaxes is ax and event.xdata is not None:
      state['mouse'] = np.array([event.xdata, event.ydata], dtype=np.float32)

  def on_key(event):
    if event.key == 'q':
      state['quit'] = True
    elif event.key == 'n':
      state['new_episode'] = True
    elif event.key == ' ':
      state['paused'] = not state['paused']
    elif event.key == 'p':
      state['policy_drive'] = not state['policy_drive']
    elif event.key == 's':
      state['save'] = True

  fig.canvas.mpl_connect('motion_notify_event', on_move)
  fig.canvas.mpl_connect('key_press_event', on_key)

  key = jax.random.PRNGKey(7)
  seed = args.seed
  episode = 0

  def save_session(env, traj, records, step):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(SAVE_DIR, f'session_{stamp}')
    with open(base + '.pkl', 'wb') as fp:
      pickle.dump(dict(
          traj=np.array(traj), goal=env._goal_pos.copy(),
          obstacle=env._cur_obstacle, steps=step,
          success=bool(env.success()),
          records=[dict(step=r.step_idx, expectation=r.expectation,
                        variance=r.variance, probs=r.probs)
                   for r in records] if records else None), fp)
    fig.savefig(base + '.png', dpi=120)
    print(f'세션 저장: {base}.png / .pkl')

  while not state['quit'] and plt.fignum_exists(fig.number):
    # ---------------- 에피소드 셋업
    np.random.seed(seed)
    env = ObstacleAvoidPoint2D()
    ts = env.reset()
    traj = [env._cur_pos.copy()]
    records = []
    center, radius = env._cur_obstacle
    state['new_episode'] = False
    reached = False

    ax.clear()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.add_patch(patches.Circle(center, radius, facecolor='gray',
                                edgecolor='black', alpha=0.5))
    ax.add_patch(patches.Circle(env._goal_pos, env._success_radius,
                                edgecolor='green', ls='--', fill=False))
    ax.scatter(*env._goal_pos, marker='*', s=180, color='orange', zorder=5)
    line, = ax.plot([], [], '-', color='blue', lw=1.2, alpha=0.8)
    dot, = ax.plot([], [], 'o', color='red', ms=9, zorder=6)
    cursor_dot, = ax.plot([], [], 'x', color='purple', ms=10, mew=2, zorder=6)

    if probe is not None:
      ax_dist.clear()
      ax_dist.set_xlabel('steps-to-go'); ax_dist.set_ylabel('prob')
      ax_dist.set_title('STG 분포 (진짜 골 기준, 실시간)', fontsize=10)
      dist_line, = ax_dist.plot([], [], color='C0', lw=1.0)
      ax_dist.set_xlim(0, 300)
      ax_ts.clear(); ax_ts2.clear()
      ax_ts.set_xlabel('step'); ax_ts.set_ylabel('E[STG]', color='C0')
      ax_ts2.set_ylabel('σ²', color='C3')
      e_line, = ax_ts.plot([], [], color='C0', lw=1.2)
      v_line, = ax_ts2.plot([], [], color='C3', lw=1.2, ls='--')
      exps, varis = [], []

    step = 0
    # ---------------- 메인 루프 (시간 제한 없음)
    while (not state['quit']) and (not state['new_episode']) \
          and plt.fignum_exists(fig.number):
      if state['save']:
        save_session(env, traj, records, step)
        state['save'] = False

      if not state['paused']:
        obs = ts.observation
        rec = None
        act_norm = None
        if probe is not None:
          key, sub = jax.random.split(key)
          act_norm, logits = probe._logits_and_act(obs, sub)
          rec = probe._record(step, obs, logits)
          records.append(rec)

        if state['policy_drive'] and probe is not None:
          action = probe.unnormalize_action(act_norm)[0]
        elif state['mouse'] is not None:
          action = pd_controller(env._cur_pos, env._cur_vel, state['mouse'])
        else:
          action = np.zeros(2, dtype=np.float32)  # 커서 진입 전엔 정지

        ts = env.step(np.asarray(action, dtype=np.float32))
        traj.append(env._cur_pos.copy())
        step += 1
        if env.success() and not reached:
          reached = True

        t = np.array(traj)
        line.set_data(t[:, 0], t[:, 1])
        dot.set_data([t[-1, 0]], [t[-1, 1]])
        if state['mouse'] is not None and not state['policy_drive']:
          cursor_dot.set_data([state['mouse'][0]], [state['mouse'][1]])
        else:
          cursor_dot.set_data([], [])

        mode = '정책 조종' if state['policy_drive'] else '마우스 조종'
        info = (f'ep{episode} (seed {seed})  [{mode}]  step {step}  '
                f'속력 {np.linalg.norm(env._cur_vel):.4f}'
                f'{"  ★성공! (n=새판)" if reached else ""}')
        if rec is not None:
          info += f'\nE[STG]={rec.expectation:.1f}  σ²={rec.variance:.0f}'
          dist_line.set_data(probe.bin_vals, rec.probs)
          ax_dist.set_ylim(0, max(0.05, float(rec.probs.max()) * 1.2))
          exps.append(rec.expectation); varis.append(rec.variance)
          e_line.set_data(range(len(exps)), exps)
          v_line.set_data(range(len(varis)), varis)
          ax_ts.relim(); ax_ts.autoscale_view()
          ax_ts2.relim(); ax_ts2.autoscale_view()
        ax.set_title(info + '\n(p=정책토글 space=일시정지 n=새판 s=저장 q=종료)',
                     fontsize=9)
      plt.pause(max(1.0 / args.fps, 0.001))

    episode += 1
    seed += 1


if __name__ == '__main__':
  main()
