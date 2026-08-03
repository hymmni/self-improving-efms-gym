r"""GraspCarry2D 탐색 롤아웃을 STG 학습용 데모로 저장.

    python collect_carry_demos.py --episodes 500 --explore-range 0.4 200 \
        --out data/grasp_carry_demos.pkl

`src/dp_policy.py`의 `collect_rollouts()`(PushT)와 **동일 포맷·동일 라벨 정의**를 쓴다:
  - 성공 에피소드만 저장한다(성공 시점까지 궤적을 자른다. 논문(SI-EFM)과 동일).
  - `time_to_success[t] = (성공 시점 - t)`, 성공 상태에서 0.
  - `observation`은 `env.step()`/`env.reset()`이 실제로 내주는 **스택된** 값
    그대로다(은닉 물성 없음 — `env.observe_frame()`의 설계).

정책은 `ScriptedCarryPolicy`를 `explore_range`로 돌린다. `_speed_cap()`의 안전식은
정의상 절대 안 떨어뜨리는 속도만 내놓으므로(그게 그 식의 목적이다), 그 식을
우회해야 얕은 파지에서 실제로 놓치는/넘어지는 사례가 데이터에 남는다 — 안전식
그대로는 명목 속도든 상한을 아무리 올리든 위험 영역에 진입하지 않는다는 것이
실측됐다(`policy.py` 문서, `experiments/` 기록 참고).
"""

import argparse
import os
import pickle
from typing import Optional, Tuple

import numpy as np

from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import FRAME_FIELDS, GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy


def collect(n_episodes: int, explore_range: Tuple[float, float],
            seed0: int = 0, explore_seed: int = 0,
            allow_regrasp: bool = True, speed: Optional[float] = None,
            keep_failures: bool = False,
            config: Optional[CarryConfig] = None) -> dict:
  """`speed`가 `_lift_lead`(연직 리드 = 최악 처짐 + speed)를 정한다 — `explore_range`는
  수평 채널만 우회하므로, 연직 슬립(들어올릴 때 그립력이 못 버티는 것)까지
  실제로 만들려면 `speed`도 올려야 한다(둘은 독립된 채널이다).

  `keep_failures=True`면 실패 에피소드도 저장한다. 그 안의 모든 스텝은
  `time_to_success = max_steps`(카테고리 STG 헤드의 마지막 bin, 유효한
  0..max_steps-1 범위 밖)로 라벨링된다 — "성공까지 남은 스텝"이 아니라
  "이 궤적은 결국 실패했다"는 별도 클래스다. 이 라벨을 학습에 그대로 쓰면
  범주형 분포 하나에서 `P(성공) = 1 - P(마지막 bin)`과, 성공 bin만
  재정규화한 "성공한다면 몇 스텝"을 둘 다 뽑을 수 있다(성공률과 성공 분포
  quantile을 같은 모델에서 얻는 방법 — 실패를 배제하던 기존 기본값과 달리
  조건 A를 액션-조건부로 검증하려면 이게 필요하다)."""
  cfg = config or CarryConfig()
  env = GraspCarry2D(cfg)
  rng = np.random.default_rng(explore_seed)

  obs_all, act_all, ttg_all, eid_all = [], [], [], []
  contact_all, held_all, speed_all, succ_all = [], [], [], []
  mass_all, mu_all = [], []
  outcomes: dict = {}
  kept = 0

  for e in range(n_episodes):
    # 정책 인스턴스를 매 에피소드 새로 만들되, 탐색용 RNG는 실행 전체에서 하나만
    # 이어 써서(재현성 있게) 같은 seed0 재실행이 같은 데이터를 낸다.
    policy = ScriptedCarryPolicy(config=cfg, allow_regrasp=allow_regrasp,
                                 explore_range=explore_range, rng=rng,
                                 speed=speed)
    obs, info = env.reset(seed=seed0 + e)
    policy.reset()

    obs_l, act_l, contact_l, held_l, speed_l = [obs], [], [], [], []
    for _ in range(cfg.max_steps):
      a = policy(env)
      act_l.append(a.astype(np.float32))
      contact_l.append(env.contact_length())
      held_l.append(bool(env.is_held()))
      speed_l.append(float(policy._speed_hold))
      obs, _, terminated, truncated, info = env.step(a)
      if terminated or truncated:
        break
      obs_l.append(obs)

    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    is_success = info['outcome'] == 'success'
    if not is_success and not keep_failures:
      continue                                    # 미성공 -> 제외 (논문과 동일)

    L = len(act_l)
    assert len(obs_l) == L, '관측/액션 스텝 수 불일치'
    if is_success:
      ttg = (L - 1 - np.arange(L)).astype(np.float32)
    else:
      ttg = np.full(L, cfg.max_steps, dtype=np.float32)   # 실패 bin(마지막)
    obs_all.append(np.array(obs_l, dtype=np.float32))
    act_all.append(np.array(act_l, dtype=np.float32))
    contact_all.append(np.array(contact_l, dtype=np.float32))
    held_all.append(np.array(held_l, dtype=bool))
    speed_all.append(np.array(speed_l, dtype=np.float32))
    succ_all.append(np.full(L, is_success, dtype=bool))
    ttg_all.append(ttg)
    eid_all.append(np.full(L, kept, dtype=np.int32))
    mass_all.append(np.full(L, info['mass'], dtype=np.float32))
    mu_all.append(np.full(L, info['friction'], dtype=np.float32))
    kept += 1

  data = {
      'observation': {'frame': (np.concatenate(obs_all) if obs_all
                                else np.zeros((0, env.obs_dim),
                                             dtype=np.float32))},
      'action': (np.concatenate(act_all) if act_all
                else np.zeros((0, 4), dtype=np.float32)),
      'time_to_success': (np.concatenate(ttg_all) if ttg_all
                          else np.zeros((0,), dtype=np.float32)),
      'episode_id': (np.concatenate(eid_all) if eid_all
                    else np.zeros((0,), dtype=np.int32)),
      # 관측 가능 진단값 — 학습에 넣어도 되지만 필수는 아니다.
      'contact_length': (np.concatenate(contact_all) if contact_all
                        else np.zeros((0,), dtype=np.float32)),
      'is_held': (np.concatenate(held_all) if held_all
                 else np.zeros((0,), dtype=bool)),
      'commanded_speed': (np.concatenate(speed_all) if speed_all
                         else np.zeros((0,), dtype=np.float32)),
      # keep_failures=False면 전부 True(성공만 남았으므로). True면 실패
      # 에피소드의 스텝은 전부 False — time_to_success가 실패 bin(=max_steps)
      # 인지와 동치이지만, 필터링을 더 읽기 쉽게 하려고 따로 둔다.
      'is_success': (np.concatenate(succ_all) if succ_all
                    else np.zeros((0,), dtype=bool)),
      # 은닉 물성 — **진단·분석 전용**. env.observe_frame()이 이미 관측에서
      # 뺐으므로 여기 있다고 학습 입력에 오염되지 않지만, 실수로 넣지 마라.
      'hidden_mass': (np.concatenate(mass_all) if mass_all
                     else np.zeros((0,), dtype=np.float32)),
      'hidden_friction': (np.concatenate(mu_all) if mu_all
                         else np.zeros((0,), dtype=np.float32)),
      'meta': {
          'source': ('collect_carry_demos.py: '
                     'ScriptedCarryPolicy(explore_range=...) rollouts'),
          'frame_fields': list(FRAME_FIELDS),
          'obs_history': cfg.obs_history,
          'action_fields': ('x', 'y', 'theta', 'grip'),
          'explore_range': tuple(explore_range),
          'speed': speed,
          'allow_regrasp': allow_regrasp,
          'keep_failures': keep_failures,
          'failure_bin': cfg.max_steps,
          'requested_episodes': n_episodes,
          'kept_episodes': kept,
          'outcomes': outcomes,
          'seed0': seed0,
          'explore_seed': explore_seed,
          'control_hz': cfg.control_hz,
          'max_steps': cfg.max_steps,
      },
  }
  kept_label = f'{kept}(전체)' if keep_failures else f'성공 {kept}'
  print(f'수집 완료: {kept_label}/{n_episodes} 에피소드, '
        f'{len(data["time_to_success"])} 스텝, outcomes={outcomes}')
  return data


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=500)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--explore-range', type=float, nargs=2, default=[0.4, 200.0],
                  metavar=('LOW', 'HIGH'))
  ap.add_argument('--explore-seed', type=int, default=0)
  ap.add_argument('--speed', type=float, default=None,
                  help=('연직 리드(_lift_lead) 상한. explore_range는 수평만 '
                        '우회하므로, 연직 슬립 채널까지 위험하게 하려면 '
                        '이것도 명목값(기본 12.5)보다 올려야 한다.'))
  ap.add_argument('--no-regrasp', action='store_true',
                  help='재파지를 금지하고 항상 직접 운반한다.')
  ap.add_argument('--keep-failures', action='store_true',
                  help=('실패 에피소드도 저장한다. 그 안의 모든 스텝은 '
                        'time_to_success=max_steps(마지막 bin, 실패 클래스)로 '
                        '라벨링된다. 기본은 꺼져 있다(성공만, 논문과 동일).'))
  ap.add_argument('--out', default='data/grasp_carry_demos.pkl')
  args = ap.parse_args(argv)

  data = collect(args.episodes, tuple(args.explore_range), seed0=args.seed0,
                 explore_seed=args.explore_seed,
                 allow_regrasp=not args.no_regrasp, speed=args.speed,
                 keep_failures=args.keep_failures)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump(data, fp)
  print(f'saved {args.out}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
