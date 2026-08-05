r"""기댓값 전용 STG vs 위험 인지 STG — 자동 데이터 수집 효율 비교.

## 이 실험이 주장하려는 것

STG 예측기는 관례상 **성공 데모만으로** 학습한다(SI-EFM, `dp_policy.
collect_rollouts`, `collect_carry_demos.py`의 기본값). 그러면 그 예측기가
내놓는 값은 정의상 **"성공한다고 치면 몇 스텝"** 이다 — 실패를 표현할 클래스가
데이터에 아예 없다. 따라서 그 기댓값만으로 행동을 고르면 **"빠르게 가면
실패한다"를 원리적으로 말할 수 없다**(성공 궤적들 중에서는 빠른 쪽이 항상
더 짧으므로 항상 빠른 쪽이 이긴다). 이것이 조건 A가 뜻하는 바다.

자동 데이터 수집·자가성장 관점에서 이게 치명적인 이유: **실패한 시도는
시간만 쓰고 데이터셋에 아무것도 남기지 않는다.** 그래서 진짜 비용 함수는
성공률도, 성공 시 스텝 수도 아니고 **총 환경 스텝당 얻은 성공 데모 수**다.

## 통제 비교 설계

같은 아키텍처·하이퍼파라미터·시드, **학습 데이터 관례만** 다른 두 모델
(`src/train_carry_qstg.py`):

  모델 E (기댓값 전용): `--success-only` — 기존 관례 재현. P(성공)이 구조상
      ~1로 붕괴한다(실측: 성공 분류 정확도가 다수결 베이스라인과 소수점까지
      동일한 0.734 = 항상 "성공"이라고 답함).
  모델 R (위험 인지): 실패 bin 포함 전체. P(성공)과 E[스텝|성공]을 모두 낸다.

두 모델을 **같은 상태기계**(`ScriptedCarryPolicy`)의 같은 자리
(`speed_selector` 훅, `_speed_cap`과 동일 위치)에 꽂아 파지 직후 운반 속도만
고르게 한다. 상태기계의 나머지는 전부 동일하다.

  선택 E: argmin E[스텝]                       (기댓값만 씀)
  선택 R: argmax P(성공) / E[스텝|성공]        (재시도 포함 기대 처리량)

참고군으로 물리식(`_speed_cap`의 회전 저항 유도식, 은닉 물성의 보수적
분위수를 씀)도 같이 돌린다 — 학습 없이 손으로 유도한 안전 상한이다.

    python compare_carry_selectors.py --episodes 200
"""

import argparse
import pickle

import numpy as np
import jax.numpy as jnp

from probe_carry_qstg import load_ckpt
from src.train_carry_qstg import split_success_fail, succ_mean_quantile
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy

# 후보 리드(mm). calibrate_carry.py의 스윕과 같은 격자를, 학습 데이터를 모은
# 상한(`collect_carry_demos.py --speed 150`)까지로 자른 것.
CANDIDATE_LEADS = (12.5, 25.0, 50.0, 75.0, 100.0, 150.0)
# 세 팔 모두 같은 값을 쓴다. `speed`는 (1) 수평 리드의 상한이자 (2) **연직**
# 리드(`_lift_lead = 최악 처짐 + speed`)를 정한다. 팔마다 다르게 두면 수평
# 선택 규칙 말고 연직 물리까지 달라져 비교가 오염된다. 값은 데이터를 모은
# 설정(--speed 150)과 맞춰 학습 분포 안에 머물게 한다.
NOMINAL_SPEED = 150.0


class PredictorSelector:
  """예측기로 후보 리드 중 하나를 고르는 `speed_selector`.

  **두 모델(E/R) 다 완전히 같은 규칙을 쓴다**: "예측 성공확률이
  `success_threshold` 이상인 후보 중 가장 빠른 것, 하나도 안 넘으면 가장
  느린(가장 안전한) 후보". 차이는 오직 그 확률을 누가 예측하느냐뿐이다 —
  모델 E는 성공 데모만 봐서 모든 후보에 ~1.0을 답하므로 이 규칙에서
  **항상 최고속을 고를 수밖에 없다.** 이전 버전은 모델별로 다른 점수식
  (E는 argmin 스텝, R은 처리량 비율)을 썼는데, 그러면 "모델이 달라서
  다른가 규칙이 달라서 다른가"가 섞인다.
  """

  def __init__(self, ckpt_path: str, success_threshold: float = 0.95,
               candidates=CANDIDATE_LEADS):
    self.ck, self.apply_fn = load_ckpt(ckpt_path)
    self.success_threshold = success_threshold
    self.candidates = np.asarray(candidates, dtype=np.float64)
    self.stats = self.ck['norm_stats']
    ff = self.ck['frame_fields']
    self.ex_i = ff.index('ee_x')
    self.last0 = (self.ck['obs_history'] - 1) * len(ff)
    self.fail_bin = self.ck['fail_bin']
    self.bin_vals = jnp.arange(self.ck['num_bins'], dtype=jnp.float32)
    self.chosen = []           # 진단용 — 실제로 고른 리드들
    self.p_succ_at_pick = []   # 진단용 — 그 선택의 예측 성공확률

  def __call__(self, env, contact_len, arm):
    obs = np.asarray(env._stacked_obs(), dtype=np.float32)[None]
    L = env.cfg.world_width
    ex_mm = float(obs[0, self.last0 + self.ex_i]) * L
    # 목적지(타겟 박스)를 향하는 방향으로 각 후보 리드만큼 목표를 둔다.
    sign = np.sign(env.tgt_box.center_x - ex_mm) or 1.0
    ey_mm = float(env.gripper.pose[1])
    n = len(self.candidates)
    tx = ex_mm + sign * self.candidates
    acts = np.stack([tx, np.full(n, ey_mm), np.zeros(n), np.ones(n)],
                    axis=-1).astype(np.float32)
    acts_n = (acts - self.stats['act_mean']) / self.stats['act_std']
    obs_n = (obs - self.stats['frame_mean']) / self.stats['frame_std']
    x = jnp.asarray(np.concatenate([np.repeat(obs_n, n, axis=0), acts_n],
                                   axis=-1))
    logits = self.apply_fn(self.ck['params'], x)
    p_succ, succ_probs = split_success_fail(logits, self.fail_bin)
    mean, _ = succ_mean_quantile(succ_probs, self.bin_vals[:self.fail_bin])
    mean = np.asarray(mean); p_succ = np.asarray(p_succ)

    # 후보는 느림→빠름 순(CANDIDATE_LEADS)이라고 가정 — 문턱 통과하는 가장
    # 빠른 후보를 찾는다. 하나도 못 넘으면 가장 느린(=가장 안전한) 후보로.
    order = np.argsort(self.candidates)               # 느림 -> 빠름
    ok = p_succ[order] >= self.success_threshold
    if ok.any():
      pick_idx = order[np.where(ok)[0][-1]]            # 통과하는 것 중 가장 빠름
    else:
      pick_idx = order[0]                               # 전부 불안 -> 가장 느림
    pick = float(self.candidates[pick_idx])
    self.chosen.append(pick)
    self.p_succ_at_pick.append(float(p_succ[pick_idx]))
    return pick


def run(make_policy, episodes, seed0, cfg):
  """같은 시드 집합으로 굴리고 데이터 수집 관점 지표를 낸다."""
  env = GraspCarry2D(cfg)
  n_succ, succ_steps, total_steps, outcomes = 0, [], 0, {}
  for e in range(episodes):
    policy = make_policy()
    env.reset(seed=seed0 + e)
    policy.reset()
    info = env._info()
    for _ in range(cfg.max_steps):
      _, _, term, trunc, info = env.step(policy(env))
      if term or trunc:
        break
    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    total_steps += info['steps']          # 실패한 시도의 스텝도 비용이다
    if info['outcome'] == 'success':
      n_succ += 1
      succ_steps.append(info['steps'])
  return dict(episodes=episodes, n_succ=n_succ, outcomes=outcomes,
              success_rate=n_succ / episodes, total_steps=total_steps,
              mean_succ_steps=float(np.mean(succ_steps)) if succ_steps else float('nan'),
              demos_per_1k_steps=1000.0 * n_succ / max(total_steps, 1))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--seed0', type=int, default=900000,
                  help='수집 데이터(seed0=0..499)와 겹치지 않는 홀드아웃 구간.')
  ap.add_argument('--exp-ckpt',
                  default='checkpoints/grasp_carry_qstg_exponly/predictor.pkl')
  ap.add_argument('--risk-ckpt',
                  default='checkpoints/grasp_carry_qstg/predictor.pkl')
  ap.add_argument('--success-threshold', type=float, default=0.95,
                  help='두 모델이 공유하는 규칙의 문턱값(예측 성공확률).')
  args = ap.parse_args()

  cfg = CarryConfig()
  sel_e = PredictorSelector(args.exp_ckpt, args.success_threshold)
  sel_r = PredictorSelector(args.risk_ckpt, args.success_threshold)

  arms = [
      ('기댓값 전용 (성공만 학습)',
       lambda: ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED,
                                   speed_selector=sel_e)),
      ('위험 인지 (실패 포함 학습)',
       lambda: ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED,
                                   speed_selector=sel_r)),
      ('물리식 (학습 없음, 참고군)',
       lambda: ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED)),
  ]

  rows = []
  for name, mk in arms:
    r = run(mk, args.episodes, args.seed0, cfg)
    rows.append((name, r))
    print(f'{name}: 완료', flush=True)

  print(f'\n{"선택 규칙":<26} {"성공률":>7} {"성공시 스텝":>10} '
        f'{"총 스텝":>9} {"1k스텝당 데모":>12}')
  for name, r in rows:
    print(f'{name:<26} {r["success_rate"]:>7.1%} {r["mean_succ_steps"]:>10.1f} '
          f'{r["total_steps"]:>9d} {r["demos_per_1k_steps"]:>12.2f}')
  print()
  for name, r in rows:
    print(f'  {name}: outcomes={r["outcomes"]}')
  print(f'\n공유 규칙: "예측 성공확률 >= {args.success_threshold:.0%}인 후보 중 '
        f'가장 빠른 것"')
  for sel, label in ((sel_e, '기댓값 전용'), (sel_r, '위험 인지')):
    if sel.chosen:
      c = np.asarray(sel.chosen)
      uniq, cnt = np.unique(c, return_counts=True)
      dist = ', '.join(f'{u:g}:{n}' for u, n in zip(uniq, cnt))
      ps = np.asarray(sel.p_succ_at_pick)
      print(f'  {label}: 선택한 리드 분포 = {dist}')
      print(f'    선택 시점에 자기가 예측한 성공확률: '
            f'평균={ps.mean():.3f} 최소={ps.min():.3f}')


if __name__ == '__main__':
  main()
