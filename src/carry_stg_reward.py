r"""얼려진 관측-only STG 예측기를 SI-EFM 식(1)~(3)의 보상/성공판정으로 감싼다.

배경(코드가 아니라 여기 남기는 이유: 이 파일의 존재 이유 자체가 이론적
근거이기 때문이다) — 논문 식(5)는 `r_t = d(o_t,g) - d(o_{t+1},g)`가 잠재함수
`Phi = -d`인 PBRS(Ng et al. 1999)임을 보인다. 통상 PBRS는 "정책 불변"이
보장되는데, 그 보장은 shaping이 더해지는 **원래 보상**이 있다는 전제에서
나온다. SI-EFM에는 더해질 원래 보상이 없다 — 식(5)를 전개하면 base reward
항조차 `-(1-gamma) d(o_{t+1},g)`로 `d` 자신이 만든다. 즉 여기서는 **`d`의
정의가 곧 목적함수 자체**다. `d`를 기댓값(mean)에서 꼬리 위험(CVaR)으로
바꾸면 목적이 "기대 잔여 스텝 최소화"에서 "최악 구간 잔여 스텝 최소화"로
바뀐다 — 정책 불변이 아니라 정책이 달라지는 것이 기대된 결과다. 이 클래스는
그 정의(통계량)를 교체 가능한 부품으로 노출한다.

phase 4 step 1(`src/train_carry_dstg.py`)이 학습한 두 체크포인트를 그대로
쓴다:
  succ (fail_bin=None): 성공 transition만 학습, num_bins=200.
  fail (fail_bin=200)  : 실패까지 포함, 마지막 bin이 "실패" 클래스.

파라미터는 절대 업데이트하지 않는다 (논문 Algorithm 1: "Initialize and
freeze a separate Stage 1 checkpoint for reward computation and success
detection").

실행:
  python -m src.carry_stg_reward --ckpt checkpoints/grasp_carry_dstg_succ/predictor.pkl
  python -m src.carry_stg_reward --ckpt checkpoints/grasp_carry_dstg_fail/predictor.pkl \
      --statistic cvar --episodes 50
"""

import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from src.train_carry_dstg import build_dstg_net
from src.train_carry_qstg import succ_cvar


# succ_cvar(probs, bin_vals, alpha)는 수학적으로 "확률질량 1짜리 카테고리컬
# 분포의 상위 (1-alpha) 꼬리 CVaR"만 계산한다 — 입력이 "성공만 재정규화한
# 확률"이어야 한다는 제약은 어디에도 없다. 그래서 실패 bin을 포함한 전체
# 분포(질량 합 1)를 그대로 넣으면 "실패를 포함한 하나의 분포의 CVaR"이 된다.
# 이를 여기서 `_tail_cvar`로 재노출해 의도를 분명히 한다(계산 로직 자체는
# src/train_carry_qstg.py를 그대로 재사용 — 새로 구현하지 않는다).
_tail_cvar = succ_cvar


def _d_from_probs(probs, bin_vals, statistic, cvar_alpha):
  """(B, num_bins) 확률 -> (B,) d값. 통계량 선택 지점 — d의 정의가 여기서 갈린다."""
  if statistic == 'mean':
    return jnp.sum(probs * bin_vals[None, :], axis=-1)
  elif statistic == 'cvar':
    return _tail_cvar(probs, bin_vals, cvar_alpha)
  else:
    raise ValueError(f"statistic must be 'mean' or 'cvar', got {statistic!r}")


class StgReward:
  """얼려진 관측-only STG 예측기를 논문 식(1)~(3)의 d/보상/성공판정으로 감싼다.

  통계량(statistic)을 바꾸면 d의 정의가 바뀐다 — 그것이 이 클래스의 존재
  이유다. 파라미터는 절대 업데이트하지 않는다 (논문 Algorithm 1: "Initialize
  and freeze a separate Stage 1 checkpoint for reward computation and success
  detection").
  """

  def __init__(self, ckpt_path: str, statistic: str = 'mean',
               cvar_alpha: float = 0.8, fail_value: float | None = None,
               threshold: float | None = None):
    """
    statistic : 'mean' | 'cvar'
    cvar_alpha: statistic='cvar'일 때 상위 (1-alpha) 꼬리만 본다 (기본 0.8 → 최악 20%)
    fail_value: 실패 bin에 부여할 스텝 환산값. None이면 max_steps(=200)를 쓴다.
                fail_bin이 None인 체크포인트에서는 무시된다.
    threshold : 식(3)의 s. None이면 calibrate_threshold()로 정해서 넣어야 한다.
    """
    if statistic not in ('mean', 'cvar'):
      raise ValueError(f"statistic must be 'mean' or 'cvar', got {statistic!r}")
    with open(ckpt_path, 'rb') as fp:
      ck = pickle.load(fp)

    self.ckpt_path = ckpt_path
    self.statistic = statistic
    self.cvar_alpha = cvar_alpha
    self.threshold = threshold

    self.params = ck['params']
    self.norm_stats = ck['norm_stats']
    self.obs_dim = int(ck['obs_dim'])
    self.num_bins = int(ck['num_bins'])
    self.fail_bin = ck['fail_bin']
    self.max_steps = int(ck['max_steps'])
    self.layer_sizes = tuple(ck['layer_sizes'])
    self.meta = ck['meta']
    self.fail_value = (float(fail_value) if fail_value is not None
                       else float(self.max_steps))

    apply_fn, _ = build_dstg_net(self.layer_sizes, self.obs_dim, self.num_bins)
    self._apply = jax.jit(apply_fn)

    bin_vals = np.arange(self.num_bins, dtype=np.float32)
    if self.fail_bin is not None:
      bin_vals[self.fail_bin] = self.fail_value
    self._bin_vals = jnp.asarray(bin_vals)

  def _probs(self, obs_raw: np.ndarray) -> np.ndarray:
    obs_raw = np.asarray(obs_raw, dtype=np.float32)
    obs_n = ((obs_raw - self.norm_stats['frame_mean'])
             / self.norm_stats['frame_std'])
    logits = self._apply(self.params, jnp.asarray(obs_n))
    return np.asarray(jax.nn.softmax(logits, axis=-1))

  def d(self, obs_raw: np.ndarray) -> np.ndarray:
    """obs_raw: (B, obs_dim) 정규화 **안 된** 원본 관측 (env._stacked_obs()의 것).
    반환 (B,) float — 값이 클수록 나쁘다(목표까지 멀다).
    내부에서 이 체크포인트의 norm_stats로 정규화한다 — 호출자가 정규화하지 않는다.
    """
    probs = jnp.asarray(self._probs(obs_raw))
    return np.asarray(_d_from_probs(probs, self._bin_vals, self.statistic,
                                    self.cvar_alpha))

  def success(self, obs_raw: np.ndarray) -> np.ndarray:
    """식(3): 1[d(o) <= threshold]. 반환 (B,) bool."""
    if self.threshold is None:
      raise ValueError(
          'threshold가 설정되지 않았다 — calibrate_threshold()로 구한 뒤 '
          'reward.threshold = s 로 설정하라.')
    return self.d(obs_raw) <= self.threshold

  def mean_std(self, obs_raw: np.ndarray) -> tuple:
    """예측 분포의 평균 μ(o)와 표준편차 σ(o)를 함께 낸다(references/context_8.md
    분석용 — d()는 statistic에 따라 mean/cvar 중 하나만 내므로 σ가 따로 필요).
    fail_bin이 있는 체크포인트는 d()와 동일하게 fail_value로 치환한 bin_vals를
    쓴다(같은 분포 위에서 일관되게). 반환: (mu, sigma) 각 (B,) float.
    """
    probs = self._probs(obs_raw)
    bv = np.asarray(self._bin_vals)
    mu = np.sum(probs * bv[None, :], axis=-1)
    var = np.sum(probs * (bv[None, :] - mu[:, None]) ** 2, axis=-1)
    return mu, np.sqrt(np.maximum(var, 0.0))


def _val_episode_ids(data: dict, seed: int) -> set:
  """예측기 체크포인트 학습 시 쓴 것과 동일한 10% 에피소드 val 분할을
  재현한다.

  From: src/train_carry_dstg.py (val_eps 계산 로직) — 체크포인트가 실제로
  학습에서 못 본 에피소드에서만 문턱을 캘리브레이션하기 위해 그대로 따른다.
  """
  rng = np.random.default_rng(seed)
  ep_ids = np.unique(data['episode_id'])
  n_val = max(len(ep_ids) // 10, 1)
  return set(rng.choice(ep_ids, size=n_val, replace=False).tolist())


def calibrate_threshold(reward: StgReward, data: dict, val_episode_ids,
                        ) -> tuple:
  """held-out 에피소드에서 식(3)의 s를 정한다.

  라벨: 그 transition이 "성공 시점"인가. 성공 에피소드의 마지막 프레임
        (time_to_success == 0)을 양성, 나머지를 음성으로 한다.
  방법: s 후보를 훑어 F1이 최대가 되는 s를 고른다.
  반환: (best_s, metrics) — metrics에 precision/recall/f1/best_s를 담는다.
  """
  mask = np.isin(data['episode_id'], list(val_episode_ids))
  obs = data['observation']['frame'][mask]
  is_succ = np.asarray(data['is_success'])[mask]
  ttg = np.asarray(data['time_to_success'])[mask]
  label = is_succ & (ttg == 0)

  d_vals = reward.d(obs)
  lo, hi = float(d_vals.min()), float(d_vals.max())
  s_candidates = np.linspace(lo, hi, 400)

  best = dict(f1=-1.0, s=float(s_candidates[0]), precision=0.0, recall=0.0)
  for s in s_candidates:
    pred = d_vals <= s
    tp = float(np.sum(pred & label))
    fp = float(np.sum(pred & ~label))
    fn = float(np.sum(~pred & label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    if f1 > best['f1']:
      best = dict(f1=f1, s=float(s), precision=precision, recall=recall)

  metrics = dict(precision=best['precision'], recall=best['recall'],
                f1=best['f1'], best_s=best['s'])
  return best['s'], metrics


# ---------------------------------------------------------------- 검증 스크립트
def _diagnostic_rollout(policy, env, reward: StgReward, seed: int,
                        max_steps: int):
  """rollout()(run_bc_stg_guided.py)과 같은 loop 구조로 env.step을 몰되,
  매 스텝 reward.d/success를 곁다리로 기록한다."""
  env.reset(seed=seed)
  policy.reset()
  d_trace = []
  first_pred_success_step = None
  info = env._info()
  for t in range(max_steps):
    obs = np.asarray(env._stacked_obs(), dtype=np.float32)[None]
    d_val = float(reward.d(obs)[0])
    pred_succ = bool(reward.success(obs)[0])
    d_trace.append(d_val)
    if pred_succ and first_pred_success_step is None:
      first_pred_success_step = t
    _, _, term, trunc, info = env.step(policy(env))
    if term or trunc:
      break
  return info, d_trace, first_pred_success_step


def _run_rollout_eval(reward: StgReward, bc_ckpt: str, episodes: int,
                      seed0: int, max_steps: int):
  # 순환 import 회피(run_bc_stg_guided.py가 src.* 모듈을 import하므로 여기서
  # 모듈 레벨에 두면 -m 실행 시 얽힐 수 있다 — 함수 내부 import로 늦춘다).
  from run_bc_stg_guided import BCStgGuided
  from src.grasp_carry.config import CarryConfig
  from src.grasp_carry.env import GraspCarry2D

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)
  policy = BCStgGuided(bc_ckpt, None, seed=1)   # 순수 BC — STG 평가로 후보 고르지 않음

  step_diffs = []
  n_false_success = 0
  n_env_success = 0
  n_missed = 0
  outcomes = {}
  sample_episode = None   # 성공 에피소드 하나의 d_trace(sanity check용)

  for e in range(episodes):
    info, d_trace, pred_step = _diagnostic_rollout(
        policy, env, reward, seed0 + e, max_steps)
    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    if info['outcome'] == 'success':
      n_env_success += 1
      if pred_step is None:
        n_missed += 1
      else:
        step_diffs.append(pred_step - info['steps'])
      if sample_episode is None:
        sample_episode = d_trace
    else:
      if pred_step is not None:
        n_false_success += 1

  n_fail_episodes = episodes - n_env_success
  false_success_rate = (n_false_success / n_fail_episodes
                        if n_fail_episodes > 0 else float('nan'))
  miss_rate = n_missed / n_env_success if n_env_success > 0 else float('nan')

  return dict(
      episodes=episodes, outcomes=outcomes,
      n_env_success=n_env_success, n_fail_episodes=n_fail_episodes,
      false_success_rate=false_success_rate, miss_rate=miss_rate,
      step_diff_mean=float(np.mean(step_diffs)) if step_diffs else float('nan'),
      step_diff_median=float(np.median(step_diffs)) if step_diffs else float('nan'),
      n_step_diff_samples=len(step_diffs),
      sample_episode_d_trace=sample_episode,
  )


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ckpt', required=True,
                  help='checkpoints/grasp_carry_dstg_succ|fail/predictor.pkl')
  ap.add_argument('--statistic', choices=['mean', 'cvar'], default='mean')
  ap.add_argument('--cvar-alpha', type=float, default=0.8)
  ap.add_argument('--fail-value', type=float, default=None)
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--bc-ckpt', default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--episodes', type=int, default=100)
  ap.add_argument('--seed0', type=int, default=900000)
  args = ap.parse_args()

  reward = StgReward(args.ckpt, statistic=args.statistic,
                     cvar_alpha=args.cvar_alpha, fail_value=args.fail_value)
  print(f'체크포인트: {args.ckpt}')
  print(f'  fail_bin={reward.fail_bin}  num_bins={reward.num_bins}  '
        f'statistic={args.statistic}'
        + (f'  cvar_alpha={args.cvar_alpha}' if args.statistic == 'cvar' else ''))

  # ---- 1. 문턱 캘리브레이션 -------------------------------------------------
  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  val_eps = _val_episode_ids(data, seed=reward.meta['seed'])
  best_s, metrics = calibrate_threshold(reward, data, val_eps)
  reward.threshold = best_s
  print(f'\n[1] 문턱 캘리브레이션 (held-out {len(val_eps)}개 에피소드, '
        f'seed={reward.meta["seed"]})')
  print(f'  s={metrics["best_s"]:.3f}  precision={metrics["precision"]:.3f}  '
        f'recall={metrics["recall"]:.3f}  f1={metrics["f1"]:.3f}')

  # ---- 2. 학습된 판정기 vs 환경 ground truth --------------------------------
  cfg_max_steps = 200
  ev = _run_rollout_eval(reward, args.bc_ckpt, args.episodes, args.seed0,
                         cfg_max_steps)
  print(f'\n[2] 판정기 vs 환경 ground truth ({args.episodes} 에피소드, '
        f'순수 BC 정책, seed0={args.seed0})')
  print(f'  outcomes={ev["outcomes"]}')
  print(f'  env 성공 에피소드={ev["n_env_success"]}  '
        f'env 실패 에피소드={ev["n_fail_episodes"]}')
  print(f'  거짓 성공률(실패 에피소드 중 도중에 성공이라 말한 비율)='
        f'{ev["false_success_rate"]:.1%}')
  print(f'  누락률(성공 에피소드 중 끝까지 성공이라 안 말한 비율)='
        f'{ev["miss_rate"]:.1%}')
  print(f'  조기 오판 스텝차(판정 스텝 - 실제 성공 스텝, 음수=조기 오판): '
        f'평균={ev["step_diff_mean"]:.2f}  중앙값={ev["step_diff_median"]:.2f}  '
        f'(n={ev["n_step_diff_samples"]})')

  # ---- 3. d 프로파일 sanity check ------------------------------------------
  print('\n[3] d 프로파일 sanity check (성공 에피소드 1개, 10스텝 간격)')
  trace = ev['sample_episode_d_trace']
  if trace is None:
    print('  (episodes 안에 성공 에피소드가 없어 확인 불가)')
  else:
    shown = trace[::10]
    print('  ' + '  '.join(f't={i*10}:{v:.1f}' for i, v in enumerate(shown)))
    diffs = np.diff(trace)
    frac_decreasing = float(np.mean(diffs <= 0))
    monotone = frac_decreasing >= 0.7
    print(f'  스텝간 비증가 비율={frac_decreasing:.1%}  '
          f'대체로 단조감소={monotone}'
          + ('' if monotone else '  [경고: d가 단조 감소하지 않음 — 보상이 노이즈일 수 있음]'))


if __name__ == '__main__':
  main()
