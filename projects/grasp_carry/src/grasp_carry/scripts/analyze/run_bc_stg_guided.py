r"""디퓨전 BC 정책 + STG 예측기 후보 선별 — 손으로 짠 후보 없이 끝단까지 학습됨.

`ScriptedCarryPolicy`(사람이 짠 상태기계)를 아예 쓰지 않는다. 대신:

  1. 매 제어 스텝, 학습된 디퓨전 BC 정책(`checkpoints/grasp_carry_diff100`)에서
     같은 관측에 대해 노이즈 시드만 다르게 K개 액션 후보를 뽑는다(디퓨전
     샘플링 자체가 확률적이라 "그럴듯한" 후보가 자연히 갈린다 — 사람이 속도
     격자를 정한 게 아니다).
  2. 각 후보를 액션-조건부 STG 예측기(AI-E 또는 AI-R)에 넣어 성공확률·기대
     스텝을 얻는다.
  3. 공유 규칙으로 하나를 고른다: "예측 성공확률이 문턱 이상인 후보 중
     기대 스텝이 가장 짧은 것 — 문턱을 넘는 게 없으면 성공확률이 제일 높은
     것."(`compare_carry_selectors.py`와 동일한 규칙, 후보가 스칼라 속도가
     아니라 임의 액션이라 "빠른 것"을 "기대 스텝이 짧은 것"으로 일반화했다.)
  4. 고른 액션 하나를 실행하고 다음 스텝으로.

AI-E(성공 데모만 학습)와 AI-R(실패 포함 학습)을 완전히 같은 규칙·같은 BC
정책에 꽂아 비교한다. `--no-guidance`면 STG 평가 없이 디퓨전 정책이 뽑은
첫 후보를 그냥 쓴다(순수 BC 베이스라인).

    python -m grasp_carry.scripts.analyze.run_bc_stg_guided --episodes 200
"""

import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from grasp_carry.scripts.analyze.probe_carry_qstg import load_ckpt as load_qstg_ckpt
from grasp_carry.diffusion_act import build_diffusion_act_chunk
from grasp_carry.train_carry_qstg import split_success_fail, succ_mean_quantile
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D

K_CANDIDATES = 8


def load_bc_policy(ckpt_path: str):
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  nets = build_diffusion_act_chunk(
      (256, 256, 256), act_dim, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=1, act_dim=act_dim)
  sample = jax.jit(lambda p, o, k: nets.sample_chunk(p, o, k))
  return ck, sample


class BCStgGuided:
  """`env`를 받아 액션을 내는 콜러블 — `ScriptedCarryPolicy`와 같은 자리에서
  쓰지만 상태기계가 아니라 (BC 후보 생성 + STG 평가)만으로 끝단까지 학습됨."""

  def __init__(self, bc_ckpt: str, qstg_ckpt: str | None, k=K_CANDIDATES,
               success_threshold=0.95, seed=0):
    self.bc_ck, self.bc_sample = load_bc_policy(bc_ckpt)
    self.qstg_ck, self.qstg_apply = (load_qstg_ckpt(qstg_ckpt)
                                     if qstg_ckpt else (None, None))
    self.k = k
    self.success_threshold = success_threshold
    self._key = jax.random.PRNGKey(seed)
    if self.qstg_ck is not None:
      self.fail_bin = self.qstg_ck['fail_bin']
      self.bin_vals = jnp.arange(self.qstg_ck['num_bins'], dtype=jnp.float32)
    # 진단용
    self.p_succ_at_pick = []

  def reset(self):
    pass

  def __call__(self, env) -> np.ndarray:
    stats = self.bc_ck['norm_stats']
    obs = np.asarray(env._stacked_obs(), dtype=np.float32)[None]
    obs_n = (obs - stats['frame_mean']) / stats['frame_std']
    obs_batch = jnp.asarray(np.repeat(obs_n, self.k, axis=0))
    self._key, sub = jax.random.split(self._key)
    cand_n = np.asarray(self.bc_sample(self.bc_ck['params'], obs_batch, sub))
    cand = cand_n * stats['act_std'] + stats['act_mean']   # (K, 4) 실좌표

    if self.qstg_ck is None:
      return cand[0].astype(np.float32)          # 순수 BC — 후보 중 아무거나

    qstats = self.qstg_ck['norm_stats']
    cand_qn = (cand - qstats['act_mean']) / qstats['act_std']
    x = jnp.asarray(np.concatenate([np.repeat(
        (obs - qstats['frame_mean']) / qstats['frame_std'], self.k, axis=0),
        cand_qn], axis=-1))
    logits = self.qstg_apply(self.qstg_ck['params'], x)
    p_succ, succ_probs = split_success_fail(logits, self.fail_bin)
    mean, _ = succ_mean_quantile(succ_probs, self.bin_vals[:self.fail_bin])
    p_succ = np.asarray(p_succ); mean = np.asarray(mean)

    ok = p_succ >= self.success_threshold
    if ok.any():
      idx = np.where(ok)[0][np.argmin(mean[ok])]    # 안전권 중 제일 빠름
    else:
      idx = int(np.argmax(p_succ))                   # 다 불안 -> 그나마 나은 것
    self.p_succ_at_pick.append(float(p_succ[idx]))
    return cand[idx].astype(np.float32)


def rollout(policy, env, seed, max_steps):
  env.reset(seed=seed)
  policy.reset()
  info = env._info()
  for _ in range(max_steps):
    _, _, term, trunc, info = env.step(policy(env))
    if term or trunc:
      break
  return info


def run(policy, episodes, seed0, cfg):
  env = GraspCarry2D(cfg)
  n_succ, succ_steps, total_steps, outcomes = 0, [], 0, {}
  episode_log = []          # (outcome, steps) — 리셋 비용을 사후에 다시 계산하기 위해 보존
  for e in range(episodes):
    info = rollout(policy, env, seed0 + e, cfg.max_steps)
    outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
    total_steps += info['steps']
    episode_log.append((info['outcome'], info['steps']))
    if info['outcome'] == 'success':
      n_succ += 1
      succ_steps.append(info['steps'])
  return dict(episodes=episodes, n_succ=n_succ, outcomes=outcomes,
              success_rate=n_succ / episodes, total_steps=total_steps,
              mean_succ_steps=float(np.mean(succ_steps)) if succ_steps else float('nan'),
              demos_per_1k_steps=1000.0 * n_succ / max(total_steps, 1),
              episode_log=episode_log)


def demos_per_1k_with_reset(r: dict, reset_cost: dict) -> float:
  """`reset_cost`: outcome -> 리셋에 드는 추가 스텝 비용(가정치).

  실제 로봇에서는 리셋이 공짜가 아니다 — 특히 전도(물체가 넘어짐)는 사람이
  다시 세워줘야 해서 타임아웃(정책이 그냥 시간 초과)보다 훨씬 비싸다. 기존
  `demos_per_1k_steps`는 이 비용을 전부 0으로 뒀다(실패해도 즉시 공짜로
  리셋된다고 가정) — 그래서 "빨리 실패하는 것"이 지표상 유리해지는 착시가
  있었다.
  """
  cost = sum(steps + reset_cost.get(outcome, 0.0)
            for outcome, steps in r['episode_log'])
  return 1000.0 * r['n_succ'] / max(cost, 1.0)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--seed0', type=int, default=900000)
  ap.add_argument('--bc-ckpt', default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--exp-ckpt', default='checkpoints/grasp_carry_qstg_exponly/predictor.pkl')
  ap.add_argument('--risk-ckpt', default='checkpoints/grasp_carry_qstg/predictor.pkl')
  ap.add_argument('--k', type=int, default=K_CANDIDATES)
  ap.add_argument('--success-threshold', type=float, default=0.95)
  ap.add_argument('--reset-cost-success', type=float, default=20.0,
                  help='성공 후 다음 에피소드 준비 비용(스텝 환산 가정치).')
  ap.add_argument('--reset-cost-timeout', type=float, default=20.0,
                  help='타임아웃 후 리셋 비용 — 정책이 교착했을 뿐 물리적 '
                       '개입은 필요 없다고 가정.')
  ap.add_argument('--reset-cost-tipped', type=float, default=200.0,
                  help='전도 후 리셋 비용 — 넘어진 물체를 사람이 다시 세워야 '
                       '한다고 보고 타임아웃의 10배로 가정(임의값, 조정 가능).')
  args = ap.parse_args()
  reset_cost = {'success': args.reset_cost_success,
                'timeout': args.reset_cost_timeout,
                'tipped': args.reset_cost_tipped}

  cfg = CarryConfig()
  arms = [
      ('순수 BC (STG 평가 없음)',
       BCStgGuided(args.bc_ckpt, None, k=args.k, seed=1)),
      ('BC + 기댓값 전용 STG',
       BCStgGuided(args.bc_ckpt, args.exp_ckpt, k=args.k,
                   success_threshold=args.success_threshold, seed=2)),
      ('BC + 위험 인지 STG',
       BCStgGuided(args.bc_ckpt, args.risk_ckpt, k=args.k,
                   success_threshold=args.success_threshold, seed=3)),
  ]

  rows = []
  for name, pol in arms:
    r = run(pol, args.episodes, args.seed0, cfg)
    rows.append((name, pol, r))
    print(f'{name}: 완료', flush=True)

  print(f'\n{"팔":<24} {"성공률":>7} {"성공시 스텝":>10} '
        f'{"총 스텝":>9} {"1k스텝당 데모":>12} {"리셋비용 반영":>13}')
  for name, pol, r in rows:
    with_reset = demos_per_1k_with_reset(r, reset_cost)
    print(f'{name:<24} {r["success_rate"]:>7.1%} {r["mean_succ_steps"]:>10.1f} '
          f'{r["total_steps"]:>9d} {r["demos_per_1k_steps"]:>12.2f} '
          f'{with_reset:>13.2f}')
  print(f'\n(리셋 비용 가정: 성공={reset_cost["success"]:g} '
        f'타임아웃={reset_cost["timeout"]:g} '
        f'전도={reset_cost["tipped"]:g} 스텝, --reset-cost-*로 조정 가능)')
  print()
  for name, pol, r in rows:
    print(f'  {name}: outcomes={r["outcomes"]}')
    if pol.p_succ_at_pick:
      ps = np.asarray(pol.p_succ_at_pick)
      print(f'    선택 시점 예측 성공확률: 평균={ps.mean():.3f} 최소={ps.min():.3f}')


if __name__ == '__main__':
  main()
