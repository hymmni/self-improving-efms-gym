r"""`src/train_carry_qstg.py` 체크포인트로 조건 A/B를 직접 재현.

`calibrate_carry.py`는 실제 롤아웃(ScriptedCarryPolicy를 여러 속도로 굴림)의
집계 통계로 조건 A/B를 봤다. 이 스크립트는 그 대신 **학습된 액션-조건부
예측기 하나**에, 같은 상태에서 "느리게"/"빠르게"에 해당하는 두 가상 액션을
넣어 예측된 P(성공)·E[스텝|성공]·q0.8[스텝|성공]을 직접 비교한다 — "모델이
이 위험 구조를 실제로 배웠는가"에 대한 답이다.

    python probe_carry_qstg.py --ckpt checkpoints/grasp_carry_qstg/predictor.pkl \
        --data data/grasp_carry_demos_v3.pkl

`run_probe()`는 `verify_carry_qstg_condb.py`가 재사용한다(조건 B로 걸린
상태들의 episode_id를 뽑아 실제 물리로 재검증하기 위함).
"""

import argparse
import pickle

import numpy as np
import jax.numpy as jnp

from grasp_carry.train_carry_qstg import (build_qstg_net, split_success_fail,
                                  succ_mean_quantile)
from grasp_carry.config import CarryConfig

# 명목/공격적 리드(mm) — calibrate_carry.py의 스윕에서 쓴 것과 같은 스케일.
SLOW_LEAD = 12.5
FAST_LEAD = 150.0
_MIN_EXECUTED_DX = 1.0   # 이보다 작으면 실행 액션의 좌우 방향이 애매해 제외


def load_ckpt(path):
  with open(path, 'rb') as fp:
    ck = pickle.load(fp)
  apply_fn, _ = build_qstg_net((256, 256, 256),
                               obs_act_dim=(len(ck['frame_fields'])
                                           * ck['obs_history'] + 4),
                               num_bins=ck['num_bins'])
  return ck, apply_fn


def run_probe(ckpt_path: str, data_path: str, val_only: bool = True) -> dict:
  """조건 A/B 판정 배열과, 그걸 만든 원 데이터의 `episode_id`를 함께 돌려준다."""
  ck, apply_fn = load_ckpt(ckpt_path)
  stats = ck['norm_stats']
  frame_fields = ck['frame_fields']
  obs_hist = ck['obs_history']
  fail_bin = ck['fail_bin']
  bin_vals = jnp.arange(ck['num_bins'], dtype=jnp.float32)

  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)

  cfg = CarryConfig()
  L = cfg.world_width
  ex_i = frame_fields.index('ee_x')
  last0 = (obs_hist - 1) * len(frame_fields)   # 스택된 관측 안 최신 프레임 시작

  # 학습 스크립트와 동일한 규칙으로 held-out 에피소드를 고른다.
  rng = np.random.default_rng(ck['meta']['seed'])
  ep_ids_all = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids_all, size=max(len(ep_ids_all) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps)) if val_only \
      else np.ones(len(data['episode_id']), dtype=bool)
  # 잡고 옮기는 중(carry/relocate) 상태만 — 결정 지점이 "속도를 고르는" 순간.
  mask = val_mask & data['is_held']
  idx = np.where(mask)[0]

  obs = data['observation']['frame'][idx]                 # (N, 60) 정규화 전
  act = data['action'][idx]                                # 실제 실행됐던 액션
  ep_id = data['episode_id'][idx]
  ex_mm = obs[:, last0 + ex_i] * L
  dx = act[:, 0] - ex_mm
  keep = np.abs(dx) >= _MIN_EXECUTED_DX
  obs, ex_mm, dx, ep_id = obs[keep], ex_mm[keep], dx[keep], ep_id[keep]
  target_y = act[keep, 1]
  sign = np.sign(dx)

  def make_action(lead):
    tx = ex_mm + sign * lead
    a = np.stack([tx, target_y, np.zeros_like(tx), np.ones_like(tx)], axis=-1)
    return (a - stats['act_mean']) / stats['act_std']

  obs_n = (obs - stats['frame_mean']) / stats['frame_std']
  x_slow = jnp.asarray(np.concatenate([obs_n, make_action(SLOW_LEAD)], axis=-1))
  x_fast = jnp.asarray(np.concatenate([obs_n, make_action(FAST_LEAD)], axis=-1))

  def query(x):
    logits = apply_fn(ck['params'], x)
    p_succ, succ_probs = split_success_fail(logits, fail_bin)
    mean, q80 = succ_mean_quantile(succ_probs, bin_vals[:fail_bin], q=0.8)
    return np.asarray(p_succ), np.asarray(mean), np.asarray(q80)

  ps_slow, mean_slow, q80_slow = query(x_slow)
  ps_fast, mean_fast, q80_fast = query(x_fast)

  cond_a = (mean_fast < mean_slow) & (ps_fast < ps_slow)
  cond_b = (mean_fast < mean_slow) & (q80_fast > q80_slow)

  return dict(episode_id=ep_id, n_candidates=len(idx), n_directed=keep.sum(),
              ps_slow=ps_slow, ps_fast=ps_fast, mean_slow=mean_slow,
              mean_fast=mean_fast, q80_slow=q80_slow, q80_fast=q80_fast,
              cond_a=cond_a, cond_b=cond_b, cond_ab=cond_a & cond_b)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ckpt', default='checkpoints/grasp_carry_qstg/predictor.pkl')
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--val-only', action='store_true', default=True)
  args = ap.parse_args()

  r = run_probe(args.ckpt, args.data, val_only=args.val_only)
  n = len(r['mean_slow'])
  print(f'후보 상태(검증셋 中 is_held): {r["n_candidates"]}개')
  print(f'좌우 방향이 뚜렷한 상태(|dx|>={_MIN_EXECUTED_DX}mm): {r["n_directed"]}개')
  print(f'\n예측기 기준 조건 A/B (n={n} 상태, 리드 {SLOW_LEAD}mm vs {FAST_LEAD}mm)')
  print(f'  평균 P(성공): 느림={r["ps_slow"].mean():.3f}  빠름={r["ps_fast"].mean():.3f}')
  print(f'  평균 E[스텝|성공]: 느림={r["mean_slow"].mean():.1f}  '
        f'빠름={r["mean_fast"].mean():.1f}')
  print(f'  평균 q0.8[스텝|성공]: 느림={r["q80_slow"].mean():.1f}  '
        f'빠름={r["q80_fast"].mean():.1f}')
  print(f'  조건 A(기댓값은 빠름 선호, 성공률은 느림 선호): {r["cond_a"].sum()}/{n} '
        f'({r["cond_a"].mean():.1%})')
  print(f'  조건 B(기댓값 vs q0.8 부호 반전): {r["cond_b"].sum()}/{n} '
        f'({r["cond_b"].mean():.1%})')
  print(f'  동시 만족: {r["cond_ab"].sum()}/{n} ({r["cond_ab"].mean():.1%})')


if __name__ == '__main__':
  main()
