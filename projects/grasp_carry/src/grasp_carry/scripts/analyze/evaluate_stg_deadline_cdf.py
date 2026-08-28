r"""μ(평균) 하나만 쓰는 "μ>B면 포기" 규칙(precision=1.0, recall=0.364,
`evaluate_stg_deadline.py`)에 σ(분산)까지 같이 쓰면 재현율을 올릴 수 있는지
검증한다 (사용자 질문 2026-08-09).

세 규칙을 같은 held-out 60개 에피소드에서 비교한다:
  A. mean-only   : 에피소드 중 한 스텝이라도 μ(o) > B
  B. CDF(tail)   : 에피소드 중 한 스텝이라도 P(steps-to-go > B | o) > eps
                   (카테고리컬 분포를 그대로 적분 — μ,σ를 굳이 분리 안 해도
                   분포 모양 전체를 쓰는 셈이라 σ 정보가 자동으로 들어간다)
  C. mean+sigma  : 에피소드 중 한 스텝이라도 μ(o) + k*σ(o) > B (정규근사로
                   CDF 규칙을 근사한 명시적 μ,σ 결합형 — B가 규칙 A로는 뭘
                   놓쳤는지 σ 항이 메꾸는지 직접 보기 위함)

A가 이미 recall 0.364에서 잡은 4건은 B, C도 당연히 잡을 것(둘 다 A보다 약한
조건이 아니라 다른 정보를 추가로 쓰므로 정확히 상위집합은 아니지만 대체로
겹친다) — 관심사는 **A가 놓친 7건(FN) 중 몇 건을 B/C가 추가로 잡는지, 그리고
그 대가로 오탐(성공인데 포기판정)이 몇 건 새로 생기는지**다.

실행:
    python evaluate_stg_deadline_cdf.py --ckpt checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl
"""

import argparse
import pickle

import numpy as np

from grasp_carry.carry_stg_reward import StgReward, _val_episode_ids


def episode_rule_eval(eid, val_eps, is_succ_step, signal, thresh_grid, mode_name):
  """signal: 매 transition별 스칼라(예: mu, tail_prob, mu+k*sigma).
  각 threshold 후보에 대해 '에피소드 중 signal이 그 문턱을 한 번이라도 넘으면
  실패 예측'으로 episode-level precision/recall/f1을 계산, 최댓값을 고른다.
  단, 사용자가 원하는 건 "정밀도 1.0을 유지한 채 재현율을 올릴 수 있는가"이므로
  precision==1.0인 후보들 중 recall이 최대인 것도 같이 report한다."""
  results = []
  for th in thresh_grid:
    tp = fp_ = fn = tn = 0
    for e in val_eps:
      sel = eid == e
      is_success_ep = bool(is_succ_step[sel].any())
      crossed = bool((signal[sel] > th).any())
      actual_fail = not is_success_ep
      if crossed and actual_fail: tp += 1
      elif crossed and not actual_fail: fp_ += 1
      elif not crossed and actual_fail: fn += 1
      else: tn += 1
    precision = tp / (tp + fp_) if (tp + fp_) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision) and not np.isnan(recall) else 0.0)
    results.append(dict(th=float(th), tp=tp, fp=fp_, fn=fn, tn=tn,
                        precision=precision, recall=recall, f1=f1 if not np.isnan(f1) else 0.0))

  best_f1 = max(results, key=lambda r: r['f1'])
  perfect_precision = [r for r in results if r['precision'] == 1.0]
  best_at_p1 = max(perfect_precision, key=lambda r: r['recall']) if perfect_precision else None

  print(f'\n=== {mode_name} ===')
  print(f'  best-F1 지점     : th={best_f1["th"]:.2f}  '
        f'P={best_f1["precision"]:.3f} R={best_f1["recall"]:.3f} F1={best_f1["f1"]:.3f}  '
        f'(TP={best_f1["tp"]} FP={best_f1["fp"]} FN={best_f1["fn"]} TN={best_f1["tn"]})')
  if best_at_p1:
    print(f'  precision=1.0 유지, recall 최대: th={best_at_p1["th"]:.2f}  '
          f'R={best_at_p1["recall"]:.3f}  '
          f'(TP={best_at_p1["tp"]} FP={best_at_p1["fp"]} FN={best_at_p1["fn"]} TN={best_at_p1["tn"]})')
  else:
    print('  precision=1.0을 유지하는 문턱 없음(항상 오탐 발생)')
  return best_f1, best_at_p1


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ckpt', default='checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl')
  args = ap.parse_args()

  reward = StgReward(args.ckpt, statistic='mean')
  meta = reward.meta
  B = meta['deadline_B']
  data_path = meta['data']
  print(f'B = {B:.1f}')

  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  val_eps = _val_episode_ids(data, seed=meta['seed'])
  mask = np.isin(data['episode_id'], list(val_eps))
  obs = data['observation']['frame'][mask]
  eid = data['episode_id'][mask]
  is_succ_step = np.asarray(data['is_success'])[mask]

  mu, sigma = reward.mean_std(obs)
  probs = reward._probs(obs)
  bin_vals = np.asarray(reward._bin_vals)
  tail_mask = bin_vals > B
  tail_prob = probs[:, tail_mask].sum(axis=-1)

  n_fail_eps = sum(1 for e in val_eps if not is_succ_step[eid == e].any())
  n_succ_eps = len(val_eps) - n_fail_eps
  print(f'held-out {len(val_eps)}개 에피소드 (성공 {n_succ_eps}, 실패 {n_fail_eps})')

  # A. mean-only (B=evaluate_stg_deadline.py와 동일 규칙, th=B 고정이지만
  #    여기선 여러 th를 훑어 A 규칙 자체의 최대 잠재력도 같이 본다)
  mu_grid = np.linspace(mu.min(), mu.max(), 200)
  episode_rule_eval(eid, val_eps, is_succ_step, mu, mu_grid, 'A. mean-only (mu > th)')

  # B. CDF(tail) — 분포 전체를 적분(=평균,분산뿐 아니라 꼬리 모양까지 반영)
  tail_grid = np.linspace(0.0, max(tail_prob.max(), 1e-6), 200)
  episode_rule_eval(eid, val_eps, is_succ_step, tail_prob, tail_grid, 'B. CDF tail-prob (P(steps>B) > eps)')

  # C. mean + k*sigma (명시적 mu,sigma 결합, k 몇 개 스캔)
  for k in (0.5, 1.0, 1.5, 2.0):
    signal = mu + k * sigma
    sig_grid = np.linspace(signal.min(), signal.max(), 200)
    episode_rule_eval(eid, val_eps, is_succ_step, signal, sig_grid, f'C. mu + {k}*sigma > th')


if __name__ == '__main__':
  main()
