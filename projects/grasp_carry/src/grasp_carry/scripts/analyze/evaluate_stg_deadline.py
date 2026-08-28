r"""fail-aware(deadline 모드) STG 예측기가 held-out 롤아웃 에피소드에서 실제로
실패를 판별하는지 평가한다 (사용자 질문 2026-08-09: "rollout 데이터로 STG
예측기 학습 시키면 실패 상황에만 리셋+평균 길이 밖으로 나가서 실패 판별
가능한지").

`src/carry_stg_reward.calibrate_threshold`는 "이 transition이 성공의 마지막
프레임인가"를 양성/음성으로 나눠 F1을 낸다 — 그 자체로도 유효한 지표지만,
지금 확인하려는 건 다른 질문이다: **"매 스텝 μ(o) > B(=reset_cost+T_hat)"를
포기(=이 에피소드는 실패로 간다) 판정으로 쓰면, 진짜 실패 에피소드에서만
이게 걸리고 성공 에피소드에서는 안 걸리는가?** — 즉 이 예측기가 renewal
theory 포기판정(context_8.md)의 실전 판별기로 쓸 만한지를 직접 잰다.

held-out(val) 에피소드는 예측기 학습 때와 동일한 분할(`_val_episode_ids`)을
재현해서 쓴다 — 학습에 쓴 transition에서 재면 낙관적으로 나온다.

실행:
    python evaluate_stg_deadline.py --ckpt checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl
"""

import argparse
import pickle

import numpy as np

from grasp_carry.carry_stg_reward import StgReward, calibrate_threshold, _val_episode_ids


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ckpt', default='checkpoints/grasp_carry_dstg_deadline_v5rollout/predictor.pkl')
  args = ap.parse_args()

  reward = StgReward(args.ckpt, statistic='mean')
  meta = reward.meta
  B = meta.get('deadline_B')
  data_path = meta['data']
  print(f'체크포인트: {args.ckpt}')
  print(f'학습 데이터: {data_path}')
  print(f'B = reset_cost + T_hat = {B}  (T_hat={meta.get("T_hat")}, '
        f'T_hat_std={meta.get("T_hat_std")}, reset_cost={meta.get("reset_cost")})')
  if B is None:
    raise SystemExit('이 체크포인트는 deadline 모드로 학습되지 않았다(meta에 deadline_B 없음).')

  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  val_eps = _val_episode_ids(data, seed=meta['seed'])
  mask = np.isin(data['episode_id'], list(val_eps))
  obs = data['observation']['frame'][mask]
  eid = data['episode_id'][mask]
  is_succ_step = np.asarray(data['is_success'])[mask]

  mu, _ = reward.mean_std(obs)

  # ---- (1) 파이썬 식(3) 스타일 F1: "이 transition이 성공의 마지막 프레임인가"
  best_s, m = calibrate_threshold(reward, data, val_eps)
  print(f'\n[1] 식(3) 스타일 성공-프레임 판별 F1: s={best_s:.1f}  '
        f'precision={m["precision"]:.3f}  recall={m["recall"]:.3f}  f1={m["f1"]:.3f}')

  # ---- (2) 에피소드 단위: "이 에피소드 안에서 mu(o)가 한 번이라도 B를 넘는가"
  #          그걸로 "이 에피소드는 실패로 간다"를 예측했다고 보고, 실제
  #          outcome(성공/실패)과 비교한다.
  print(f'\n[2] 에피소드 단위 "μ>B 한번이라도 발생 -> 실패 예측" 판별 (n={len(val_eps)}개 에피소드)')
  tp = fp_ = fn = tn = 0
  first_cross_step = {}   # 에피소드별: 처음 B를 넘은 스텝 인덱스(없으면 None)
  for e in val_eps:
    sel = eid == e
    mu_e = mu[sel]
    succ_e = bool(is_succ_step[sel][0]) if not is_succ_step[sel][-1] else bool(is_succ_step[sel][-1])
    # is_success는 에피소드 전체에 대해 상수(True면 전 스텝 True, False면 전 스텝 False)
    is_success_ep = bool(is_succ_step[sel].any())
    crossed = np.where(mu_e > B)[0]
    predicted_fail = len(crossed) > 0
    first_cross_step[int(e)] = int(crossed[0]) if len(crossed) else None
    actual_fail = not is_success_ep
    if predicted_fail and actual_fail:
      tp += 1
    elif predicted_fail and not actual_fail:
      fp_ += 1
    elif not predicted_fail and actual_fail:
      fn += 1
    else:
      tn += 1

  n = tp + fp_ + fn + tn
  precision = tp / (tp + fp_) if (tp + fp_) else float('nan')
  recall = tp / (tp + fn) if (tp + fn) else float('nan')
  f1 = (2 * precision * recall / (precision + recall)
        if (precision + recall) and not np.isnan(precision) and not np.isnan(recall) else float('nan'))
  accuracy = (tp + tn) / n if n else float('nan')
  print(f'  TP(실패인데 감지)={tp}  FP(성공인데 오탐)={fp_}  '
        f'FN(실패인데 놓침)={fn}  TN(성공인데 안울림)={tn}')
  print(f'  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  accuracy={accuracy:.3f}')

  fp_eps = [int(e) for e in val_eps if is_succ_step[eid == e].any() and first_cross_step[int(e)] is not None]
  if fp_eps:
    print(f'  오탐(성공인데 B 넘음) 에피소드: {fp_eps[:10]}')


if __name__ == '__main__':
  main()
