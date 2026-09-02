r"""디퓨전 BC 정책을 qstg critic으로 파인튜닝한다.

`train_carry_actor.py`와 같은 방식(얼려진 critic에 역전파)이지만, 대상이
"속도 하나"가 아니라 **사전학습된 디퓨전 정책의 액션 생성 자체**다.

디퓨전 샘플링(역확산)은 고정된 횟수만큼 신경망을 반복 호출하는 절차라,
노이즈 시드를 고정하면 `params -> 샘플된 액션`까지 전부 미분 가능하다
(`nets.sample_chunk`는 `jax.lax.scan`으로 구현돼 있어 그 자체로 미분
가능한 계산 그래프다). 그래서 디퓨전 정책의 최종 출력에 대고 바로
critic의 그래디언트를 역전파할 수 있다 — DDPG의 actor 업데이트를 actor가
디퓨전 정책인 경우로 확장한 것.

**BC 정규화 필수**: critic 보상만 최대화하면 정책이 원래 배운 그럴듯한
동작에서 멀어지며 무너질 수 있다(보상 해킹). 원본(고정) 정책이 같은
노이즈로 냈을 액션과의 L2 거리를 벌점으로 같이 준다.

    python -m grasp_carry.scripts.train.finetune_carry_diffusion \
        --diff-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
        --qstg-ckpt checkpoints/grasp_carry_qstg/predictor.pkl \
        --out checkpoints/grasp_carry_diff100_finetuned/predictor.pkl
"""

import argparse
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax

from grasp_carry.scripts.analyze.probe_carry_qstg import load_ckpt as load_qstg_ckpt
from grasp_carry.train_carry_qstg import split_success_fail, succ_mean_quantile
from grasp_carry.diffusion_act import build_diffusion_act_chunk


def load_diffusion_policy(ckpt_path):
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  nets = build_diffusion_act_chunk(
      (256, 256, 256), act_dim, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=1, act_dim=act_dim)
  return ck, nets


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--qstg-ckpt', default='checkpoints/grasp_carry_qstg/predictor.pkl')
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--steps', type=int, default=1000)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--lr', type=float, default=1e-5,
                  help='사전학습 lr(3e-4)보다 훨씬 작게 — 파인튜닝이라 BC로 '
                       '배운 것을 급격히 무너뜨리면 안 된다.')
  ap.add_argument('--speed-penalty', type=float, default=0.1)
  ap.add_argument('--bc-reg', type=float, default=1.0,
                  help='원본(고정) 정책의 같은-노이즈 액션과의 L2 벌점 가중치 '
                       '— 이게 없으면 critic 보상만 좇다 정책이 무너질 수 있다.')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  dck, dnets = load_diffusion_policy(args.diff_ckpt)
  qck, qapply = load_qstg_ckpt(args.qstg_ckpt)
  dstats = dck['norm_stats']
  qstats = qck['norm_stats']
  fail_bin = qck['fail_bin']
  bin_vals = jnp.arange(qck['num_bins'], dtype=jnp.float32)

  # is_held 필터 — qstg critic이 실제로 학습한 상태 분포와 맞춘다.
  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  obs_raw = data['observation']['frame'][data['is_held']]
  n = len(obs_raw)
  print(f'파인튜닝용 상태 {n}개 (is_held)')

  obs_d_n = jnp.asarray((obs_raw - dstats['frame_mean']) / dstats['frame_std'],
                        dtype=jnp.float32)
  obs_q_n = jnp.asarray((obs_raw - qstats['frame_mean']) / qstats['frame_std'],
                        dtype=jnp.float32)

  frozen_params = dck['params']    # BC 정규화 기준(고정, 업데이트 안 함)
  params = dck['params']           # 파인튜닝 대상

  optimizer = optax.adam(args.lr)
  opt_state = optimizer.init(params)

  def to_qstg_action(act_d_n):
    act_mm = act_d_n * dstats['act_std'] + dstats['act_mean']
    return (act_mm - qstats['act_mean']) / qstats['act_std']

  def loss_fn(p, idx, key):
    ob_d, ob_q = obs_d_n[idx], obs_q_n[idx]
    key, sub = jax.random.split(key)
    act_n = dnets.sample_chunk(p, ob_d, sub)                 # 미분 가능
    x = jnp.concatenate([ob_q, to_qstg_action(act_n)], axis=-1)
    logits = qapply(qck['params'], x)                        # 얼려진 critic
    p_succ, succ_probs = split_success_fail(logits, fail_bin)
    mean, _ = succ_mean_quantile(succ_probs, bin_vals[:fail_bin])
    reward = p_succ - args.speed_penalty * (mean / qck['max_steps'])

    key, sub2 = jax.random.split(key)
    act_n_orig = jax.lax.stop_gradient(dnets.sample_chunk(frozen_params, ob_d, sub2))
    bc_reg = jnp.mean(jnp.sum((act_n - act_n_orig) ** 2, axis=-1))

    loss = -jnp.mean(reward) + args.bc_reg * bc_reg
    return loss, (jnp.mean(reward), bc_reg)

  @jax.jit
  def train_step(params, opt_state, key):
    key, sub, sub2 = jax.random.split(key, 3)
    idx = jax.random.randint(sub, (args.batch,), 0, n)
    (loss, (reward, bc_reg)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        params, idx, sub2)
    updates, opt_state = optimizer.update(g, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, reward, bc_reg

  key = jax.random.PRNGKey(args.seed)
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    params, opt_state, loss, reward, bc_reg = train_step(params, opt_state, sub)
    if step % 100 == 0 or step == 1:
      print(f'step {step:5d}  loss={float(loss):.4f}  '
            f'평균보상={float(reward):.4f}  bc_reg={float(bc_reg):.4f}', flush=True)

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  new_ck = dict(dck)
  new_ck['params'] = jax.device_get(params)
  new_ck['meta'] = dict(dck['meta'])
  new_ck['meta'].update(finetuned_with=args.qstg_ckpt, finetune_steps=args.steps,
                        finetune_lr=args.lr, bc_reg=args.bc_reg,
                        speed_penalty=args.speed_penalty)
  with open(args.out, 'wb') as fp:
    pickle.dump(new_ck, fp)
  print(f'저장: {args.out}')


if __name__ == '__main__':
  main()
