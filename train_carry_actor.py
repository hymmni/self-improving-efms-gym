r"""AI-E/AI-R를 보상으로 삼아 "속도를 고르는" 정책을 진짜로 학습시킨다.

이전 두 시도와 다른 점: 후보를 여러 개 뽑아서 고르는 게 아니라(best-of-K
필터), 작은 신경망 하나(actor)를 만들어 **가중치를 실제로 업데이트**한다.
`src/train_carry_qstg.py`가 학습한 예측기(AI-E 또는 AI-R)는 미분 가능한
신경망이므로, "이 예측기가 좋다고 말하는 방향"으로 actor를 역전파로 직접
최적화한다 — DDPG류 actor-critic의 actor 업데이트와 같은 방식이다. 진짜
환경 롤아웃 없이(예측기만 쿼리해서) 학습하고, **학습이 끝난 뒤에만** 실제
시뮬레이터로 성능을 채점한다 — "정답을 매번 알 수 없어 학습된 보상 모델에
의존해야 하는 상황"을 흉내낸다.

    python train_carry_actor.py --qstg-ckpt checkpoints/grasp_carry_qstg/predictor.pkl \
        --out checkpoints/grasp_carry_actor_risk/actor.pkl
"""

import argparse
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from probe_carry_qstg import load_ckpt as load_qstg_ckpt
from src.train_carry_qstg import split_success_fail, succ_mean_quantile, succ_cvar
from src.grasp_carry.config import CarryConfig

LEAD_MIN, LEAD_MAX = 0.4, 200.0
_MIN_EXECUTED_DX = 1.0


def build_actor(obs_dim, layer_sizes=(64, 64)):
  def _net(x):
    h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                    activate_final=True)(x)
    raw = hk.Linear(1)(h)[..., 0]
    return LEAD_MIN + (LEAD_MAX - LEAD_MIN) * jax.nn.sigmoid(raw)
  tn = hk.without_apply_rng(hk.transform(_net))
  def init(rng):
    return tn.init(rng, jnp.zeros((2, obs_dim), jnp.float32))
  return tn.apply, init


def gather_states(data_path, cfg):
  """v3 데이터에서 "잡고 옮기는 중" 상태와, 액션 구성에 필요한 (ex_mm, sign,
  target_y)를 뽑는다. `probe_carry_qstg.py`와 같은 필터."""
  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  from src.grasp_carry.env import FRAME_FIELDS
  ex_i = FRAME_FIELDS.index('ee_x')
  obs_hist = 4
  last0 = (obs_hist - 1) * len(FRAME_FIELDS)
  L = cfg.world_width

  mask = data['is_held']
  idx = np.where(mask)[0]
  obs = data['observation']['frame'][idx]
  act = data['action'][idx]
  ex_mm = obs[:, last0 + ex_i] * L
  dx = act[:, 0] - ex_mm
  keep = np.abs(dx) >= _MIN_EXECUTED_DX
  return dict(obs=obs[keep], ex_mm=ex_mm[keep], sign=np.sign(dx[keep]),
              target_y=act[keep, 1])


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--qstg-ckpt', required=True)
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--steps', type=int, default=3000)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--speed-penalty', type=float, default=0.1,
                  help='보상 = P(성공) - 이 값 * (예측 스텝 통계/max_steps). '
                       '순수 P(성공)만 쓰면 동점일 때 속도 무관심해진다.')
  ap.add_argument('--objective', choices=['mean', 'cvar', 'renewal'], default='mean',
                  help=('mean=평균(재파지처럼 꼬리만 줄이는 행동의 가치가 '
                        '다수의 좋은 경우에 묻힘). cvar=최악 (1-cvar-alpha) '
                        '구간만 재정규화한 기댓값. renewal=renewal-reward '
                        '정리 형태 P(성공)/E[사이클 시간] — --speed-penalty '
                        '없이 --fail-cost 하나로 안전-속도 저울을 표현한다 '
                        '(demos_per_1k_with_reset과 같은 양을 직접 최적화).'))
  ap.add_argument('--cvar-alpha', type=float, default=0.8,
                  help='cvar일 때: 하위 이 비율은 무시하고 위쪽 꼬리만 본다.')
  ap.add_argument('--fail-cost', type=float, default=200.0,
                  help=('renewal일 때: 실패 사이클 하나의 비용(스텝 환산). '
                        'eval_carry_actor.py의 전도 리셋 비용(200) 가정과 '
                        '맞췄다 — 예측기의 실패 bin은 실패까지 실제로 몇 '
                        '스텝 걸렸는지 구분 안 하므로 상수로 근사한다.'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  cfg = CarryConfig()
  qck, qapply = load_qstg_ckpt(args.qstg_ckpt)
  qstats = qck['norm_stats']
  fail_bin = qck['fail_bin']
  bin_vals = jnp.arange(qck['num_bins'], dtype=jnp.float32)

  st = gather_states(args.data, cfg)
  n = len(st['obs'])
  print(f'학습용 상태 {n}개 (v3의 is_held, 방향 뚜렷한 것)')

  obs_n_all = ((st['obs'] - qstats['frame_mean']) / qstats['frame_std']).astype(np.float32)
  ex_mm_all = jnp.asarray(st['ex_mm'], dtype=jnp.float32)
  sign_all = jnp.asarray(st['sign'], dtype=jnp.float32)
  ty_all = jnp.asarray(st['target_y'], dtype=jnp.float32)
  obs_n_all = jnp.asarray(obs_n_all)

  OBS_DIM = obs_n_all.shape[-1]
  apply_fn, init_fn = build_actor(OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = init_fn(sub)
  optimizer = optax.adam(args.lr)
  opt_state = optimizer.init(params)

  act_mean = jnp.asarray(qstats['act_mean']); act_std = jnp.asarray(qstats['act_std'])
  fr_mean = jnp.asarray(qstats['frame_mean']); fr_std = jnp.asarray(qstats['frame_std'])

  def reward_fn(p, idx):
    obs_n = obs_n_all[idx]
    lead = apply_fn(p, obs_n)                              # (B,) 미분 가능
    tx = ex_mm_all[idx] + sign_all[idx] * lead
    ty = ty_all[idx]
    act_mm = jnp.stack([tx, ty, jnp.zeros_like(tx), jnp.ones_like(tx)], axis=-1)
    act_n = (act_mm - act_mean) / act_std
    x = jnp.concatenate([obs_n, act_n], axis=-1)
    logits = qapply(qck['params'], x)                       # 얼려진 critic
    p_succ, succ_probs = split_success_fail(logits, fail_bin)
    if args.objective == 'renewal':
      # renewal-reward 정리: 장기 평균 보상률 = E[사이클당 보상]/E[사이클당 시간].
      # 여기서 사이클=에피소드, 보상=성공 여부(0/1), 시간=성공까지 스텝 또는
      # 실패 비용. λ를 손으로 정하는 대신, "성공 하나당 실제로 드는 기대 비용"
      # 으로 자연스럽게 안전-속도 저울이 생긴다.
      mean, _ = succ_mean_quantile(succ_probs, bin_vals[:fail_bin])
      cycle_cost = p_succ * mean + (1.0 - p_succ) * args.fail_cost
      return p_succ / jnp.maximum(cycle_cost, 1e-6), lead
    if args.objective == 'cvar':
      step_stat = succ_cvar(succ_probs, bin_vals[:fail_bin], alpha=args.cvar_alpha)
    else:
      step_stat, _ = succ_mean_quantile(succ_probs, bin_vals[:fail_bin])
    return p_succ - args.speed_penalty * (step_stat / qck['max_steps']), lead

  def loss_fn(p, idx):
    r, lead = reward_fn(p, idx)
    return -jnp.mean(r), (jnp.mean(r), jnp.mean(lead))

  @jax.jit
  def train_step(params, opt_state, key):
    idx = jax.random.randint(key, (args.batch,), 0, n)
    (loss, (r, lead)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, idx)
    updates, opt_state = optimizer.update(g, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, r, lead

  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    params, opt_state, loss, r, lead = train_step(params, opt_state, sub)
    if step % 300 == 0 or step == 1:
      print(f'step {step:5d}  loss={float(loss):.4f}  '
            f'평균보상={float(r):.4f}  평균선택리드={float(lead):.1f}mm', flush=True)

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump({'params': jax.device_get(params), 'obs_dim': OBS_DIM,
                'norm_stats': {'frame_mean': qstats['frame_mean'],
                              'frame_std': qstats['frame_std']},
                'meta': {'qstg_ckpt': args.qstg_ckpt, 'steps': args.steps,
                         'speed_penalty': args.speed_penalty,
                         'objective': args.objective,
                         'cvar_alpha': args.cvar_alpha,
                         'fail_cost': args.fail_cost}}, fp)
  print(f'저장: {args.out}')


if __name__ == '__main__':
  main()
