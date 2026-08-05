r"""원논문(Ghasemipour et al. 2025, Self-Improving Embodied Foundation Models)의
Stage-2 Self-Improvement 알고리즘을 우리 스칼라 액터에 그대로 적용한다.

## 지금까지 만든 것과 뭐가 다른가

`train_carry_actor.py`는 두 가지가 논문과 달랐다:
  1. 보상이 `P(성공) - speed_penalty*예측스텝`(임의로 뺀 것)이었다.
     논문 식(2)은 `r_t = d(o_t,g) - d(o_{t+1},g)` — "이 행동으로 기대
     스텝수가 얼마나 줄었나"뿐이다. 단위가 같은(스텝) 두 값을 빼는 거라
     `speed_penalty` 같은 임의의 저울추가 필요 없다.
  2. 실제 환경 롤아웃 없이 저장된 상태 배치 + 얼려진 critic에 역전파했다.
     논문 Algorithm 1은 **on-policy REINFORCE**다 — 지금 정책으로 실제로
     굴려서 궤적을 모으고, Monte Carlo 리턴을 계산해서 정책 그래디언트로
     업데이트한다.

이 스크립트는 두 가지 다 논문대로 고친다. `d(o,g)`는 관측만 보고 기대
스텝수를 내는 함수여야 하므로(논문 식(1), 액션은 안 들어간다), 액션-조건부
qstg critic(AI-E/AI-R) 대신 **관측-only STG 헤드**
(`checkpoints/grasp_carry_diff100`가 이미 갖고 있다 —
`train_carry_predictor.py`에서 디퓨전 액션 헤드와 같이 학습한 그 STG
헤드)를 그대로 재사용한다. 논문처럼 이 체크포인트를 통째로 얼려서
"보상 계산 + 성공 판정 전용"으로 쓴다.

## 우리 세팅에 맞춘 단순화 — 왜 그랬는지

논문의 정책은 RT-2식 이산 토큰 정책이라 `log p(a_t|o_t,g)`가 소프트맥스
로그확률로 바로 나온다. 우리 디퓨전 정책은 연속값이라 정확한 로그확률이
없다(역확산 전체를 적분해야 함 — 비용이 크다). 그래서 REINFORCE 대상을
디퓨전 정책 전체가 아니라, 로그확률이 깔끔한 **우리 스칼라 액터**(파지
직후 속도 하나만 고르는 지점, `speed_selector`)로 좁혔다.

액터가 에피소드당 1~2번(재파지 있으면 2번)만 결정을 내리므로, 논문의
"매 스텝 dense reward"를 우리 상황에 맞게 이렇게 옮겼다: **각 결정 시점의
보상 = "그 결정 시점의 d" − "다음 결정 시점(또는 에피소드 끝)의 d"**.
이건 논문 식(2)를 그대로 텔레스코핑한 것과 같다 — 재파지가 있으면
두 구간으로 나뉘어 각 결정이 자기 구간의 개선분만큼만 credit을 받는다.

## 액터를 확률적으로 만든 이유

REINFORCE는 `log p(action)`이 있어야 한다. 지금까지 액터는 결정론적(같은
관측 -> 항상 같은 리드)이었다 — 확률이 없다. 그래서 액터가 "무제한
실수(z)"를 평균으로 하는 가우시안에서 z를 샘플하고, 그 z를 시그모이드로
[0.4, 200]mm에 눌러 담는 구조로 바꿨다. `log p(z)`는 그냥 가우시안
로그밀도이고(시그모이드는 z를 리드로 바꾸는 결정론적 변환일 뿐이라
z 자체의 분포에는 야코비안 보정이 필요 없다), 탐색(exploration)은
`--explore-std`로 z의 표준편차를 정해서 만든다.

    python train_carry_actor_reinforce.py \
        --diff-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
        --out checkpoints/grasp_carry_actor_reinforce/actor.pkl
"""

import argparse
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy

LEAD_MIN, LEAD_MAX = 0.4, 200.0
NOMINAL_SPEED = 150.0   # _lift_lead(연직 리드)를 정하는 값. 데이터 수집 때와 통일.


# ============================================================ d(o,g) — 보상원
def load_d_function(ckpt_path: str):
  """`checkpoints/grasp_carry_diff100`의 관측-only STG 헤드를 `d(o,g)`로 쓴다.

  논문 식(1): d(o,g) = E[steps-to-go | o, g]. 액션을 안 받는다 — 그래서
  qstg(액션-조건부) 대신 이 체크포인트를 쓴다. 이 함수가 반환하는
  `d_fn(params, obs_n)`은 정규화된 관측 배치를 받아 기대 스텝수를 낸다.
  논문처럼 이 파라미터는 통째로 얼려서 **정책 업데이트에 관여하지 않는다.**
  """
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  from src.diffusion_act import build_diffusion_act_chunk
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  nets = build_diffusion_act_chunk(
      (256, 256, 256), act_dim, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=1, act_dim=act_dim)
  bin_vals = jnp.arange(ck['dc_config']['num_bins'], dtype=jnp.float32)
  max_steps = ck['dc_config']['num_bins']

  def d_fn(obs_n_batch):
    logits = nets.dist_logits(ck['params'], obs_n_batch)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * bin_vals[None, :], axis=-1)   # (B,) 기대 스텝수

  return ck, d_fn, max_steps


# ============================================================== 확률적 액터
def build_stochastic_actor(obs_dim, layer_sizes=(64, 64)):
  """관측 -> 가우시안 평균 mu(무제한 실수). 표준편차는 학습 안 하고
  `--explore-std`로 고정한다(탐색량을 직접 통제하기 위해 — 논문의 이산
  정책은 소프트맥스 온도로 자연히 탐색하지만, 우리 스칼라 액터는 명시적
  으로 정해줘야 한다)."""
  def _net(x):
    h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                    activate_final=True)(x)
    return hk.Linear(1)(h)[..., 0]                        # mu(x), 무제한 실수
  tn = hk.without_apply_rng(hk.transform(_net))
  def init(rng):
    return tn.init(rng, jnp.zeros((2, obs_dim), jnp.float32))
  return tn.apply, init


def z_to_lead(z):
  """무제한 실수 z -> [LEAD_MIN, LEAD_MAX] 리드(mm). 결정론적 변환이라
  z의 확률(가우시안 로그밀도)에는 영향을 안 준다 — 야코비안 보정 불필요."""
  return LEAD_MIN + (LEAD_MAX - LEAD_MIN) * jax.nn.sigmoid(z)


def gaussian_log_prob(z, mu, std):
  return -0.5 * ((z - mu) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)


# ======================================================== 실제 롤아웃 1개
class StochasticSelector:
  """`ScriptedCarryPolicy`의 `speed_selector`로 꽂는다. 매 결정마다 확률적으로
  샘플하고, 그 결정 시점의 (관측, z, mu)를 기록해둔다 — 나중에 REINFORCE
  업데이트에 쓴다."""

  def __init__(self, apply_fn, params, stats, std, rng):
    self.apply_fn = apply_fn
    self.params = params
    self.stats = stats
    self.std = std
    self.rng = rng
    self.decisions = []   # [(obs_raw(60,), z, mu), ...] — 이 에피소드의 결정들

  def __call__(self, env, contact_len, arm):
    obs_raw = np.asarray(env._stacked_obs(), dtype=np.float32)
    obs_n = (obs_raw - self.stats['frame_mean']) / self.stats['frame_std']
    mu = float(self.apply_fn(self.params, jnp.asarray(obs_n[None]))[0])
    self.rng, sub = jax.random.split(self.rng)
    z = mu + self.std * float(jax.random.normal(sub, ()))
    self.decisions.append((obs_raw, z, mu))
    return float(np.clip(np.asarray(z_to_lead(z)), LEAD_MIN, LEAD_MAX))


def rollout_episode(env, cfg, apply_fn, params, actor_stats, std, rng,
                    d_fn, d_stats, seed):
  """에피소드 하나를 실제로 굴리고, 각 결정 구간의 (관측, z, mu, 텔레스코핑
  리턴)을 반환한다.

  텔레스코핑 리턴 = 논문 식(2)를 우리 결정 단위로 옮긴 것:
      이 결정 시점의 d(o,g) − 다음 결정 시점(또는 에피소드 끝)의 d(o,g)
  재파지가 있으면 결정이 2개고, 각 결정은 **자기 구간의 개선분만** 받는다
  (다음 결정 이후 벌어지는 일은 그 다음 결정의 책임이지 이 결정의 책임이
  아니다).
  """
  sel = StochasticSelector(apply_fn, params, actor_stats, std, rng)
  pol = ScriptedCarryPolicy(config=cfg, speed=NOMINAL_SPEED, speed_selector=sel)
  env.reset(seed=seed)
  pol.reset()
  obs_at_decisions = []   # 각 결정 "직후" 관측 스냅샷 인덱스를 못 미리 알므로
  final_obs = None
  info = env._info()
  for t in range(cfg.max_steps):
    obs, _, term, trunc, info = env.step(pol(env))
    final_obs = obs
    if term or trunc:
      break

  # d(o,g)는 배치로 한 번에 계산 — 결정 시점 관측들 + 마지막 관측.
  decisions = sel.decisions
  if not decisions:
    return []   # 파지 자체를 못 한 에피소드 — 이 액터가 관여 안 함
  obs_batch_raw = np.stack([d[0] for d in decisions] + [final_obs], axis=0)
  obs_batch_n = (obs_batch_raw - d_stats['frame_mean']) / d_stats['frame_std']
  d_vals = np.asarray(d_fn(jnp.asarray(obs_batch_n, dtype=jnp.float32)))
  # d_vals[i] = i번째 결정 시점의 d, d_vals[-1] = 에피소드 끝 시점의 d.
  # 실패(전도/타임아웃)면 최종 관측이 성공과 거리가 멀어 d가 크게 남는다 —
  # 논문처럼 별도 "실패 페널티" 상수 없이, d 자체가 자연히 그 정보를 담는다.

  out = []
  for i, (obs_raw, z, mu) in enumerate(decisions):
    R = float(d_vals[i] - d_vals[i + 1])   # 이 구간에서 줄어든 기대 스텝수
    out.append((obs_raw, z, mu, R))
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--diff-ckpt', default='checkpoints/grasp_carry_diff100/predictor.pkl',
                  help='d(o,g) 계산 + 초기 정책 통계에 쓸, 얼려질 Stage-1 체크포인트.')
  ap.add_argument('--iterations', type=int, default=30,
                  help='수집→업데이트 반복 횟수(논문 Algorithm 1의 while 루프).')
  ap.add_argument('--episodes-per-iter', type=int, default=64,
                  help='매 반복마다 실제로 굴릴 에피소드 수.')
  ap.add_argument('--explore-std', type=float, default=1.0,
                  help='z(시그모이드 전 잠재값)의 샘플링 표준편차 — 탐색량.')
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--reinforce-scale', type=float, default=1.0,
                  help='논문 Algorithm 1의 상수 c(REINFORCE 손실 크기 조절).')
  ap.add_argument('--seed0', type=int, default=0,
                  help='학습용 롤아웃 시드 시작점 — 900000+(평가용)과 안 겹치게.')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  # d(o,g) 계산기 — 통째로 얼림(논문: "freeze a separate Stage 1 checkpoint
  # for reward computation").
  d_ck, d_fn, max_steps = load_d_function(args.diff_ckpt)
  d_stats = d_ck['norm_stats']

  # 액터는 이 체크포인트의 관측 정규화 통계를 그대로 재사용(같은 관측
  # 공간이니 새로 계산할 이유가 없다).
  actor_apply, actor_init = build_stochastic_actor(d_ck['obs_dim'])
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  actor_params = actor_init(sub)
  optimizer = optax.adam(args.lr)
  opt_state = optimizer.init(actor_params)

  def reinforce_loss(p, obs_n_batch, z_batch):
    mu = actor_apply(p, obs_n_batch)
    logp = gaussian_log_prob(z_batch, mu, args.explore_std)
    return logp   # 배치별 log p(z|obs) — 바깥에서 R과 곱해 평균 낸다

  @jax.jit
  def update(params, opt_state, obs_n_batch, z_batch, R_batch):
    def loss_fn(p):
      logp = reinforce_loss(p, obs_n_batch, z_batch)
      # 논문 REINFORCE 손실: -c * R_t * log p(a_t|o_t,g). 배치 평균을 낸다.
      return -args.reinforce_scale * jnp.mean(R_batch * logp)
    loss, g = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  seed_counter = args.seed0
  for it in range(1, args.iterations + 1):
    # ---- 1) 지금 정책으로 실제 롤아웃 수집(리플레이 버퍼 역할) ----
    batch_obs, batch_z, batch_R = [], [], []
    n_succ = 0
    for _ in range(args.episodes_per_iter):
      key, sub = jax.random.split(key)
      decisions = rollout_episode(env, cfg, actor_apply, actor_params,
                                  d_stats, args.explore_std, sub,
                                  d_fn, d_stats, seed_counter)
      seed_counter += 1
      for obs_raw, z, mu, R in decisions:
        batch_obs.append(obs_raw); batch_z.append(z); batch_R.append(R)
      if env._info()['outcome'] == 'success':
        n_succ += 1

    if not batch_obs:
      print(f'iter {it}: 이 배치에서 결정이 하나도 안 나옴(파지 실패만 반복) — 건너뜀')
      continue

    obs_n_batch = jnp.asarray(
        (np.stack(batch_obs) - d_stats['frame_mean']) / d_stats['frame_std'],
        dtype=jnp.float32)
    z_batch = jnp.asarray(batch_z, dtype=jnp.float32)
    R_batch = jnp.asarray(batch_R, dtype=jnp.float32)

    # ---- 2) 정책 그래디언트 업데이트 ----
    actor_params, opt_state, loss = update(actor_params, opt_state,
                                           obs_n_batch, z_batch, R_batch)

    print(f'iter {it:3d}  결정수={len(batch_obs):4d}  '
          f'에피소드성공률={n_succ/args.episodes_per_iter:.1%}  '
          f'평균R(=이번 배치의 d 개선량)={float(R_batch.mean()):+.2f}  '
          f'loss={float(loss):.4f}', flush=True)

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump({'params': jax.device_get(actor_params), 'obs_dim': d_ck['obs_dim'],
                'norm_stats': {'frame_mean': d_stats['frame_mean'],
                              'frame_std': d_stats['frame_std']},
                'explore_std': args.explore_std,
                'meta': {'algorithm': 'paper-exact REINFORCE (Ghasemipour et al. 2025)',
                         'diff_ckpt': args.diff_ckpt, 'iterations': args.iterations,
                         'episodes_per_iter': args.episodes_per_iter,
                         'reinforce_scale': args.reinforce_scale}}, fp)
  print(f'저장: {args.out}')


if __name__ == '__main__':
  main()
