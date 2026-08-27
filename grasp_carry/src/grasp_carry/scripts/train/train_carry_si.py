r"""SI-EFM Stage-2(Algorithm 1) — DDPO-SF로 파지·운반 디퓨전 정책을 자기개선한다.

배경(코드가 아니라 여기 남기는 이유는 이 스크립트의 존재 이유 자체가
알고리즘 선택의 근거이기 때문이다) — 논문 Algorithm 1은:

  1. Stage-1 체크포인트를 하나 복사해 얼린다(보상/성공 판정 전용).
  2. 현재 정책으로 on-policy 롤아웃을 모은다.
  3. r_t = d(o_t,g) - d(o_{t+1},g)  (식 2)
  4. R_t = sum_{i>=t} gamma^(i-t) r_i
  5. REINFORCE: loss = -c * mean(R_t * log p(a_t|o_t,g))
  6. 그 배치는 한 번 쓰고 버린다(off-policy·부트스트래핑을 피하는 설계).

논문 정책은 이산 토큰이라 log p(a|o)가 바로 나오지만, 이 레포의 정책은 연속
디퓨전(`src/diffusion_act.py`)이라 닫힌 형태가 없다. `src/ddpo.py`(phase 4
step 0)가 역확산 100단계 각각을 가우시안 전이로 보고 단계별 로그확률의 합으로
대체한다(DDPO, Black et al. 2023) — 이 알고리즘이 DDPO-SF(score function,
바닐라 REINFORCE)인 이유는 데이터 재사용이 없어 논문의 on-policy 설계와
정확히 맞기 때문이다.

보상/성공 판정은 `src/carry_stg_reward.py`(phase 4 step 2)가 감싼 얼려진
관측-only STG 예측기(`src/train_carry_dstg.py`, phase 4 step 1)에서 나온다.
환경의 outcome/terminated/truncated는 어디에도 학습 신호로 들어가지 않는다
(로그에만 쓴다) — 논문 핵심 주장("외부 감독 없는 자기개선")이 걸려 있다.

실행:
  python train_carry_si.py \
      --policy-ckpt checkpoints/grasp_carry_diff100/predictor.pkl \
      --d-ckpt checkpoints/grasp_carry_dstg_succ/predictor.pkl \
      --statistic mean \
      --iterations 100 --episodes-per-iter 32 \
      --gamma 0.9 --reinforce-scale 5e-2 --lr 1e-5 \
      --termination learned \
      --out checkpoints/grasp_carry_si_mean_succ/predictor.pkl
"""

import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from grasp_carry.diffusion_act import build_diffusion_act_chunk
from grasp_carry.ddpo import build_ddpo
from grasp_carry.carry_stg_reward import StgReward, calibrate_threshold, _val_episode_ids
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D

# From: run_bc_stg_guided.py load_bc_policy() — 대상 체크포인트가 학습된 고정 구조.
LAYER_SIZES = (256, 256, 256)
DRIFT_N_OBS = 256
DRIFT_SEED = 12345
DRIFT_SAMPLE_SEED = 999_999


# ---------------------------------------------------------------- 순수 함수부
# (테스트가 직접 부르는 부분 — env/정책 없이 검증 가능해야 한다)

def compute_step_rewards(d_vals: np.ndarray) -> np.ndarray:
  """식(2): r_t = d(o_t,g) - d(o_{t+1},g).

  d_vals: (T+1,) 한 에피소드의 d(o_0..o_T). 반환 (T,).

  주의: 인자로 받는 것은 d값뿐이다 — env의 outcome/terminated/success는
  여기 어디에도 없다. 논문의 핵심 주장("외부 감독 없는 자기개선")이 이
  경계에 걸려 있으므로, 이 함수의 시그니처 자체가 그 경계다.
  """
  d_vals = np.asarray(d_vals, dtype=np.float64)
  return (d_vals[:-1] - d_vals[1:]).astype(np.float32)


def sanitize_terminal_d(d_vals: np.ndarray, outcome: str, reset_cost: float,
                        mean_episode_len: float, near_success_thresh: float = 10.0,
                        lookback: int = 5) -> np.ndarray:
  """`compute_step_rewards`에 넣기 **전에** d_vals를 정제한다 — 그 함수 자체의
  "d값만 받는다"는 경계는 그대로 둔 채, 호출부에서 알고 있는 env 결과(termination=env가
  이미 쓰고 있는 바로 그 신호)로 종단 근처의 신뢰 못할 예측값만 덮어쓴다.

  2026-08-11 실측(2D): 실패 에피소드의 24.5%가 **마지막 스텝**에서 mu<10(거의 확실히
  성공한다고 예측)로 나왔다 — succ-only로 학습된 예측기가 실패 종단 상태를 한 번도 못
  봐서, 종단 근처 상태를 "성공에 가깝다"로 낙관 오판하는 것으로 보인다. 이 상태로 두면
  `r_t = d(o_t)-d(o_{t+1})` 보상이 실패 직전 스텝에 큰 양수를 잘못 준다(리워드 해킹
  경로).

  2026-08-12: 첫 구현은 속은 프레임들을 max_steps(가장 나쁜 값, 200)로 개별
  덮어써서 실측 DDPO에서 3/4 조건이 학습 붕괴했다 — r_t=d(o_t)-d(o_{t+1})가 한
  프레임에서 -190 근처까지 튀어(정상 보상 크기 ±5~15의 10배 이상) REINFORCE
  그래디언트를 배치 하나가 지배해버린 것(사용자가 직접 진단). 두 가지를 고쳤다:
  1) 목표값을 max_steps가 아니라 `reset_cost + mean_episode_len`(현실적인 "실패
     비용" — 다시 시도하는 데 드는 여분 스텝 + 평균 한 번의 시도 길이)로 낮춘다.
  2) 속은 프레임 각각을 독립적으로 덮어쓰지 않고, lookback 구간 전체를 그 앞의
     신뢰할 수 있는 값에서 목표값까지 **선형보간**해서 매끄럽게 이어지게 한다 —
     한 스텝짜리 급격한 점프가 아니라 lookback개 스텝에 걸쳐 변화가 분산된다.
  """
  if outcome == 'success':
    return d_vals
  d_vals = np.asarray(d_vals, dtype=np.float64).copy()
  n = min(lookback, len(d_vals))                 # 에피소드가 lookback보다 짧을 수 있다
  tail = d_vals[-n:]
  fooled = tail < near_success_thresh
  if not fooled.any():
    return d_vals
  target = reset_cost + mean_episode_len
  start_val = float(d_vals[-n - 1]) if len(d_vals) > n else float(d_vals[0])
  ramp = np.linspace(start_val, target, n + 1)[1:]
  d_vals[-n:] = ramp
  return d_vals


def truncate_before_false_success(d_vals: np.ndarray, outcome: str,
                                  near_success_thresh: float = 10.0, lookback: int = 5) -> int:
  """실패 에피소드의 마지막 lookback 프레임 중 mu가 near_success_thresh 밑으로 떨어진
  **첫 지점부터는 신뢰할 수 없다**(허위 근접성공) — 그 값을 다른 걸로 재라벨링하는 대신
  그 지점 **이전까지만** 학습(보상 계산)에 쓰도록 자른다.

  2026-08-12: `sanitize_terminal_d`(값을 재라벨링)는 즉시 덮어쓰기 버전도 선형보간
  버전도 실측 DDPO에서 반복적으로 학습을 붕괴시켰다(사용자 진단) — 재라벨링은 "얼마로
  바꿀지"를 새로 정해야 하고, 그 값이 원래(허위로 낮은) 값과 차이가 크면 그 차이 자체가
  또 다른 보상 크기 문제를 만든다(보간으로 스텝당 점프는 줄여도 누적 리턴에 실리는 총
  크기는 못 줄임). 이 함수는 값을 발명하지 않는다 — 허위로 낮아진 구간 자체를 애초에
  return 계산에서 빼서, 그 구간이 만드는 거짓 양의 보상 스파이크가 계산되지 않게 한다.

  반환: 사용할 d_vals 길이 n(즉 d_vals[:n]을 쓴다) — 자를 게 없으면 len(d_vals) 그대로.
  """
  if outcome == 'success':
    return len(d_vals)
  n_lookback = min(lookback, len(d_vals))
  tail_start = len(d_vals) - n_lookback
  tail = d_vals[tail_start:]
  fooled = np.where(tail < near_success_thresh)[0]
  if len(fooled) == 0:
    return len(d_vals)
  return tail_start + int(fooled[0])


def compute_sigma_shaping(sigma_vals: np.ndarray, gamma: float) -> np.ndarray:
  """sigma(예측기 불확실성) 감소량을 PBRS 스타일 가산 보조보상으로 만든다.

  Phi_sigma(o) := -sigma(o), 종단 potential은 0으로 고정한다(Ng et al. 1999
  정책-불변 조건 — compute_step_rewards_v2의 phi[-1]=0과 동일 원칙, 예측기가
  실제 종단 상태를 못 봐서 낼 수 있는 값을 신뢰하지 않는다는 뜻).

  2026-08-11: --variance-weight(역분산 곱셈 가중치)는 sigma가 실패 에피소드
  전체에 걸쳐 높게 유지된다는 걸 사후에 확인해서 "실패로부터 배우는 신호"까지
  광범위하게 깎아 역효과였다. 이 함수는 곱셈이 아니라 **가산** 항이라 1차
  보상(compute_step_rewards/_v2)은 그대로 두고, "sigma가 줄어드는 방향(=더
  확신하는/익숙한 상태로 수렴)"에만 별도 보너스를 얹는다 — 성공 에피소드는
  sigma가 목표에 가까워질수록 꾸준히 감소하고 실패 에피소드는 끝까지 높게
  유지된다는 동일한 실측 근거가, 여기서는 오히려 유리하게 작용할 수 있다는
  가설(사용자 제안, 검증 전).
  """
  sigma_vals = np.asarray(sigma_vals, dtype=np.float64)
  phi = -sigma_vals.copy()
  phi[-1] = 0.0
  return (gamma * phi[1:] - phi[:-1]).astype(np.float32)


def compute_step_rewards_v2(d_vals: np.ndarray, outcome: str, step_cost: float,
                            success_bonus: float, gamma: float, kappa: float,
                            plain_shaping: bool = False) -> np.ndarray:
  """r_t = -step_cost + success_bonus*1{t가 성공으로 끝나는 마지막 스텝} + shaping_t,
  Phi(o) := -d(o,g).

  references/context_9.md의 R-learning식 rho_hat 자기보정 대신, 이 태스크(에피소드+
  할인 구조)에 더 자연스러운 고정 스텝비용으로 바꾼 버전(2026-08-10) — 할인(gamma<1)이
  이미 "무한 누적 방지" 역할을 하므로 R-learning의 rho_hat(원래 무할인·비에피소드
  환경용)은 중복이었다는 진단에 따른 변경.

  shaping_t의 두 형태(2026-08-10, 사용자 질문에서 파생):
    plain_shaping=False(기본): kappa*(gamma*Phi(o_{t+1}) - Phi(o_t)) — 엄밀한 PBRS.
      원래 식 r_t=d(o_t)-d(o_{t+1})을 대수적으로 풀면 이 항 + 암묵적 "기본보상"
      -(1-gamma)*d(o_{t+1},g)로 정확히 분해된다(논문 식(5)와 동치) — 이 모드는 그
      암묵적 기본보상을 위 step_cost/success_bonus로 완전히 교체한 것이다.
    plain_shaping=True: kappa*(d(o_t,g)-d(o_{t+1},g)) — 원래 식을 그대로 shaping
      자리에 넣은 것. 대수적으로는 위 PBRS 항 + (1-gamma)*Phi(o_{t+1})까지 포함하므로,
      원래의 암묵적 기본보상이 (1-gamma)배만큼 안 없어지고 새 기본보상과 같이 섞인다
      (완전한 교체가 아니라 절충 — 구현이 더 단순하다는 장점과 맞바꾼 것).
  Phi(o_T)는 두 모드 다 0으로 고정한다(Ng et al. 1999 PBRS 정책-불변 조건 — 종단상태에서
  예측기가 못 본 값을 낼 수 있어 필요. plain_shaping 모드에서는 d(o_T)=0 강제와 동치).

  이 함수는 outcome(env의 성공/실패 결과)을 인자로 받는다 — GraspCarry2D는 성공하는
  순간 바로 에피소드가 끝나는 구조라(성공=마지막 스텝), "몇 번째 스텝에서 성공했는지"를
  따로 안 받아도 마지막 스텝 하나에만 success_bonus를 주면 된다.
  """
  d_vals = np.asarray(d_vals, dtype=np.float64)
  T = len(d_vals) - 1
  phi = -d_vals
  phi = phi.copy()
  phi[-1] = 0.0                                     # 종단상태 potential 강제 고정
  if plain_shaping:
    shaping = kappa * (phi[1:] - phi[:-1])
  else:
    shaping = kappa * (gamma * phi[1:] - phi[:-1])  # (T,)
  core = np.zeros(T, dtype=np.float64)
  if outcome == 'success':
    core[-1] = success_bonus
  r = -step_cost + core + shaping
  return r.astype(np.float32)


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
  """R_t = sum_{i>=t} gamma^(i-t) * r_i, 역순 누적으로 계산. rewards: (T,) -> (T,)."""
  rewards = np.asarray(rewards, dtype=np.float64)
  T = rewards.shape[0]
  returns = np.zeros(T, dtype=np.float64)
  acc = 0.0
  for t in range(T - 1, -1, -1):
    acc = rewards[t] + gamma * acc
    returns[t] = acc
  return returns.astype(np.float32)


def make_pair_index_table(n_steps: int):
  """`xs`(n_steps+1, ...) 안에서 학습에 쓰는 (i_in, i_out, kk) 대응표.

  `src/ddpo.py`의 `chain_logp` scan body(kk = n_steps-1-i, i=0..n_steps-2)와
  정확히 같은 규칙: xs[i] --(kk=n_steps-1-i)--> xs[i+1]. kk=0(i=n_steps-1)은
  결정론적 전이라 제외 — 그래서 표의 길이는 n_steps-1이다.

  반환: (i_in, i_out, kk) 각각 (n_steps-1,) int 배열.
  """
  i_in = np.arange(n_steps - 1)
  i_out = i_in + 1
  kk = (n_steps - 1) - i_in
  return i_in, i_out, kk


# ----------------------------------------------------------------- 체크포인트

def load_policy_for_training(ckpt_path: str):
  """`run_bc_stg_guided.load_bc_policy`와 같은 방식으로 nets를 복원하고,
  DDPO 함수를 그 위에 얹는다."""
  with open(ckpt_path, 'rb') as fp:
    ck = pickle.load(fp)
  m = ck['meta']
  act_dim = len(ck['norm_stats']['act_mean'])
  # 2026-08-11: --horizon>1(액션 청킹) 체크포인트 지원. 청킹 안 쓰는 체크포인트는
  # meta에 horizon이 없으므로 기본값 1(기존 동작과 동일)로 되돌아간다.
  horizon = int(m.get('horizon', 1))
  exec_horizon = int(m.get('exec_horizon', horizon))
  nets = build_diffusion_act_chunk(
      LAYER_SIZES, act_dim * horizon, ck['dc_config']['num_bins'], ck['obs_dim'],
      n_diffusion_steps=m['diffusion_steps'], backbone=m['backbone'],
      horizon=horizon, act_dim=act_dim)
  ddpo = build_ddpo(nets, act_dim * horizon)
  return ck, nets, ddpo, act_dim, horizon, exec_horizon


def _dist_head_keys(params) -> list:
  """From: tests/test_ddpo.py _is_dist_head_key — STG(dist) 헤드는 haiku가
  생성 순서상 가장 큰 접미사를 붙인 mlp_2/*, linear_2 뿐이다."""
  return [k for k in params.keys()
          if k.startswith('mlp_2') or k == 'linear_2']


# --------------------------------------------------------------------- 롤아웃

def collect_episode(nets, ddpo, params, reward: StgReward, env: GraspCarry2D,
                    seed: int, key, act_mean, act_std, frame_mean, frame_std,
                    termination: str, max_steps: int, sample_fn,
                    act_dim: int = None, horizon: int = 1, exec_horizon: int = 1):
  """정책 롤아웃 1개를 수집한다. 그래디언트 없음(inference만).

  --horizon>1(액션 청킹)이면 한 번의 sample_fn 호출(=하나의 결정 시점)마다
  exec_horizon개의 raw env step을 실행한다(receding horizon) — obs_n_all/xs_all은
  raw step이 아니라 **결정 시점** 단위로 하나씩 쌓인다. d_vals/보상도 자연히
  결정 시점 간격으로 계산되므로(한 결정=exec_horizon raw step만큼의 진행), 예측기
  라벨(raw step 카운트다운)과 단위가 안 맞는 문제는 없다 — d()가 매기는 값 자체는
  항상 raw step 기준이고, 두 결정 시점 사이의 실제 raw 진행량이 그 차이에 그대로
  반영될 뿐이다.

  반환: obs_n_all (T, obs_dim) — 각 결정 시점에서 정책이 실제로 조건으로 쓴 정규화
  관측, xs_all (T, n_steps+1, chunk_flat_dim), d_vals (T+1,), outcome(str),
  episode_len(T).
  """
  if act_dim is None:
    act_dim = len(act_mean)
  env.reset(seed=seed)
  obs_raw_list = [np.asarray(env._stacked_obs(), dtype=np.float32)]
  obs_n_list = []
  xs_list = []
  outcome = 'running'
  step_count = 0

  while step_count < max_steps:
    obs_raw_t = obs_raw_list[-1]
    obs_n_t = (obs_raw_t - frame_mean) / frame_std
    obs_n_list.append(obs_n_t)

    key, sub = jax.random.split(key)
    x_final, xs = sample_fn(params, jnp.asarray(obs_n_t[None]), sub)
    chunk_n = np.asarray(x_final)[0].reshape(horizon, act_dim)
    chunk = chunk_n * act_std + act_mean
    xs_list.append(np.asarray(xs)[:, 0, :])          # (n_steps+1, chunk_flat_dim)

    stop = False
    obs_raw_next = obs_raw_t
    for h in range(min(exec_horizon, horizon)):
      if step_count >= max_steps:
        stop = True
        break
      _, _, term, trunc, info = env.step(chunk[h])
      step_count += 1
      obs_raw_next = np.asarray(env._stacked_obs(), dtype=np.float32)
      outcome = info['outcome']
      if termination == 'learned':
        pred_succ = bool(reward.success(obs_raw_next[None])[0])
        stop = pred_succ or term or trunc
      else:  # 'env'
        stop = term or trunc
      if stop:
        break
    obs_raw_list.append(obs_raw_next)
    if stop:
      break

  obs_raw_all = np.stack(obs_raw_list)                # (T+1, obs_dim)
  d_vals = reward.d(obs_raw_all)                       # (T+1,) — 보상은 항상 d로만
  # 2026-08-11: mean_std()는 self.statistic(mean/cvar)과 무관하게 항상 원 분포의
  # (mu, sigma)를 낸다 — --variance-weight가 켜졌을 때만 쓰는 sigma는 reward
  # 통계량 선택과 독립적으로 항상 이걸로 잰다.
  _, sigma_vals = reward.mean_std(obs_raw_all)          # (T+1,)
  obs_n_all = np.stack(obs_n_list)                      # (T, obs_dim)
  xs_all = np.stack(xs_list)                            # (T, n_steps+1, chunk_flat_dim)
  return obs_n_all, xs_all, d_vals, sigma_vals, outcome, key


# --------------------------------------------------------------------- 학습

def build_train_step(ddpo, dist_head_keys):
  def loss_fn(params, obs_b, x_in_b, x_out_b, kk_b, R_b, reinforce_scale):
    lp = ddpo.step_logp(params, obs_b, x_in_b, x_out_b, kk_b)      # (B,)
    return -reinforce_scale * jnp.mean(R_b * lp)

  grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=())

  def train_step(params, opt_state, optimizer, obs_b, x_in_b, x_out_b, kk_b,
                R_b, reinforce_scale):
    loss, grads = grad_fn(params, obs_b, x_in_b, x_out_b, kk_b, R_b,
                          reinforce_scale)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads

  return train_step


def _assert_dist_head_frozen(params_before, params_after, dist_head_keys):
  for k in dist_head_keys:
    for pname, arr_after in params_after[k].items():
      arr_before = params_before[k][pname]
      if not np.array_equal(np.asarray(arr_before), np.asarray(arr_after)):
        raise RuntimeError(
            f'STG(dist) 헤드 파라미터 {k}/{pname}가 DDPO 업데이트로 바뀌었다 — '
            f'chain_logp/step_logp 그래디언트가 dist 헤드로 새고 있다는 뜻이다. '
            f'src/ddpo.py의 헤드 분리 로직(_is_dist_head_key)이 이 체크포인트의 '
            f'파라미터 키 구조와 어긋났을 가능성이 높다. 즉시 중단.')


# --------------------------------------------------------------------- 진단

def _load_drift_obs(data_path, frame_mean, frame_std):
  with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
  frames = data['observation']['frame']
  rng = np.random.default_rng(DRIFT_SEED)
  idx = rng.choice(len(frames), size=min(DRIFT_N_OBS, len(frames)),
                   replace=False)
  obs_raw = frames[idx].astype(np.float32)
  obs_n = (obs_raw - frame_mean) / frame_std
  return jnp.asarray(obs_n)


def _drift_metric(sample_fn, params, base_params, drift_obs_n):
  key = jax.random.PRNGKey(DRIFT_SAMPLE_SEED)
  x_cur, _ = sample_fn(params, drift_obs_n, key)
  x_base, _ = sample_fn(base_params, drift_obs_n, key)
  return float(jnp.mean(jnp.linalg.norm(x_cur - x_base, axis=-1)))


# ------------------------------------------------------------------------ main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--policy-ckpt',
                  default='checkpoints/grasp_carry_diff100/predictor.pkl')
  ap.add_argument('--d-ckpt', required=True)
  ap.add_argument('--statistic', choices=['mean', 'cvar'], default='mean')
  ap.add_argument('--cvar-alpha', type=float, default=0.8)
  ap.add_argument('--iterations', type=int, default=100)
  ap.add_argument('--episodes-per-iter', type=int, default=32)
  ap.add_argument('--gamma', type=float, default=0.9)
  ap.add_argument('--reinforce-scale', type=float, default=5e-2)
  ap.add_argument('--lr', type=float, default=1e-5)
  ap.add_argument('--termination', choices=['learned', 'env'], default='learned')
  ap.add_argument('--advantage-norm', action='store_true', default=False)
  ap.add_argument('--variance-weight', action='store_true', default=False,
                  help=('결정 시점마다 예측기의 sigma(불확실성) 역수를 리턴에 곱한다 '
                        '(에피소드 내 평균 가중치=1로 정규화). 2026-08-11: 예측기가 '
                        '확신 없는(=낯선/OOD) 상태에서 나온 학습 신호를 깎아 iteration간 '
                        '진동을 줄일 수 있는지 시험. 실측 결과 실패 에피소드의 sigma가 '
                        '처음부터 끝까지 높게 유지되는 것으로 나타나(성공 신호와 달리 '
                        '실패 신호는 특정 순간에만 튀는 게 아님), 이 옵션은 "실패로부터 '
                        '배우는 신호"를 광범위하게 죽여 오히려 역효과였다 — 기본 off 권장.'))
  ap.add_argument('--sigma-shaping-coef', type=float, default=0.0,
                  help=('sigma(예측기 불확실성) 감소량을 PBRS 스타일 가산 보조보상으로 '
                        '더한다: Phi_sigma=-sigma, 종단 potential 0 고정(정책-불변, '
                        'compute_sigma_shaping 참고). --variance-weight(곱셈 가중치)와 '
                        '달리 1차 보상은 그대로 두고 별도 항으로 얹는다. 0(기본)이면 비활성.'))
  ap.add_argument('--sanitize-terminal', action='store_true', default=False,
                  help=('--reward-v2 미사용(compute_step_rewards) 경로 전용. 실패로 끝난 '
                        '에피소드의 마지막 --sanitize-lookback개 프레임 중 예측기 mu가 '
                        '--near-success-thresh 밑으로 떨어지면, 그 lookback 구간 전체를 '
                        '앞선 신뢰 가능한 값에서 (--reset-cost + 평균 에피소드 길이)까지 '
                        '선형보간해서 덮어쓴다. 2026-08-11 실측: succ-only로 학습된 예측기가 '
                        '실패 종단 상태를 한 번도 못 봐서, 실패 에피소드의 24.5%가 마지막 '
                        '스텝에서 mu<10(거의 성공처럼) 오판 — 그대로 두면 실패 직전 스텝에 '
                        '큰 양의 보상이 잘못 들어간다(리워드 해킹 경로). 2026-08-12: 첫 '
                        '구현(max_steps로 개별 프레임 덮어쓰기)은 r_t가 한 스텝에서 -190 '
                        '근처까지 튀어 REINFORCE가 4개 조건 중 3개에서 학습 붕괴했다(사용자 '
                        '진단) — 목표값을 현실적인 값으로 낮추고 급격한 점프 대신 선형보간으로 '
                        '바꿔 재구현.'))
  ap.add_argument('--near-success-thresh', type=float, default=10.0,
                  help='--sanitize-terminal/--truncate-terminal 공용. 이 값 밑이면 '
                       '"허위 근접성공"으로 본다.')
  ap.add_argument('--sanitize-lookback', type=int, default=5,
                  help='--sanitize-terminal/--truncate-terminal 공용. 에피소드 끝에서 '
                       '몇 프레임까지 검사할지.')
  ap.add_argument('--truncate-terminal', action='store_true', default=False,
                  help=('--reward-v2 미사용 경로 전용. --sanitize-terminal의 대안 —'
                        '재라벨링 대신, 실패 에피소드의 마지막 --sanitize-lookback개 '
                        '프레임 중 mu가 --near-success-thresh 밑으로 떨어진 첫 지점 '
                        '**이전까지만** 잘라서 학습에 쓴다(그 지점부터의 관측/액션/보상은 '
                        '전부 버림). 2026-08-12: sanitize-terminal(재라벨링, 즉시/보간 '
                        '둘 다)은 계속 붕괴해서, "잘못된 값을 무엇으로 바꿀지" 자체를 '
                        '고민하는 대신 그 구간을 아예 안 쓰는 쪽으로 바꿈(사용자 제안) — '
                        '거짓 양의 보상 스파이크가 애초에 계산되지 않는다. '
                        '--sanitize-terminal과 동시에 켜면 이쪽이 우선한다.'))
  ap.add_argument('--reset-cost', type=float, default=20.0,
                  help=('--sanitize-terminal 전용. 실패 종단 목표값 = reset_cost + 평균 '
                        '에피소드 길이(calib_data 성공 에피소드 기준 자동 계산). 재시도에 '
                        '드는 여분 스텝을 나타내는 값.'))
  ap.add_argument('--success-only', action='store_true', default=False,
                  help=('실패 에피소드를 REINFORCE 업데이트에서 빼고 성공 에피소드만 쓴다'
                        '(self-imitation류 — 논문 원안의 전체-분포 REINFORCE가 아니다). '
                        '2026-08-10: 전체-분포 버전이 이 태스크에서 계속 초반에 정점을 찍고'
                        '(예: it=13) 이후 성공 0%에서 못 벗어나는 패턴을 반복 관찰해서, 대안으로'
                        '실험. 배치에 성공이 하나도 없으면 그 iteration은 업데이트를 건너뛴다.'))
  ap.add_argument('--logp-batch', type=int, default=4096)
  ap.add_argument('--seed0', type=int, default=0)
  ap.add_argument('--reward-v2', action='store_true', default=False,
                  help=('references/context_9.md 설계에서 R-learning식 rho_hat 자기보정을 '
                        '고정 스텝비용으로 바꾼 버전을 쓴다: r_t = -step_cost + '
                        'success_bonus*1{성공} + kappa*(gamma*Phi(o_{t+1})-Phi(o_t)). '
                        '2026-08-10: rho_hat은 원래 무할인(discount 없는) 환경용 기법인데 '
                        '우리는 이미 gamma<1로 할인하는 에피소드 구조라 중복이었다는 진단에 '
                        '따른 대안. 기본은 꺼져 있어 기존 compute_step_rewards(순수 PBRS)를 씀.'))
  ap.add_argument('--step-cost', type=float, default=1.0,
                  help='--reward-v2 전용. 매 스텝 고정 비용(빠른 완료를 유도).')
  ap.add_argument('--success-bonus', type=float, default=50.0,
                  help='--reward-v2 전용. 성공하는 마지막 스텝에 주는 보너스.')
  ap.add_argument('--kappa', type=float, default=1.0,
                  help='--reward-v2 전용. PBRS 진행항 가중치.')
  ap.add_argument('--plain-shaping', action='store_true', default=False,
                  help=('--reward-v2 전용. 진행항을 kappa*(gamma*Phi(o_{t+1})-Phi(o_t)) '
                        '대신 원래 식 그대로 kappa*(d(o_t,g)-d(o_{t+1},g))로 쓴다. '
                        '대수적으로는 여전히 옳은 PBRS 항 + (1-gamma)*Phi(o_{t+1})(원래의 '
                        '암묵적 기본보상 일부)를 포함하므로 완전한 교체는 아니지만 더 단순함.'))
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  if args.seed0 >= 900_000:
    raise ValueError('--seed0가 평가 시드 대역(900000+)과 겹친다 — 학습 시드는 '
                     '그 아래 값을 써라.')

  ck, nets, ddpo, act_dim, horizon, exec_horizon = load_policy_for_training(args.policy_ckpt)
  if horizon > 1:
    print(f'[청킹] horizon={horizon} exec_horizon={exec_horizon} (act_dim={act_dim}, '
          f'chunk_flat_dim={act_dim * horizon})')
  params = ck['params']
  base_params = jax.tree.map(np.copy, params)          # 드리프트 비교용 고정 사본
  norm_stats = ck['norm_stats']
  act_mean, act_std = norm_stats['act_mean'], norm_stats['act_std']
  frame_mean, frame_std = norm_stats['frame_mean'], norm_stats['frame_std']
  n_steps = nets.n_steps
  dist_head_keys = _dist_head_keys(params)

  reward = StgReward(args.d_ckpt, statistic=args.statistic,
                     cvar_alpha=args.cvar_alpha)
  # reward.meta['data']를 쓴다(하드코딩된 별도 경로 대신) — 캘리브레이션/드리프트
  # 진단이 항상 이 --d-ckpt가 실제로 학습된 데이터와 일치하게 강제하기 위함.
  # 예전엔 DRIFT_DATA_PATH가 'data/grasp_carry_demos_v3.pkl'로 고정돼 있어서,
  # v4/v5 계열 d-ckpt를 쓸 때도 v3로 캘리브레이션해 문턱이 완전히 어긋났었다
  # (f1=0.151 — --termination learned가 이 잘못된 문턱을 파고들며 붕괴, 2026-08-09).
  calib_data_path = reward.meta['data']
  with open(calib_data_path, 'rb') as fp:
    calib_data = pickle.load(fp)
  val_eps = _val_episode_ids(calib_data, seed=reward.meta['seed'])
  best_s, calib_metrics = calibrate_threshold(reward, calib_data, val_eps)
  reward.threshold = best_s
  print(f'[calib] threshold s={best_s:.3f}  f1={calib_metrics["f1"]:.3f}  '
        f'(data={calib_data_path}, held-out {len(val_eps)} episodes)')

  # --sanitize-terminal 목표값(reset_cost + 평균 에피소드 길이) 계산용 — calib_data의
  # 성공 에피소드 길이 평균. success-only 데이터면 전체가 성공이라 그대로 다 쓴다.
  calib_succ_mask = np.asarray(calib_data['is_success'], dtype=bool)
  calib_eids = np.asarray(calib_data['episode_id'])
  succ_eids = np.unique(calib_eids[calib_succ_mask])
  mean_episode_len = float(np.mean([np.sum(calib_eids == e) for e in succ_eids])) if len(succ_eids) else 0.0
  if args.sanitize_terminal:
    print(f'[sanitize-terminal] target={args.reset_cost + mean_episode_len:.1f} '
          f'(reset_cost={args.reset_cost:.1f} + mean_episode_len={mean_episode_len:.1f}, '
          f'{len(succ_eids)}개 성공 에피소드 기준)')

  cfg = CarryConfig()
  env = GraspCarry2D(cfg)

  optimizer = optax.adam(args.lr)
  opt_state = optimizer.init(params)
  train_step = build_train_step(ddpo, dist_head_keys)
  sample_fn = jax.jit(ddpo.sample_with_trace)

  drift_obs_n = _load_drift_obs(calib_data_path, frame_mean, frame_std)

  rng_key = jax.random.PRNGKey(args.seed0 + 1_000_000)   # 정책 샘플링 전용 스트림
  env_seed_counter = args.seed0

  first_update_checked = False
  n_pairs_per_step = n_steps - 1

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

  # best-so-far 체크포인트 — episodes_per_iter가 작으면(여기 32도 이 태스크엔 작다) 정점을
  # 찍고 다시 나빠지는 게 흔하다(2026-08-10 실측: it=30 성공률 50% -> it=40 9.4%). 마지막
  # iteration만 저장하면 그 정점을 잃는다 — mani_sim/train_si.py에서 겪은 뒤 고친 것과
  # 동일한 문제라 여기도 같은 방식으로 고친다.
  best_succ_rate, best_R_mean, best_it = -1.0, -np.inf, None
  out_root, out_ext = os.path.splitext(args.out)
  best_out = f'{out_root}_best{out_ext}'

  def _save_ckpt(path, epoch):
    out_ckpt = dict(ck)
    out_ckpt['params'] = params
    out_ckpt['meta'] = dict(ck['meta'])
    out_ckpt['meta'].update(dict(
        si_d_ckpt=args.d_ckpt, si_statistic=args.statistic, si_gamma=args.gamma,
        si_reinforce_scale=args.reinforce_scale, si_lr=args.lr,
        si_iterations=epoch, si_episodes_per_iter=args.episodes_per_iter,
        si_termination=args.termination, si_advantage_norm=args.advantage_norm,
        si_base_ckpt=args.policy_ckpt,
    ))
    with open(path, 'wb') as fp:
      pickle.dump(out_ckpt, fp)

  for it in range(1, args.iterations + 1):
    t_iter_start = time.time()

    ep_obs_n, ep_xs, ep_R, ep_len, ep_outcome = [], [], [], [], []
    ep_outcome_trained = []  # ep_obs_n/ep_xs/ep_R와 항상 같은 길이·순서로 유지되는 outcome
                             # (--truncate-terminal의 n<2 스킵으로 ep_outcome과 길이가
                             # 갈라질 수 있어 --success-only 필터링은 반드시 이걸로 인덱싱)
    for _ in range(args.episodes_per_iter):
      obs_n_all, xs_all, d_vals, sigma_vals, outcome, rng_key = collect_episode(
          nets, ddpo, params, reward, env, env_seed_counter, rng_key,
          act_mean, act_std, frame_mean, frame_std, args.termination,
          cfg.max_steps, sample_fn, act_dim=act_dim, horizon=horizon,
          exec_horizon=exec_horizon)
      env_seed_counter += 1
      ep_outcome.append(outcome)  # env_succ_rate 등은 아래 학습배치 필터링과 무관하게
                                  # 항상 이 시점(=실제로 굴린 결과)을 기준으로 낸다 —
                                  # --truncate-terminal의 n<2 조기 continue가 이 기록을
                                  # 건너뛰지 않도록 reward 계산보다 먼저 남겨둔다.

      obs_n_use, xs_use, sigma_use = obs_n_all, xs_all, sigma_vals

      if args.reward_v2:
        rewards = compute_step_rewards_v2(d_vals, outcome, args.step_cost,
                                          args.success_bonus, args.gamma, args.kappa,
                                          plain_shaping=args.plain_shaping)
      else:
        d_vals_use = d_vals
        if args.truncate_terminal:
          n = truncate_before_false_success(d_vals, outcome,
                                            near_success_thresh=args.near_success_thresh,
                                            lookback=args.sanitize_lookback)
          if n < 2:
            continue  # 쓸 transition이 하나도 안 남으면 이 에피소드는 이번 배치에서 버린다
          d_vals_use = d_vals[:n]
          obs_n_use = obs_n_all[:n - 1]
          xs_use = xs_all[:n - 1]
          sigma_use = sigma_vals[:n]
        elif args.sanitize_terminal:
          d_vals_use = sanitize_terminal_d(d_vals, outcome, args.reset_cost, mean_episode_len,
                                           near_success_thresh=args.near_success_thresh,
                                           lookback=args.sanitize_lookback)
        rewards = compute_step_rewards(d_vals_use)

      if args.sigma_shaping_coef != 0.0:
        rewards = rewards + args.sigma_shaping_coef * compute_sigma_shaping(sigma_use, args.gamma)

      returns = compute_returns(rewards, args.gamma)

      if args.variance_weight:
        # 역분산 가중치: 예측기가 그 스텝의 시작 상태(o_t)를 확신하지 못할수록
        # (sigma_vals[t] 큼) 그 결정에서 나온 학습 신호를 깎는다. 에피소드
        # 안에서만 정규화(평균 가중치=1)해서 reinforce_scale의 전체 크기 감각은
        # 유지한다 — 2026-08-11, 사용자 가설: 예전(2026-08-07) sigma 활용
        # 시도가 실패한 건 그때 예측기의 sigma가 OOD에 반응 안 했기 때문일
        # 수 있다(오늘 새 예측기로 재검증됨, sigma_ratio 1.1x -> 2.5x).
        #
        # 안전장치 이중화(2026-08-11, 사용자 지적으로 추가) — 실측: 이 예측기의
        # sigma 분포가 median=30.3인데 min=0.0(!)까지 내려간다. 원래 1e-3 엡실론만
        # 있었을 때는 sigma가 아주 작아지는 순간 가중치가 수백~수천 배로 튀어
        # 그 한 스텝이 배치 전체 그래디언트를 지배할 수 있었다:
        #   1) SIGMA_FLOOR로 분모 자체에 바닥을 둔다(중앙값의 1/30 수준 밑으로는
        #      "이 정도로 확신한다"고 안 믿는다).
        #   2) 정규화 후 최종 가중치도 [0.1, 5.0]로 다시 한번 자른다(그래도 남을 수
        #      있는 꼬리를 완전히 막는 이중 방어).
        SIGMA_FLOOR = 1.0
        w = 1.0 / np.maximum(sigma_use[:-1], SIGMA_FLOOR)
        w = w / w.mean()
        w = np.clip(w, 0.1, 5.0)
        returns = returns * w

      ep_obs_n.append(obs_n_use)
      ep_xs.append(xs_use)
      ep_R.append(returns)
      ep_len.append(len(rewards))
      ep_outcome_trained.append(outcome)

    # env_succ_rate 등 로그는 항상 이 iteration에서 실제로 굴린 episodes_per_iter개
    # 전체를 기준으로 낸다(--success-only 여부와 무관) — 필터링은 학습 배치 구성에만 적용.
    n_env_succ = sum(1 for o in ep_outcome if o == 'success')

    if args.success_only:
      keep = [i for i, o in enumerate(ep_outcome_trained) if o == 'success']
      if not keep:
        # 배치에 성공이 하나도 없다 — 업데이트를 건너뛴다(빈 배치로 그래디언트를 못 냄).
        drift = _drift_metric(sample_fn, params, base_params, drift_obs_n)
        iter_time = time.time() - t_iter_start
        print(f'it={it:4d}  (성공 0/{args.episodes_per_iter} - 업데이트 건너뜀)  '
              f'drift_L2={drift:7.4f}  time={iter_time:5.1f}s', flush=True)
        continue
      ep_obs_n = [ep_obs_n[i] for i in keep]
      ep_xs = [ep_xs[i] for i in keep]
      ep_R = [ep_R[i] for i in keep]

    obs_n_all = np.concatenate(ep_obs_n, axis=0)         # (N, obs_dim)
    xs_all = np.concatenate(ep_xs, axis=0)                # (N, n_steps+1, act_dim)
    R_all = np.concatenate(ep_R, axis=0)                   # (N,)
    N = obs_n_all.shape[0]

    if args.advantage_norm:
      R_train = (R_all - R_all.mean()) / (R_all.std() + 1e-8)
    else:
      R_train = R_all

    # ---- (env-step, kk) 쌍 나열 → 셔플 → 미니배치 순회 (이 배치로 딱 한 번)
    i_in, i_out, kk = make_pair_index_table(n_steps)       # 각 (n_steps-1,)
    j_idx = np.repeat(np.arange(N), n_pairs_per_step)
    p_idx = np.tile(np.arange(n_pairs_per_step), N)
    kk_all = kk[p_idx]
    i_in_all = i_in[p_idx]
    i_out_all = i_out[p_idx]

    perm = np.random.default_rng(args.seed0 + it).permutation(len(j_idx))
    j_idx, i_in_all, i_out_all, kk_all = (a[perm] for a in
                                          (j_idx, i_in_all, i_out_all, kk_all))

    n_total_pairs = len(j_idx)
    losses = []
    for start in range(0, n_total_pairs, args.logp_batch):
      sl = slice(start, start + args.logp_batch)
      bj, bi_in, bi_out, bkk = j_idx[sl], i_in_all[sl], i_out_all[sl], kk_all[sl]

      obs_b = jnp.asarray(obs_n_all[bj])
      x_in_b = jnp.asarray(xs_all[bj, bi_in])
      x_out_b = jnp.asarray(xs_all[bj, bi_out])
      kk_b = jnp.asarray(bkk, dtype=jnp.int32)
      R_b = jnp.asarray(R_train[bj], dtype=jnp.float32)

      if not first_update_checked:
        params_before = jax.tree.map(np.copy, params)

      params, opt_state, loss, _ = train_step(
          params, opt_state, optimizer, obs_b, x_in_b, x_out_b, kk_b, R_b,
          args.reinforce_scale)
      losses.append(float(loss))

      if not first_update_checked:
        _assert_dist_head_frozen(params_before, params, dist_head_keys)
        first_update_checked = True
        print('[check] 첫 업데이트 후 STG(dist) 헤드 파라미터 불변 확인됨.')

    # ---- 로그 -----------------------------------------------------------
    drift = _drift_metric(sample_fn, params, base_params, drift_obs_n)
    total_steps = sum(ep_len)
    demos_per_1k = 1000.0 * n_env_succ / max(total_steps, 1)
    iter_time = time.time() - t_iter_start

    print(f'it={it:4d}  R_mean={R_all.mean():+8.4f}  |R|_mean={np.abs(R_all).mean():7.4f}  '
          f'env_succ_rate={n_env_succ / args.episodes_per_iter:5.1%}  '
          f'ep_len_mean={np.mean(ep_len):6.1f}  demos/1k={demos_per_1k:6.2f}  '
          f'drift_L2={drift:7.4f}  loss={np.mean(losses):+9.5f}  '
          f'time={iter_time:5.1f}s', flush=True)

    if not np.isfinite(R_all).all() or not np.isfinite(drift):
      raise RuntimeError(
          f'it={it}: 리턴 또는 드리프트가 비유한(NaN/inf)이 됐다 — 중단. '
          f'R_all finite={np.isfinite(R_all).all()}  drift={drift}')

    succ_rate = n_env_succ / args.episodes_per_iter
    is_new_best = (succ_rate > best_succ_rate or
                  (succ_rate == best_succ_rate and float(R_all.mean()) > best_R_mean))
    if is_new_best:
      best_succ_rate, best_R_mean, best_it = succ_rate, float(R_all.mean()), it
      _save_ckpt(best_out, it)
      print(f'  [best 갱신] it={it} env_succ_rate={succ_rate:.1%} -> {best_out}', flush=True)

  _save_ckpt(args.out, args.iterations)
  print(f'저장(final): {args.out}')
  if best_it is not None:
    print(f'best-so-far: it={best_it} env_succ_rate={best_succ_rate:.1%} -> {best_out}')
  else:
    print(f'best-so-far: 없음(모든 iteration에서 업데이트가 건너뛰어짐 — '
          f'--success-only인데 배치마다 성공이 0이었음). {best_out}는 저장되지 않았다.')


if __name__ == '__main__':
  main()
