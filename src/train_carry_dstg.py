r"""GraspCarry2D 관측-only STG 예측기 d(o,g) := E[steps-to-go | o, g] (phase 4, step 1).

논문(SI-EFM) 식(1)의 거리 함수는 관측과 목표만 입력받는다 — 액션은 입력이
아니다. 이 스크립트는 그 `d`를 두 가지 버전으로 학습한다:

  succ: 성공 에피소드만 학습(논문 원안. Stage-1 데모가 성공 시연이므로 구조적
        으로 실패를 표현할 수 없다). num_bins=200 (0..199).
  fail (--include-failures --fail-mode class, 기본값): 실패 에피소드까지
        포함. 마지막 bin(=200)을 "실패" 클래스로 따로 두어 d가 "이 상태는
        실패할 확률이 높다"를 표현하게 한다. num_bins=201 (0..199 성공 +
        200 실패). **주의**: 이 fail_bin=200(=max_steps)이라는 숫자는 근거
        없는 상수다 — 이후 `carry_stg_reward.py --fail-value`로 d를 계산할
        때 "실패 = 200스텝짜리 성공"으로 취급하게 되는데, 200은 그냥
        max_steps를 재사용한 것일 뿐 실제 비용과 무관하다(phase 4 arm B가
        이 값 때문에 붕괴함 — 정책이 "가만히 있으면 타임아웃도 200, 움직여서
        실패해도 200"이라 움직일 유인이 없어짐).
  fail (--include-failures --fail-mode deadline): 실패 transition의 라벨을
        별도 클래스가 아니라 **숫자**로 직접 준다(같은 num_bins=200 축 안에
        성공/실패가 자연스럽게 합쳐짐, bimodal 아님). 실패 에피소드 안에서도
        "언제 실패가 판정됐는가"로 두 구간을 나눠 다르게 라벨링한다 — 판정
        전 상태는 같은 상황에서 성공했을 수도 있는 것이라(하나의 rollout
        실현일 뿐) 정답을 모르고, 판정 순간(env.py의 is_tipped()/timeout이
        실제로 발동한 그 transition)만 진짜로 "실패"라는 사실을 안다:
          - 판정 순간(에피소드당 1개): `B = reset_cost + T̂`.
            `reset_cost`(--reset-cost, 기본 30)는 사람이 넘어진 블록을 다시
            세우는 비용(자동 리셋 20보다 조금 더, 에피소드 평균 길이보다는
            짧게), `T̂`는 **이 학습 데이터에서 실측한** 성공 에피소드 평균
            길이(자동 계산). renewal-theory 포기판정 문헌의 "데드라인" 공식
            (`references/context_8.md`)과 정확히 같은 양이다.
          - 판정 이전 transition: 아직 "정상적으로 보이는 상태"라 정답이
            없으므로, 인간 데모만으로 학습된 --bootstrap-ckpt(기본 succ
            버전)의 예측 평균을 라벨로 재사용한다(부트스트랩). cross-entropy는
            개별 라벨이 숫자 하나씩이어도 비슷한 상태가 여럿 모이면 경험적
            분포로 수렴하므로(succ/class 학습도 이 성질에 의존), 부트스트랩도
            같은 방식(숫자 하나)으로 주는 게 나머지와 일관된다.

두 버전은 데이터 소스·아키텍처·하이퍼파라미터를 전부 동일하게 맞춘다 —
나중에 나오는 정책 성능 차이를 "실패를 아는가" 한 가지 요인으로 돌리기 위함.

정규화기·데이터 로딩·val 분할 방식은 `src/train_carry_predictor.py`를,
실패 bin 라벨링 규칙은 `src/train_carry_qstg.py`(`split_success_fail`)를
따른다. 아키텍처는 `src/diffusion_act.py`의 STG 헤드
(`hk.nets.MLP(...) -> hk.Linear(num_bins, with_bias=False)`)와 동일하다.

데이터: `data/grasp_carry_demos_v3.pkl` (`collect_carry_demos.py --keep-failures`)

실행:
  python -m src.train_carry_dstg --data data/grasp_carry_demos_v3.pkl \
      --out checkpoints/grasp_carry_dstg_succ/predictor.pkl
  python -m src.train_carry_dstg --data data/grasp_carry_demos_v3.pkl \
      --include-failures --out checkpoints/grasp_carry_dstg_fail/predictor.pkl
"""

import argparse
import os
import pickle
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

OBS_FIELDS = ('frame',)
DEFAULT_LAYER_SIZES = (256, 256, 256)


# -------------------------------------------------------------- normalizers
# From: src/train_carry_predictor.py (compute_stats / make_normalizers) —
# 관측 정규화만 필요하므로 액션 통계는 뺐다.
def compute_stats(data):
  obs = data['observation']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  for f in OBS_FIELDS:
    stats[f'{f}_mean'], stats[f'{f}_std'] = ms(obs[f])
  return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def make_normalizers(stats):
  def normalize_obs(obs):
    return {f: (obs[f] - stats[f'{f}_mean']) / stats[f'{f}_std']
            for f in OBS_FIELDS}
  return normalize_obs


def concat_obs(obs):
  return jnp.concatenate([jnp.asarray(obs[f]) for f in OBS_FIELDS], axis=-1)


# ------------------------------------------------------------------ network
def build_dstg_net(layer_sizes, obs_dim, num_bins):
  """관측만 받아 카테고리컬 logits을 내는 MLP.

  구조는 src/diffusion_act.py의 STG 헤드와 동일:
    hk.nets.MLP(layer_sizes, relu, activate_final=True) -> hk.Linear(num_bins, with_bias=False)
  반환: (apply_fn, init_fn) — src/train_carry_qstg.py의 build_qstg_net과 같은 형태.
  """
  def _net(obs):
    h = hk.nets.MLP(layer_sizes, activation=jax.nn.relu,
                    activate_final=True)(obs)
    return hk.Linear(num_bins, with_bias=False)(h)

  tn = hk.without_apply_rng(hk.transform(_net))

  def init(rng):
    return tn.init(rng, jnp.zeros((2, obs_dim), jnp.float32))

  return tn.apply, init


class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default='data/grasp_carry_demos_v3.pkl')
  ap.add_argument('--include-failures', action='store_true',
                  help=('실패 transition까지 포함해 학습한다. 없으면 성공 '
                        'transition만 사용(논문 원안).'))
  ap.add_argument('--fail-mode', choices=['class', 'deadline'], default='class',
                  help=('--include-failures일 때만 의미 있음. class(기본): 마지막 '
                        'bin을 별도 실패 클래스로 둔다(기존 방식, num_bins=201). '
                        'deadline: 실패 transition의 라벨을 숫자 B=reset_cost+T̂'
                        '(T̂=이 데이터의 실측 성공 에피소드 평균 길이)로 직접 준다 — '
                        '같은 num_bins=200 축 안에 성공/실패가 자연스럽게 합쳐진다. '
                        'renewal-theory 포기판정 데드라인(references/context_8.md)과 '
                        '같은 양이다.'))
  ap.add_argument('--reset-cost', type=float, default=30.0,
                  help=('--fail-mode deadline 전용. 전도 후 사람이 블록을 다시 '
                        '세우는 리셋 비용(스텝 환산). 자동 리셋(eval_carry_si.py 기본 '
                        '20)보다 조금 더 걸리되 에피소드 평균 길이보다는 짧아야 '
                        '한다는 게 근거(200처럼 임의로 크게 잡지 않는다).'))
  ap.add_argument('--outlier-iqr-mult', type=float, default=1.5,
                  help=('--fail-mode deadline 전용. T_hat/T_hat_std를 계산할 때 '
                        '성공 에피소드 길이의 IQR 기준(Tukey 표준 관행) 이 배수를 '
                        '넘는 장기 지연 에피소드를 이상치로 빼고 계산한다. 실측: '
                        '이 무리가 std를 28.09->9.25로 부풀리고 있었다.'))
  ap.add_argument('--judgment-batch-frac', type=float, default=0.15,
                  help=('--fail-mode deadline 전용. 매 배치에서 판정 순간(전체의 '
                        '~0.27%%뿐) 샘플이 최소 이 비율만큼은 들어가도록 층화한다 — '
                        '균등샘플링만으로는 배치당 평균 1개도 안 뽑혀 나머지 신호에 '
                        '묻힌다(실측). 0이면 층화 없음(기존 균등샘플링).'))
  ap.add_argument('--bootstrap-ckpt', default='checkpoints/grasp_carry_dstg_succ/predictor.pkl',
                  help=('--fail-mode deadline 전용. 실패가 "판정되기 전" transition의 '
                        '라벨 출처. 그 시점엔 아직 실패가 확정 안 됐고(같은 상황에서 '
                        '성공했을 수도 있다), 우리가 실제로 아는 건 "그 상태가 성공만 '
                        '학습한 예측기 눈엔 어때 보이는가"뿐이다 — 그래서 이 얼려진 '
                        '체크포인트(기본: succ 버전)의 예측 평균을 그 transition의 '
                        '학습 라벨로 재사용한다(부트스트랩).'))
  ap.add_argument('--steps', type=int, default=16384)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--warmup', type=int, default=500)
  ap.add_argument('--eval-every', type=int, default=500)
  ap.add_argument('--patience', type=int, default=12)
  ap.add_argument('--weight-decay', type=float, default=1e-4)
  ap.add_argument('--no-early-stop', action='store_true')
  ap.add_argument('--out', required=True, help='체크포인트 저장 경로')
  args = ap.parse_args()

  with open(args.data, 'rb') as fp:
    data = pickle.load(fp)
  N = len(data['action'])
  max_steps = int(data['meta']['max_steps'])
  is_succ = data['is_success']
  ttg = data['time_to_success']
  print(f'transitions={N}  에피소드={len(np.unique(data["episode_id"]))}  '
        f'성공율={is_succ.mean():.1%}  outcomes={data["meta"]["outcomes"]}  '
        f'실패 ttg 유일값={np.unique(ttg[~is_succ])}')

  deadline_mode = args.include_failures and args.fail_mode == 'deadline'
  if args.include_failures and args.fail_mode == 'class':
    fail_bin = max_steps
    NUM_BINS = max_steps + 1     # 0..199 성공 + 200 실패
  else:
    fail_bin = None
    NUM_BINS = max_steps         # 0..199 (succ 버전과 deadline 버전 공용)

  # ---- 정규화 / 분할 (held-out은 에피소드 단위 10%, train_carry_predictor.py와 동일)
  stats = compute_stats(data)
  normalize_obs = make_normalizers(stats)
  rng = np.random.default_rng(args.seed)
  ep_ids = np.unique(data['episode_id'])
  val_eps = set(rng.choice(ep_ids, size=max(len(ep_ids) // 10, 1),
                           replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  deadline_B = None
  if deadline_mode:
    # T_hat: 이 데이터의 실측 성공 에피소드 평균 길이 — train split(검증 유출 방지)만으로
    # 계산한다. 손으로 넣지 않고 매번 데이터에서 다시 재는 이유: 데이터가 바뀌면(예: 더
    # 많은 시연 추가) B도 같이 갱신돼야 근거가 유지된다.
    tr_ep_mask = ~val_mask
    ep_ids_tr = data['episode_id'][tr_ep_mask]
    succ_tr = is_succ[tr_ep_mask]
    uniq_tr_eps = np.unique(ep_ids_tr)
    ep_len_tr = np.array([(ep_ids_tr == e).sum() for e in uniq_tr_eps])
    ep_succ_tr = np.array([succ_tr[ep_ids_tr == e][0] for e in uniq_tr_eps])
    succ_len_tr = ep_len_tr[ep_succ_tr]

    # 성공 에피소드 길이 분포는 단일 분포가 아니다 — 실측: 425개 중 232개가
    # 40~50스텝에 몰려 있고(주류 모드), 172~173스텝에 정확히 겹치는 별도 무리가
    # 10개 있다(장기 지연 모드). IQR 기준(Tukey, 1.5×IQR 표준 관행) 이상치가
    # 57/425(13.4%)로 잡히고, 이걸 빼면 std가 28.09→9.25로 확 줄어든다 — 이
    # 장기 지연 소수 그룹이 "전형적인 변동성"을 부풀리고 있었다는 뜻이다. T_hat/
    # T_hat_std를 이 이상치를 뺀 "전형적인" 성공 에피소드만으로 계산한다 —
    # reset_cost와 더해지는 B의 의미가 "지금 포기하고 리셋하면 **전형적으로**
    # 얼마 만에 새로 성공할까"이므로, 드문 장기 지연 사례까지 그 "전형" 추정에
    # 끌어들이는 게 오히려 부적절하다.
    q1, q3 = np.percentile(succ_len_tr, [25, 75])
    iqr = q3 - q1
    upper = q3 + args.outlier_iqr_mult * iqr
    is_typical = succ_len_tr <= upper
    T_hat = float(succ_len_tr[is_typical].mean())
    T_hat_std = float(succ_len_tr[is_typical].std())
    n_outlier = int((~is_typical).sum())
    print(f'[deadline 모드] 성공 에피소드 길이(train, n={len(succ_len_tr)}): '
          f'전체 mean={succ_len_tr.mean():.2f}±{succ_len_tr.std():.2f}, '
          f'IQR 이상치(>{upper:.1f}, {args.outlier_iqr_mult:g}×IQR) {n_outlier}개 제외 후 '
          f'T_hat={T_hat:.2f}±{T_hat_std:.2f}')
    deadline_B = args.reset_cost + T_hat
    deadline_bin = int(np.clip(round(deadline_B), 0, NUM_BINS - 1))
    print(f'[deadline 모드] reset_cost={args.reset_cost:.1f}  '
          f'B={deadline_B:.2f} -> bin {deadline_bin}')

    # "실패가 실현되는 순간까지 남은 스텝"은 성공 라벨과 종류가 다른 양이다 — 그
    # 상태에서 실제로는 성공했을 수도 있는데(같은 상황이 늘 같은 결과로 가지 않는다),
    # 우리가 관측한 건 이 특정 rollout 하나가 실패로 끝났다는 사실뿐이다. 그래서
    # 딱 하나만 확실히 안다: env.py의 is_tipped()가 실제로 발동한 **그 에피소드의
    # 마지막 transition**(물리 스텝 순서상 판정이 일어난 바로 그 시점 — env.step()이
    # 종료 조건을 만족한 스텝의 관측을 그대로 반환하므로 별도 재구성 없이 "그 실패
    # 에피소드의 마지막 기록 transition"이 곧 판정 순간이다)만 B 라벨을 받는다.
    #
    # **timeout은 여기서 제외한다.** tipped와 달리 timeout의 마지막 transition은
    # "그 순간의 관측이 물리적으로 확정된 나쁜 사건"이 아니라 그냥 "시간 예산이
    # 다 됨"이다 — 그 관측 자체는 딱히 나쁘게 안 보일 수 있다(실측: succ 예측기의
    # 부트스트랩 μ가 timeout 마지막 프레임 평균 57.6, tipped 마지막 프레임 평균
    # 54.2로 거의 같아, timeout 순간이 "명백히 나쁜" 상태로 보이지 않는다). 여기에
    # 억지로 B 라벨을 박으면 "이 특정 배치·블록 배열 자체가 원래 나쁘다"는 잘못된
    # 인과를 가르친다. 그래서 timeout 에피소드는 마지막 transition을 포함해 전부
    # 부트스트랩으로 둔다 — "이 순간이 나쁘다"고 확정할 근거가 없으니 확정하지 않는다.
    #
    # tipped 판정 이전 transition, timeout 에피소드 전체는 아직 "정상적으로 진행
    # 중인 것처럼 보이는 상태"라 정답을 모른다 — 대신 인간 데모만으로 학습된
    # --bootstrap-ckpt(기본 succ 버전)의 예측 평균을 그 자리의 라벨로 쓴다
    # (부트스트랩). cross-entropy는 개별 라벨이 숫자 하나씩이어도 비슷한 상태가
    # 여럿 모이면 그 경험적 분포로 수렴하는 성질이 있으므로(succ/class 학습이
    # 이미 이 성질에 의존한다), 부트스트랩 라벨도 같은 방식(숫자 하나)으로 주는
    # 게 나머지 학습과 일관된다 — 분포를 통째로 옮기는 별도 손실을 추가할 필요가
    # 없다.
    eid_all = data['episode_id']
    uniq_eids, ep_start, ep_count = np.unique(eid_all, return_index=True, return_counts=True)
    last_idx_of_episode = ep_start + ep_count - 1
    is_last_of_episode = np.zeros(N, dtype=bool)
    is_last_of_episode[last_idx_of_episode] = True

    # 에피소드 길이==max_steps는 정의상 timeout뿐이다(tipped는 그보다 먼저 끝난다).
    # 이 방식으로 27/48(timeout/tipped) meta 집계와 정확히 일치함을 실측 확인했다.
    is_timeout_episode = ep_count == max_steps
    is_timeout_transition = np.repeat(is_timeout_episode, ep_count)

    fail_judgment_mask = is_last_of_episode & (~is_succ) & (~is_timeout_transition)
    fail_pre_mask = (~is_succ) & (~fail_judgment_mask)     # timeout 전체 + tipped 판정 이전
    print(f'[deadline 모드] 실패 transition {int((~is_succ).sum())}개 중 '
          f'판정순간={int(fail_judgment_mask.sum())}개(tipped만, B 라벨), '
          f'판정이전+timeout전체={int(fail_pre_mask.sum())}개(부트스트랩 라벨)')

    with open(args.bootstrap_ckpt, 'rb') as fp:
      boot_ck = pickle.load(fp)
    boot_apply, _ = build_dstg_net(tuple(boot_ck['layer_sizes']), int(boot_ck['obs_dim']),
                                   int(boot_ck['num_bins']))
    boot_obs_n = ((data['observation']['frame'][fail_pre_mask] - boot_ck['norm_stats']['frame_mean'])
                 / boot_ck['norm_stats']['frame_std'])
    boot_logits = jax.jit(boot_apply)(boot_ck['params'], jnp.asarray(boot_obs_n, dtype=jnp.float32))
    boot_probs = jax.nn.softmax(boot_logits, axis=-1)
    boot_bin_vals = jnp.arange(int(boot_ck['num_bins']), dtype=jnp.float32)
    boot_mean = np.asarray(jnp.sum(boot_probs * boot_bin_vals[None, :], axis=-1))
    boot_labels = np.clip(np.round(boot_mean), 0, NUM_BINS - 1)
    print(f'[deadline 모드] 부트스트랩({args.bootstrap_ckpt}) 예측 평균 분포: '
          f'min={boot_mean.min():.1f} median={np.median(boot_mean):.1f} max={boot_mean.max():.1f}')

    fail_labels = np.empty(int((~is_succ).sum()), dtype=np.float32)
    # ~is_succ 순서(원배열 순서 유지) 기준으로 두 마스크를 다시 슬라이싱해 합친다.
    fail_positions = np.where(~is_succ)[0]
    judgment_in_fail = fail_judgment_mask[fail_positions]
    # 여기 넣는 deadline_bin은 **검증(val)에서만** 쓰인다 — 매 평가마다 같은 목표로
    # 재는 게 맞다(안 그러면 "최적 step" 선택 자체가 매번 흔들린다). 학습(train)에서는
    # 아래 train_step 안에서 이 자리를 매 스텝 새로 뽑은 가우시안 샘플로 덮어쓴다 —
    # 판정 순간 75개가 전부 똑같은 숫자(79)만 보면, 예측기가 "판정 순간처럼 보이는
    # 상태엔 무조건 확신을 가지라"고 배워 그 지점에서 인위적으로 좁은 σ를 낼 수 있다
    # (이 프로젝트가 계속 문제 삼아온 바로 그 함정). 같은 입력에 서로 다른 숫자
    # 라벨이 반복해서 들어오면 cross-entropy가 자연스럽게 그 퍼짐을 반영하므로,
    # 정적 상수 대신 실제 분산(성공 에피소드 길이의 실측 std)을 가진 가우시안에서
    # 매번 새로 뽑는다.
    fail_labels[judgment_in_fail] = deadline_bin
    fail_labels[~judgment_in_fail] = boot_labels

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))

  if args.include_failures:
    keep_mask = np.ones(N, dtype=bool)
  else:
    keep_mask = is_succ

  tr_mask = (~val_mask) & keep_mask
  # 검증셋도 같은 keep_mask로 거른다. succ 버전(num_bins=200)은 라벨 200을
  # 표현할 bin이 아예 없어 실패 transition을 검증에 섞으면 "구조적으로 못
  # 맞히는" 오차가 그대로 MAE를 부풀린다 — 참조 체크포인트(v2 데이터)도
  # 애초에 실패 에피소드가 없는 데이터로 검증했으므로 조건이 맞다.
  va_mask = val_mask & keep_mask

  # deadline 모드에서는 실패 transition의 라벨을 원본 ttg(=max_steps 상수, 근거도
  # 없고 에피소드 안 위치 정보도 없는 값)가 아니라 방금 구한 "실패까지 남은 스텝+B"
  # (transition마다 다름, 근거 있는 값)로 바꿔치기한다. class 모드/succ 모드는
  # 원본 ttg를 그대로 쓴다(기존 동작 유지).
  ttg_used = ttg.copy()
  if deadline_mode:
    ttg_used[~is_succ] = fail_labels

  tr_obs = jnp.asarray(obs_c[tr_mask]); tr_ttg = jnp.asarray(ttg_used[tr_mask])
  va_obs = jnp.asarray(obs_c[va_mask]); va_ttg = jnp.asarray(ttg_used[va_mask])
  va_succ = jnp.asarray(is_succ[va_mask])
  # train split 안에서 "판정 순간" transition의 위치(재샘플링 대상) — deadline_mode가
  # 아니면 전부 False라 train_step의 재샘플링 분기가 실질적으로 아무 일도 안 한다.
  tr_is_judgment_np = (fail_judgment_mask[tr_mask] if deadline_mode
                       else np.zeros(tr_mask.sum(), dtype=bool))
  tr_is_judgment = jnp.asarray(tr_is_judgment_np)
  # 판정 순간은 train split 25,583개 중 ~68개(0.27%) 뿐이다 — 배치를 그냥
  # 균등샘플링하면 batch=256 기준 평균 0.7개만 뽑혀 나머지 99.7%(부트스트랩+성공)
  # 신호에 완전히 묻힌다(실측: 이 상태로 학습한 체크포인트가 학습에 실제로 쓰인
  # 판정순간 75개 자체에 대해서도 예측 평균 μ 중앙값=59.8로 목표 B=79.4를 한참
  # 밑돎 — "실패로 흘러가는 순간 μ가 오히려 낮아지는" 눈에 띄는 오류로 나타났다).
  # 그래서 배치 구성을 층화한다: 매 배치의 일정 비율(--judgment-batch-frac)은
  # 판정순간 풀에서만 뽑고, 나머지는 기존처럼 전체에서 뽑는다.
  tr_judgment_positions = jnp.asarray(np.where(tr_is_judgment_np)[0], dtype=jnp.int32)
  mode_label = ('성공만' if not args.include_failures else
               f'실패 포함(deadline B={deadline_B:.1f}, 판정순간만 B, 그 전엔 부트스트랩)' if deadline_mode
               else '실패 포함(class)')
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps, 관측 {tr_obs.shape[-1]}차원)  [{mode_label}, num_bins={NUM_BINS}]')

  OBS_DIM = int(tr_obs.shape[-1])
  apply_fn, init_fn = build_dstg_net(DEFAULT_LAYER_SIZES, OBS_DIM, NUM_BINS)
  bin_vals = jnp.arange(NUM_BINS, dtype=jnp.float32)   # bin_size=1

  sched_lr = optax.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup,
      decay_steps=max(args.steps, args.warmup + 1), end_value=args.lr * 0.05)
  optimizer = optax.adamw(sched_lr, b1=0.95, b2=0.999,
                          weight_decay=args.weight_decay)
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = init_fn(sub)
  state = TrainState(params, optimizer.init(params))

  def loss_fn(p, bo, bt):
    logits = apply_fn(p, bo)
    return -jnp.mean(tfd.Categorical(logits=logits).log_prob(bt))

  # 배치 층화 크기(정적 파이썬 정수 — jit 트레이스 시점에 고정돼야 shape가 정해진다).
  n_judgment = 0
  if deadline_mode and int(tr_judgment_positions.shape[0]) > 0:
    n_judgment = int(round(args.batch * args.judgment_batch_frac))
  n_rest = args.batch - n_judgment
  if deadline_mode:
    print(f'[deadline 모드] 배치 층화: 매 배치 {n_judgment}개는 판정순간 풀'
          f'({int(tr_judgment_positions.shape[0])}개)에서만, 나머지 {n_rest}개는 '
          f'전체({tr_obs.shape[0]}개)에서 균등샘플링')

  @jax.jit
  def train_step(state, key):
    key, batch_key, judg_key, noise_key = jax.random.split(key, 4)
    idx_rest = jax.random.randint(batch_key, (n_rest,), 0, tr_obs.shape[0])
    if n_judgment > 0:
      j_sel = jax.random.randint(judg_key, (n_judgment,), 0, tr_judgment_positions.shape[0])
      idx = jnp.concatenate([idx_rest, tr_judgment_positions[j_sel]])
    else:
      idx = idx_rest
    bt = tr_ttg[idx]
    if deadline_mode:
      # 판정 순간 위치(idx 중 tr_is_judgment==True)는 정적 라벨(deadline_bin) 대신
      # 매 스텝 새로 뽑은 B_i ~ N(deadline_B, T_hat_std)로 덮어쓴다 — 위 주석 참고.
      # val_metrics는 이 분기를 안 타므로(별도 함수) 검증은 여전히 고정된 목표로 잰다.
      judgment_here = tr_is_judgment[idx]
      sampled = jax.random.normal(noise_key, (args.batch,)) * T_hat_std + deadline_B
      sampled_bin = jnp.clip(jnp.round(sampled), 0.0, float(NUM_BINS - 1))
      bt = jnp.where(judgment_here, sampled_bin, bt)
    l, g = jax.value_and_grad(loss_fn)(state.params, tr_obs[idx], bt)
    updates, opt_state = optimizer.update(g, state.opt_state, state.params)
    return TrainState(optax.apply_updates(state.params, updates), opt_state), l

  @jax.jit
  def val_metrics(p):
    logits = apply_fn(p, va_obs)
    nll = -jnp.mean(tfd.Categorical(logits=logits).log_prob(va_ttg))
    probs = jax.nn.softmax(logits, axis=-1)
    exp = jnp.sum(probs * bin_vals[None, :], axis=-1)
    mae = jnp.mean(jnp.abs(exp - va_ttg))
    if fail_bin is not None:
      pred_fail = jnp.argmax(logits, axis=-1) == fail_bin
      fail_acc = jnp.mean(pred_fail == (~va_succ))
    else:
      fail_acc = jnp.float32(jnp.nan)
    # deadline 모드 진단: 실패 에피소드 transition만 놓고 봤을 때 예측 기댓값이
    # 목표(B)에 얼마나 가까운지 — class 모드/succ 모드에서는 va_succ가 전부 True거나
    # (succ) 이 값이 fail_acc와 별개로 참고용이므로 항상 계산해둔다.
    fail_n = jnp.sum(~va_succ)
    mae_fail_only = jnp.sum(jnp.where(~va_succ, jnp.abs(exp - va_ttg), 0.0)) / jnp.maximum(fail_n, 1)
    return mae, nll, fail_acc, mae_fail_only

  t0 = time.time()
  best = dict(loss=np.inf, step=0, params=jax.device_get(state.params),
              mae=np.nan, nll=np.nan, fail_acc=np.nan, mae_fail=np.nan)
  since = 0
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l = train_step(state, sub)
    if step % args.eval_every == 0 or step == 1:
      mae, nll, fail_acc, mae_fail = val_metrics(state.params)
      vloss = float(nll)
      mark = ''
      if vloss < best['loss'] - 1e-4:
        best = dict(loss=vloss, step=step, params=jax.device_get(state.params),
                    mae=float(mae), nll=float(nll), fail_acc=float(fail_acc),
                    mae_fail=float(mae_fail))
        since = 0; mark = '  *best'
      else:
        since += 1
      print(f'step {step:6d}  train={float(l):.3f}  '
            f'val: MAE={float(mae):.2f} NLL={float(nll):.3f}'
            + (f' 실패분류정확도={float(fail_acc):.3f}' if fail_bin is not None else '')
            + (f' 실패transition MAE={float(mae_fail):.2f}' if deadline_mode else '')
            + f'  ({time.time()-t0:.0f}s){mark}', flush=True)
      if (not args.no_early_stop) and since >= args.patience:
        print(f'조기 종료: {args.patience}회 개선 없음 (best step={best["step"]})')
        break
  if not args.no_early_stop:
    state = TrainState(jax.tree.map(jnp.asarray, best['params']), state.opt_state)
  else:
    best['step'] = args.steps
  print(f'\n[최적] step={best["step"]}  val MAE={best["mae"]:.2f}  NLL={best["nll"]:.3f}')

  val_fail_base_rate = float(jnp.mean(~va_succ))
  meta = {
      'data': args.data, 'include_failures': bool(args.include_failures),
      'fail_mode': args.fail_mode if args.include_failures else None,
      'steps': args.steps, 'seed': args.seed,
      'best_step': int(best['step']), 'weight_decay': args.weight_decay,
      'val_mae': float(best['mae']), 'val_nll': float(best['nll']),
      'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  }
  if args.include_failures and args.fail_mode == 'class':
    meta['val_fail_accuracy'] = float(best['fail_acc'])
    meta['val_fail_base_rate'] = val_fail_base_rate
    print(f'val_fail_accuracy={best["fail_acc"]:.3f}  '
          f'val_fail_base_rate(다수결 기준선)={val_fail_base_rate:.3f}  '
          f'차이={best["fail_acc"] - val_fail_base_rate:+.3f}')
    if best['fail_acc'] - val_fail_base_rate < 0.03:
      print('[경고] 실패 분류 정확도가 다수결 기준선을 +0.03 이상 넘지 못함 '
            '— 예측기가 실패를 유의미하게 배우지 못했을 수 있음.')
  if deadline_mode:
    meta['reset_cost'] = args.reset_cost
    meta['T_hat'] = T_hat
    meta['T_hat_std'] = T_hat_std
    meta['outlier_iqr_mult'] = args.outlier_iqr_mult
    meta['n_outlier_episodes_excluded'] = n_outlier
    meta['judgment_batch_frac'] = args.judgment_batch_frac
    meta['n_judgment_train'] = int(tr_judgment_positions.shape[0])
    meta['deadline_B'] = deadline_B
    meta['deadline_bin'] = deadline_bin
    meta['bootstrap_ckpt'] = args.bootstrap_ckpt
    meta['val_mae_fail_only'] = float(best['mae_fail'])
    print(f'[deadline 진단] 실패 transition만의 val MAE={best["mae_fail"]:.2f} '
          f'(목표=판정순간이면 B, 그 전이면 부트스트랩 예측값 — 이 값에 얼마나 '
          f'가깝게 예측하는지, 작을수록 좋음)')

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': {'frame_mean': stats['frame_mean'],
                       'frame_std': stats['frame_std']},
        'obs_dim': OBS_DIM,
        'num_bins': NUM_BINS,
        'fail_bin': fail_bin,
        'max_steps': max_steps,
        'layer_sizes': DEFAULT_LAYER_SIZES,
        'meta': meta,
    }, fp)
  print(f'체크포인트 저장: {args.out}')

  if not args.include_failures:
    ref_mae = 15.04
    ratio = float(best['mae']) / ref_mae
    print(f'\n참조(grasp_carry_diff100, v2 성공 데모) val_mae={ref_mae:.2f} 대비 '
          f'이 succ 버전 val_mae={float(best["mae"]):.2f}  비율={ratio:.2f}x'
          + ('  [자릿수 일치]' if 0.5 <= ratio <= 2.0 else '  [경고: 2배 이상 벗어남]'))


if __name__ == '__main__':
  main()
