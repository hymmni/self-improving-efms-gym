"""장애물 회피 환경(5필드 관측) steps-to-go 예측기 + BC 정책 학습 (phase 3).

파이프라인 (train_sft.py의 구조를 5필드 관측에 맞게 재작성):
  1. 데모 생성: ObstacleAvoidPoint2D + demo_action_pf(노이즈 1.5e-4, 채택값).
     성공 못 한 에피소드(500스텝 상한)는 버림(~1%). 라벨 = time_to_success
     (원본 generate_dataset과 동일하게 arange(len-1,-1,-1)).
     생성 결과는 data/obstacle_demos.pkl로 저장해 재사용(골 입력 ablation 등).
  2. 관측 9차원 concat: cur_pos(2) cur_vel(2) goal_pos(2) obstacle_rel_pos(2)
     obstacle_radius(1). 정규화는 원본 make_normalizers 방식을 확장
     (goal_pos는 원본처럼 cur_pos 통계 공유, 장애물 필드는 자체 통계).
  3. 네트워크: pointmass_core.build_continuous_act_discrete_dist_v0 재사용
     (MLP 256x3, bin_size=1 -> num_bins = max_distance = 420, p99 길이 407 근거).
     bin_size=1이므로 원본 loss의 raw-ttg-as-class-label이 정확히 성립
     (phase-2에서 확인한 잠재 버그 우회와 동일).
  4. 학습 루프: 단일 디바이스 jit (PretrainLearner는 3필드 concat 하드코딩
     + pmap 구조라 재사용하지 않음). Adam 3e-4, minibatch 256, 32768 step
     (train_sft와 동일 하이퍼).
  5. 평가: held-out 전이 MAE/NLL + BC 정책 성공률(100 에피소드).

실행:
  python -m src.train_obstacle_predictor            # 전체 (생성 포함)
  python -m src.train_obstacle_predictor --episodes 10000 --steps 32768
"""

import argparse
import os
import pickle
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from pointmass_core import (
    build_continuous_act_discrete_dist_v0,
    build_mixture_act_discrete_dist_v0,
    build_discrete_distance_converter,
)
from src.obstacle_env import (ObstacleAvoidPoint2D, demo_action_pf, demo_action,
                              demo_action_committed,
                              PartialObsObstacleAvoidPoint2D,
                              demo_action_partial)

OBS_FIELDS = ('cur_pos', 'cur_vel', 'goal_pos', 'obstacle_rel_pos',
              'obstacle_radius')
EP_CAP = 500              # 데모 에피소드 스텝 상한 (넘으면 폐기)
MAX_DISTANCE = EP_CAP     # 성공 에피소드는 정의상 EP_CAP 이하 -> 라벨이 bin
NUM_BINS = MAX_DISTANCE   # 범위를 절대 못 넘음. bin_size=1.

DATA_PATH = 'data/obstacle_demos.pkl'
CKPT_PATH = 'checkpoints/obstacle/predictor.pkl'


# ------------------------------------------------------------------ dataset
def generate_demos(num_episodes, seed0=3000, noise_std=1.5e-4, action_fn=None,
                   action_fn_factory=None, env_fn=None, obs_fields=None):
  """action_fn(obs)->action. 기본(None)이면 기존 노이즈 PF 컨트롤러.
  action_fn=demo_action(접선점 조준)을 넘기면 노이즈 없는 '깔끔한' 데모가 된다
  (2026-07-20: 노이즈가 태스크 무관 불확실성을 얼마나 만드는지 분리하기 위한
  대조군 — --controller tangent).

  action_fn_factory()->action_fn 를 주면 **에피소드마다 새로** 호출해 그 에피소드
  전용 컨트롤러를 만든다(2026-07-21: 우회 방향을 에피소드 시작에 난수로 정하고
  끝까지 커밋하는 committed 컨트롤러용 — side가 관측에 없으므로 같은 관측에
  좌·우 액션이 공존하는 진짜 다봉 데이터가 된다). action_fn보다 우선한다."""
  if action_fn is None and action_fn_factory is None:
    action_fn = lambda obs: demo_action_pf(obs, noise_std=noise_std)
  fields = tuple(obs_fields) if obs_fields else OBS_FIELDS
  obs_lists = {k: [] for k in fields}
  acts, ttgs, ep_ids = [], [], []
  ep_sides = []          # 에피소드별 커밋 방향 기록(분석용; 관측엔 안 들어감)
  n_discard = 0
  t0 = time.time()
  ep = 0
  attempt = 0
  while ep < num_episodes:
    np.random.seed(seed0 + attempt)
    attempt += 1
    env = env_fn() if env_fn is not None else ObstacleAvoidPoint2D()
    ts = env.reset()
    if action_fn_factory is not None:
      ep_action_fn, ep_side = action_fn_factory()
    else:
      ep_action_fn, ep_side = action_fn, 0.0
    ep_obs, ep_act = [], []
    step = 0
    while not env.success() and step < EP_CAP:
      act = ep_action_fn(ts.observation)
      ep_obs.append(ts.observation)
      ep_act.append(np.asarray(act, dtype=np.float32))
      ts = env.step(act)
      step += 1
    if not env.success() or len(ep_obs) < 10:
      n_discard += 1
      continue
    n = len(ep_obs)
    for k in fields:
      obs_lists[k] += [o[k] for o in ep_obs]
    acts += ep_act
    # From: pointmass_core.generate_dataset — time_to_success 라벨링 방식 동일
    ttgs.append(np.arange(n - 1, -1, -1, dtype=np.float32))
    ep_ids.append(np.full(n, ep, dtype=np.int32))
    ep_sides.append(np.full(n, ep_side, dtype=np.float32))
    ep += 1
    if ep % 500 == 0:
      print(f'  {ep}/{num_episodes} episodes '
            f'({time.time() - t0:.0f}s, 폐기 {n_discard})', flush=True)

  data = dict(
      observation={k: np.stack(v).astype(np.float32)
                   for k, v in obs_lists.items()},
      action=np.stack(acts).astype(np.float32),
      time_to_success=np.concatenate(ttgs),
      episode_id=np.concatenate(ep_ids),
      # 커밋 방향(+1/-1). 관측에는 안 들어가는 숨은 변수 — 다봉성 분석용 라벨.
      commit_side=np.concatenate(ep_sides),
      meta=dict(num_episodes=num_episodes, noise_std=noise_std,
                n_discard=n_discard, ep_cap=EP_CAP, seed0=seed0),
  )
  print(f'생성 완료: {len(data["action"])} transitions / {num_episodes} eps '
        f'(폐기 {n_discard}, {time.time() - t0:.0f}s)')
  return data


def her_augment(data, k=4, seed=0, success_radius=0.15):
  """HER(Hindsight Experience Replay)식 재라벨링 — 각 스텝 i마다 같은
  에피소드의 미래 위치 j를 k개 무작위로 뽑아 goal_pos를 positions[j]로 바꾼
  가짜 전이를 추가 생성한다(2026-07-21: 스킬 체이닝에서 임의 중간 지점을
  목표로 질의할 때의 캘리브레이션을 개선하기 위해 — 원래는 실제 목표만
  goal_pos로 학습돼 그런 질의가 분포 밖이었음).

  ttg 라벨은 단순히 j-i가 아니라 '반경 안에 처음 들어온 시점'으로 정확히
  계산한다(성공반경=0.15 안에서는 이미 도착한 것으로 침 — goal_substitution
  진단에서 확인한 경로 루프 시 라벨이 j-i와 크게 어긋나는 문제를 정확히
  반영하기 위함). action 라벨은 원래 액션을 그대로 두되(가짜 목표를 향한
  게 아니므로 의미 없음), 학습 시 bc loss는 이 샘플들에서 마스킹해 액션
  헤드를 오염시키지 않는다."""
  obs = data['observation']
  eid_all = data['episode_id']
  n_ep = int(eid_all.max()) + 1
  counts = np.bincount(eid_all, minlength=n_ep)
  starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
  rng = np.random.default_rng(seed)

  new = {k_: [] for k_ in OBS_FIELDS}
  new_act, new_ttg, new_eid = [], [], []
  t0 = time.time()
  for ep in range(n_ep):
    s, c = int(starts[ep]), int(counts[ep])
    if c < 2:
      continue
    pos = obs['cur_pos'][s:s + c]
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    inside = D < success_radius  # inside[m, j]: pos[m]이 pos[j]의 반경 안
    for i in range(c - 1):
      js = rng.integers(i + 1, c, size=k)  # 미래 스텝 중 k개 (복원추출)
      for j in js:
        col = inside[i:j + 1, j]
        m_rel = int(np.argmax(col))  # col[-1]은 항상 True(자기 자신)라 안전
        new['cur_pos'].append(pos[i])
        new['cur_vel'].append(obs['cur_vel'][s + i])
        new['goal_pos'].append(pos[j])
        new['obstacle_rel_pos'].append(obs['obstacle_rel_pos'][s + i])
        new['obstacle_radius'].append(obs['obstacle_radius'][s + i])
        new_act.append(data['action'][s + i])
        new_ttg.append(float(m_rel))
        new_eid.append(ep)
    if (ep + 1) % 2000 == 0:
      print(f'  HER 증강 {ep+1}/{n_ep} eps ({time.time()-t0:.0f}s)', flush=True)

  her_data = dict(
      observation={f: np.stack(v).astype(np.float32) for f, v in new.items()},
      action=np.stack(new_act).astype(np.float32),
      time_to_success=np.array(new_ttg, dtype=np.float32),
      episode_id=np.array(new_eid, dtype=np.int32),
  )
  print(f'HER 증강 완료: {len(new_ttg)} 가짜-목표 전이 추가 '
        f'(원본 {len(data["action"])}개의 약 {len(new_ttg)/len(data["action"]):.1f}배, '
        f'{time.time()-t0:.0f}s)')
  return her_data


# -------------------------------------------------------------- normalizers
def compute_stats(data):
  obs, act = data['observation'], data['action']
  def ms(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)
  stats = {}
  stats['cur_pos_mean'], stats['cur_pos_std'] = ms(obs['cur_pos'])
  stats['cur_vel_mean'], stats['cur_vel_std'] = ms(obs['cur_vel'])
  stats['obstacle_rel_pos_mean'], stats['obstacle_rel_pos_std'] = \
      ms(obs['obstacle_rel_pos'])
  stats['obstacle_radius_mean'], stats['obstacle_radius_std'] = \
      ms(obs['obstacle_radius'])
  stats['act_mean'], stats['act_std'] = ms(act)
  # 부분관측 환경의 추가 필드(있을 때만). 0/1 플래그라 정규화하면 정보가
  # 뭉개질 수 있어 항등(mean=0,std=1)으로 두고 원값을 그대로 쓴다.
  if 'obstacle_visible' in obs:
    stats['obstacle_visible_mean'] = np.zeros(1, dtype=np.float32)
    stats['obstacle_visible_std'] = np.ones(1, dtype=np.float32)
  return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def make_normalizers_obstacle(stats):
  """원본 make_normalizers의 5필드 확장. goal_pos는 원본과 동일하게 cur_pos
  통계를 공유한다(같은 좌표 공간)."""
  def normalize_obs(obs):
    out = {
        'cur_pos':
            (obs['cur_pos'] - stats['cur_pos_mean']) / stats['cur_pos_std'],
        'cur_vel':
            (obs['cur_vel'] - stats['cur_vel_mean']) / stats['cur_vel_std'],
        'goal_pos':
            (obs['goal_pos'] - stats['cur_pos_mean']) / stats['cur_pos_std'],
        'obstacle_rel_pos':
            (obs['obstacle_rel_pos'] - stats['obstacle_rel_pos_mean'])
            / stats['obstacle_rel_pos_std'],
        'obstacle_radius':
            (obs['obstacle_radius'] - stats['obstacle_radius_mean'])
            / stats['obstacle_radius_std'],
    }
    if 'obstacle_visible' in obs:   # 0/1 플래그는 원값 그대로
      out['obstacle_visible'] = obs['obstacle_visible']
    return out

  def normalize_action(a):
    return (a - stats['act_mean']) / stats['act_std']

  def unnormalize_action(a):
    return a * stats['act_std'] + stats['act_mean']

  return normalize_obs, normalize_action, unnormalize_action


def concat_obs(obs, fields=None):
  """기본은 기존 5필드(9차원). 부분관측 데이터처럼 obstacle_visible이 들어있으면
  자동으로 뒤에 덧붙인다(10차원) — 기존 체크포인트 경로의 동작은 그대로."""
  if fields is None:
    fields = OBS_FIELDS + (('obstacle_visible',)
                           if 'obstacle_visible' in obs else ())
  return jnp.concatenate([jnp.asarray(obs[k]) for k in fields], axis=-1)


# ------------------------------------------------------------------ training
class TrainState(NamedTuple):
  params: dict
  opt_state: optax.OptState


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=10000)
  ap.add_argument('--steps', type=int, default=32768)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval-episodes', type=int, default=100)
  ap.add_argument('--controller', choices=['pf', 'tangent', 'commit'],
                  default='pf',
                  help='pf=노이즈 PF(기존 채택값), tangent=노이즈 없는 접선점'
                       ' 조준(대조군, data/ckpt에 _clean 접미사),'
                       ' commit=에피소드 시작에 좌/우를 50:50 난수로 정해 끝까지'
                       ' 커밋(관측에 없는 숨은 변수 → 진짜 다봉 데이터)')
  ap.add_argument('--angle-jitter', type=float, default=0.0,
                  help='commit 컨트롤러에서 접선 각도에 줄 ±지터(라디안).'
                       ' 모드는 좌/우 이진으로 유지하면서 각 모드에 폭을 준다.')
  ap.add_argument('--tangent-noise-std', type=float, default=0.0,
                  help='controller=tangent일 때 demo_action 출력에 더할 '
                       '가우시안 액션 노이즈 표준편차(정책 궤적엔 거의 영향 '
                       '없는 수준의 소량 노이즈를 더해, 노이즈 크기 자체가 '
                       'aleatoric에 미치는 영향만 분리하려는 대조군용). '
                       '0이면 기존 완전 무노이즈 _clean과 동일 경로.')
  ap.add_argument('--her-k', type=int, default=0,
                  help='>0이면 HER 재라벨링(her_augment) 적용 — 각 스텝마다'
                       ' 같은 에피소드의 미래 위치 k개를 가짜 goal_pos로'
                       ' 추가 학습(중간 지점 질의 캘리브레이션 개선, 스킬'
                       ' 체이닝용). ckpt/캐시에 _her{k} 접미사가 붙는다.')
  ap.add_argument('--act-head', choices=['gaussian', 'mixture'],
                  default='gaussian',
                  help='gaussian=단일 가우시안(원본), mixture=K성분 가우시안'
                       ' 믹스처(갈림길에서 좌/우 커밋 가능). ckpt에 _mixK 접미사.')
  ap.add_argument('--n-mix', type=int, default=3,
                  help='act-head=mixture일 때 믹스처 성분 수 K')
  ap.add_argument('--env', choices=['full', 'partial'], default='full',
                  help='full=기존 완전관측, partial=센싱 반경 밖 장애물 미관측'
                       ' (obstacle_visible 플래그 추가 → 관측 10차원).'
                       ' partial은 block-prob 확률로만 경로를 막아 "막힐까"'
                       ' 자체가 관측 불가능한 진짜 이봉을 만든다.')
  ap.add_argument('--sensing-radius', type=float, default=0.25,
                  help='env=partial일 때 장애물이 보이기 시작하는 표면 거리')
  ap.add_argument('--block-prob', type=float, default=0.5,
                  help='env=partial일 때 장애물이 경로를 막을 확률')
  args = ap.parse_args()

  env_fn = None
  gen_obs_fields = None
  if args.env == 'partial':
    _sr, _bp = args.sensing_radius, args.block_prob
    env_fn = lambda: PartialObsObstacleAvoidPoint2D(
        sensing_radius=_sr, block_prob=_bp)
    gen_obs_fields = OBS_FIELDS + ('obstacle_visible',)

  if args.env == 'partial':
    suffix = f'_po_s{args.sensing_radius:g}_b{args.block_prob:g}'
  elif args.controller == 'pf':
    suffix = ''
  elif args.controller == 'commit':
    suffix = f'_commit_n{args.tangent_noise_std:g}'
    if args.angle_jitter > 0:
      suffix += f'_j{args.angle_jitter:g}'
  elif args.tangent_noise_std > 0:
    suffix = f'_clean_n{args.tangent_noise_std:g}'
  else:
    suffix = '_clean'
  data_path = f'data/obstacle_demos{suffix}.pkl'
  her_suffix = f'_her{args.her_k}' if args.her_k > 0 else ''
  head_suffix = f'_mix{args.n_mix}' if args.act_head == 'mixture' else ''
  her_data_path = f'data/obstacle_demos{suffix}{her_suffix}.pkl'
  ckpt_path = f'checkpoints/obstacle{suffix}{her_suffix}{head_suffix}/predictor.pkl'

  action_fn_factory = None
  if args.env == 'partial':
    action_fn = lambda obs: demo_action_partial(obs)
  elif args.controller == 'pf':
    action_fn = None
  elif args.controller == 'commit':
    action_fn = None
    _ns, _jit = args.tangent_noise_std, args.angle_jitter
    def action_fn_factory():
      # 에피소드 시작에 좌/우 50:50 결정 → 끝까지 커밋. side는 관측에 없음.
      side = 1.0 if np.random.uniform() < 0.5 else -1.0
      def fn(obs):
        a = demo_action_committed(obs, side=side, angle_jitter=_jit)
        if _ns > 0:
          a = a + np.random.normal(0, _ns, size=2).astype(np.float32)
        return np.asarray(a, dtype=np.float32)
      return fn, side
  elif args.tangent_noise_std > 0:
    noise_std = args.tangent_noise_std
    def action_fn(obs):
      return demo_action(obs) + np.random.normal(0, noise_std, size=2).astype(
          np.float32)
  else:
    action_fn = lambda obs: demo_action(obs)

  # ---- 1) 데이터 (있으면 재사용)
  if os.path.exists(data_path):
    print(f'기존 데이터셋 사용: {data_path}')
    with open(data_path, 'rb') as fp:
      data = pickle.load(fp)
  else:
    data = generate_demos(args.episodes, action_fn=action_fn,
                          action_fn_factory=action_fn_factory,
                          env_fn=env_fn, obs_fields=gen_obs_fields)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as fp:
      pickle.dump(data, fp)
    print(f'저장: {data_path}')
  N = len(data['action'])
  lengths = np.bincount(data['episode_id'])
  print(f'transitions={N}  ep길이 p50={np.percentile(lengths,50):.0f} '
        f'p99={np.percentile(lengths,99):.0f} max={lengths.max()}')
  assert lengths.max() <= MAX_DISTANCE, 'MAX_DISTANCE보다 긴 에피소드 존재'

  stats = compute_stats(data)  # 통계는 항상 원본(HER 목표분포로 안 흔들리게)

  is_her_col = np.zeros(N, dtype=bool)
  if args.her_k > 0:
    if os.path.exists(her_data_path):
      print(f'기존 HER 데이터셋 사용: {her_data_path}')
      with open(her_data_path, 'rb') as fp:
        her_data = pickle.load(fp)
    else:
      her_data = her_augment(data, k=args.her_k, seed=args.seed)
      with open(her_data_path, 'wb') as fp:
        pickle.dump(her_data, fp)
      print(f'저장: {her_data_path}')
    n_her = len(her_data['action'])
    data = dict(
        observation={f: np.concatenate(
            [data['observation'][f], her_data['observation'][f]], axis=0)
                     for f in OBS_FIELDS},
        action=np.concatenate([data['action'], her_data['action']], axis=0),
        time_to_success=np.concatenate(
            [data['time_to_success'], her_data['time_to_success']], axis=0),
        episode_id=np.concatenate(
            [data['episode_id'], her_data['episode_id']], axis=0),
    )
    is_her_col = np.concatenate(
        [np.zeros(N, dtype=bool), np.ones(n_her, dtype=bool)])
    print(f'HER 포함 총 transitions={len(data["action"])} '
          f'(원본 {N} + HER 가짜목표 {n_her})')

  # ---- 2) 정규화/분할 (held-out은 에피소드 단위 5%, HER 샘플도 원본과 같은
  # episode_id를 쓰므로 자동으로 같은 쪽에 들어감)
  normalize_obs, normalize_action, unnormalize_action = \
      make_normalizers_obstacle(stats)
  rng = np.random.default_rng(args.seed)
  n_ep = int(data['episode_id'].max()) + 1
  val_eps = set(rng.choice(n_ep, size=max(n_ep // 20, 1), replace=False).tolist())
  val_mask = np.isin(data['episode_id'], list(val_eps))

  obs_c = np.asarray(concat_obs(normalize_obs(data['observation'])))
  act_n = np.asarray(normalize_action(data['action']))
  ttg = data['time_to_success']
  # bc(액션) 손실 가중치 — HER 가짜목표 샘플은 0(마스킹), 원본은 1
  bcw = (~is_her_col).astype(np.float32)
  tr_obs = jnp.asarray(obs_c[~val_mask]); tr_act = jnp.asarray(act_n[~val_mask])
  tr_ttg = jnp.asarray(ttg[~val_mask]); tr_bcw = jnp.asarray(bcw[~val_mask])
  va_obs = jnp.asarray(obs_c[val_mask]); va_act = jnp.asarray(act_n[val_mask])
  va_ttg = jnp.asarray(ttg[val_mask]); va_bcw = jnp.asarray(bcw[val_mask])
  va_is_her = is_her_col[val_mask]
  print(f'train {tr_obs.shape[0]} / val {va_obs.shape[0]} transitions '
        f'(val {len(val_eps)} eps)')

  # ---- 3) 네트워크 (입력 9차원). act-head에 따라 단일 가우시안/믹스처 선택
  if args.act_head == 'mixture':
    nets = build_mixture_act_discrete_dist_v0(
        (256, 256, 256), 2, NUM_BINS,
        np.ones((4, tr_obs.shape[-1]), dtype=np.float32), n_mix=args.n_mix)
  else:
    nets = build_continuous_act_discrete_dist_v0(
        (256, 256, 256), 2, NUM_BINS,
        np.ones((4, tr_obs.shape[-1]), dtype=np.float32))
  dc = build_discrete_distance_converter(0, MAX_DISTANCE, NUM_BINS)
  bin_vals = np.linspace(0, MAX_DISTANCE, NUM_BINS + 1,
                         endpoint=True, dtype=np.float32)[:-1]
  optimizer = optax.adam(args.lr)

  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  params = nets.network.init(sub)
  state = TrainState(params, optimizer.init(params))

  def loss_fn(params, bo, ba, bt, bw):
    preds = nets.network.apply(params, bo)
    # bw=0인 샘플(HER 가짜목표)은 bc 손실에서 제외 — 가짜 목표를 향해 실제로
    # 움직인 게 아니므로 원래 액션 라벨이 그 목표에 대해 의미가 없음
    bc_per = -nets.act_log_prob(preds.act_dist_params, ba)
    bc = jnp.sum(bc_per * bw) / jnp.maximum(jnp.sum(bw), 1.0)
    # bin_size=1이므로 raw ttg가 곧 클래스 인덱스 (원본 loss와 동일 형태)
    dl = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, bt))
    return bc + dl, (bc, dl)

  @jax.jit
  def train_step(state, key):
    idx = jax.random.randint(key, (args.batch,), 0, tr_obs.shape[0])
    (l, (bc, dl)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, tr_obs[idx], tr_act[idx], tr_ttg[idx], tr_bcw[idx])
    updates, opt_state = optimizer.update(g, state.opt_state)
    return (TrainState(optax.apply_updates(state.params, updates), opt_state),
            l, bc, dl)

  @jax.jit
  def val_metrics(params):
    preds = nets.network.apply(params, va_obs)
    logits = preds.dist_to_succ_dist_params.logits
    exp = jnp.sum(jax.nn.softmax(logits) * bin_vals[None, :], axis=-1)
    mae = jnp.mean(jnp.abs(exp - va_ttg))
    nll = -jnp.mean(nets.dist_log_prob(preds.dist_to_succ_dist_params, va_ttg))
    bc_per = -nets.act_log_prob(preds.act_dist_params, va_act)
    bc = jnp.sum(bc_per * va_bcw) / jnp.maximum(jnp.sum(va_bcw), 1.0)
    return mae, nll, bc

  t0 = time.time()
  for step in range(1, args.steps + 1):
    key, sub = jax.random.split(key)
    state, l, bc, dl = train_step(state, sub)
    if step % 4096 == 0 or step == 1:
      mae, nll, vbc = val_metrics(state.params)
      print(f'step {step:6d}  loss={float(l):.3f} '
            f'(bc={float(bc):.3f} dist={float(dl):.3f})  '
            f'val: MAE={float(mae):.2f} NLL={float(nll):.3f} '
            f'bc={float(vbc):.3f}  ({time.time() - t0:.0f}s)', flush=True)

  # ---- 4) BC 정책 성공률 평가 (cpu jit — 원본 rollout과 동일 방식)
  def _policy(params, norm_concat, rng):
    preds = nets.network.apply(params, norm_concat)
    act = nets.sample_act(preds.act_dist_params, rng)
    return act
  policy = jax.jit(_policy, backend='cpu')

  succ, ep_lens = 0, []
  key_eval = jax.random.PRNGKey(1234)
  for e in range(args.eval_episodes):
    np.random.seed(90000 + e)
    env = env_fn() if env_fn is not None else ObstacleAvoidPoint2D()
    ts = env.reset()
    step = 0
    while not env.success() and step < EP_CAP:
      norm = normalize_obs(jax.tree.map(lambda x: np.asarray(x)[None],
                                        ts.observation))
      key_eval, sub = jax.random.split(key_eval)
      act = np.asarray(policy(state.params, np.asarray(concat_obs(norm)), sub))[0]
      ts = env.step(np.asarray(unnormalize_action(act), dtype=np.float32))
      step += 1
    succ += int(env.success())
    ep_lens.append(step)
  success_rate = succ / args.eval_episodes
  mae, nll, vbc = val_metrics(state.params)
  print(f'\n최종: 성공률 {success_rate:.2f} ({args.eval_episodes} eps, '
        f'길이 중앙값 {np.median(ep_lens):.0f})  '
        f'val MAE={float(mae):.2f} NLL={float(nll):.3f}')

  if args.her_k > 0:
    preds = nets.network.apply(state.params, va_obs)
    exp = jnp.sum(jax.nn.softmax(preds.dist_to_succ_dist_params.logits)
                  * bin_vals[None, :], axis=-1)
    abs_err = np.asarray(jnp.abs(exp - va_ttg))
    mae_orig = abs_err[~va_is_her].mean() if (~va_is_her).any() else float('nan')
    mae_her = abs_err[va_is_her].mean() if va_is_her.any() else float('nan')
    print(f'  (HER 분리) 원본목표 val MAE={mae_orig:.2f} ({(~va_is_her).sum()}개)  '
          f'가짜중간목표 val MAE={mae_her:.2f} ({va_is_her.sum()}개)')

  # ---- 5) 체크포인트
  os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
  with open(ckpt_path, 'wb') as fp:
    pickle.dump({
        'params': jax.device_get(state.params),
        'norm_stats': stats,
        'dc_config': {'min_distance': 0, 'max_distance': MAX_DISTANCE,
                      'num_bins': NUM_BINS},
        'obs_fields': list(gen_obs_fields or OBS_FIELDS),
        'obs_dim': int(tr_obs.shape[-1]),
        'meta': {
            'env': 'ObstacleAvoidPoint2D', 'controller': args.controller,
            'noise_std': (0.0 if args.controller == 'tangent' else 1.5e-4),
            'episodes': args.episodes, 'steps': args.steps,
            'seed': args.seed, 'her_k': args.her_k,
            'act_head': args.act_head, 'n_mix': args.n_mix,
            'env_mode': args.env, 'sensing_radius': args.sensing_radius,
            'block_prob': args.block_prob,
            'val_mae': float(mae), 'val_nll': float(nll),
            'policy_success_rate': success_rate,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        },
    }, fp)
  print(f'체크포인트 저장: {ckpt_path}')


if __name__ == '__main__':
  main()
