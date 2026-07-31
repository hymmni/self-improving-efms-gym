"""사전학습 LeRobot Diffusion Policy를 로드해 gym-pusht에서 굴리는 헬퍼.

이 파일은 **lerobot conda 환경**에서만 쓴다(torch/lerobot/gym-pusht 필요).
STG 예측기 학습(JAX, emfs-gym 환경)과는 별개 프로세스로 롤아웃 데이터를 뽑는 용도.

왜 수동 로드인가:
  lerobot 0.6.0은 정규화를 정책 밖 processor 파이프라인으로 분리했다. 그래서
  구버전 체크포인트(lerobot/diffusion_pusht_keypoints)의 config를 그대로 로드하면
  스키마 불일치로 실패하고, 정책 객체는 순수 정규화 공간에서만 동작한다
  (normalize_inputs/unnormalize_outputs 모듈이 없음). 따라서:
    - 입력 관측(픽셀 좌표)을 min-max로 [-1,1] 정규화해서 먹이고,
    - 출력 액션([-1,1])을 픽셀로 역정규화해야 한다.
  이 두 단계를 빠뜨리면 도달률 0%(coverage≈0.12, 랜덤보다도 낮음)가 나온다.
  둘 다 적용하면 50 에피소드 도달률 66%로 공식 published 71%를 재현한다.
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
import safetensors.torch as st

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

REPO_ID = 'lerobot/diffusion_pusht_keypoints'


class DPPushT:
  """로드된 정책 + min-max 정규화 통계 래퍼. `.act(obs)`로 픽셀 액션을 낸다."""

  def __init__(self, repo_id=REPO_ID, device='cuda'):
    sd = st.load_file(hf_hub_download(repo_id, 'model.safetensors'))
    self.device = device
    # 체크포인트에 박제된 min/max 통계
    self.E_min = sd['normalize_inputs.buffer_observation_environment_state.min'].numpy()
    self.E_max = sd['normalize_inputs.buffer_observation_environment_state.max'].numpy()
    self.S_min = sd['normalize_inputs.buffer_observation_state.min'].numpy()
    self.S_max = sd['normalize_inputs.buffer_observation_state.max'].numpy()
    self.A_min = sd['normalize_targets.buffer_action.min'].numpy()
    self.A_max = sd['normalize_targets.buffer_action.max'].numpy()

    cfg = DiffusionConfig(
        n_obs_steps=2, horizon=16, n_action_steps=8,
        down_dims=(512, 1024, 2048), diffusion_step_embed_dim=128,
        kernel_size=5, n_groups=8, num_train_timesteps=100,
        num_inference_steps=10, noise_scheduler_type='DDIM',
        beta_schedule='squaredcos_cap_v2', clip_sample=True,
        prediction_type='epsilon', use_film_scale_modulation=True,
        use_group_norm=True, crop_shape=None, vision_backbone='resnet18')
    cfg.input_features = {
        'observation.environment_state': PolicyFeature(FeatureType.ENV, (16,)),
        'observation.state': PolicyFeature(FeatureType.STATE, (2,))}
    cfg.output_features = {'action': PolicyFeature(FeatureType.ACTION, (2,))}
    cfg.normalization_mapping = {
        'ENV': NormalizationMode.MIN_MAX, 'STATE': NormalizationMode.MIN_MAX,
        'ACTION': NormalizationMode.MIN_MAX, 'VISUAL': NormalizationMode.MEAN_STD}

    pol = DiffusionPolicy(cfg)
    # 정규화 버퍼 키는 이 정책엔 없으므로(processor로 분리) 무시하고 로드
    pol.load_state_dict({k: v for k, v in sd.items() if 'normaliz' not in k},
                        strict=False)
    self.pol = pol.eval().to(device)

  @staticmethod
  def _nrm(x, mn, mx):
    return (2 * (x - mn) / (mx - mn) - 1).astype(np.float32)   # 픽셀 -> [-1,1]

  @staticmethod
  def _unn(x, mn, mx):
    return ((x + 1) / 2 * (mx - mn) + mn).astype(np.float32)   # [-1,1] -> 픽셀

  def reset(self):
    self.pol.reset()

  def act(self, obs):
    """obs: gym-pusht environment_state_agent_pos dict. 반환: 픽셀 액션 (2,)."""
    es = self._nrm(np.asarray(obs['environment_state'], np.float32), self.E_min, self.E_max)
    ap = self._nrm(np.asarray(obs['agent_pos'], np.float32), self.S_min, self.S_max)
    batch = {
        'observation.environment_state': torch.tensor(es)[None].to(self.device),
        'observation.state': torch.tensor(ap)[None].to(self.device)}
    with torch.no_grad():
      a = self.pol.select_action(batch).cpu().numpy()[0]
    return self._unn(a, self.A_min, self.A_max)


class DPLocal:
  """로컬 lerobot 체크포인트(`.../pretrained_model` 디렉토리) 로더.

  로컬 저장본은 정규화 processor(pre/post)를 함께 저장하므로, 구 HF 모델용
  DPPushT의 수동 min-max 정규화와 달리 표준 로드+추론 경로를 그대로 쓴다:
    pre(obs) -> select_action -> post(action).  `.act(obs)`가 픽셀 액션을 낸다.
  config.json의 draccus 판별자(type) 때문에 DiffusionConfig.from_pretrained는
  실패하므로 베이스 PreTrainedConfig.from_pretrained로 로드한다.
  """

  def __init__(self, ckpt_dir, device='cuda'):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies import make_pre_post_processors
    self.device = device
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    self.pol = DiffusionPolicy.from_pretrained(ckpt_dir).eval().to(device)
    self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=ckpt_dir)

  def reset(self):
    self.pol.reset()

  def act(self, obs):
    es = torch.tensor(np.asarray(obs['environment_state'], np.float32))[None]
    ap = torch.tensor(np.asarray(obs['agent_pos'], np.float32))[None]
    batch = {'observation.environment_state': es.to(self.device),
             'observation.state': ap.to(self.device)}
    batch = self.pre(batch)
    with torch.no_grad():
      a = self.pol.select_action(batch)
    a = self.post(a)
    return a.cpu().numpy()[0].astype(np.float32)


def _make_policy(policy, ckpt, device):
  """policy 객체가 주어지면 그대로, ckpt 경로면 DPLocal, 아니면 DPPushT(HF)."""
  if policy is not None:
    return policy
  if ckpt:
    return DPLocal(ckpt, device=device)
  return DPPushT(device=device)


def collect_rollouts(n_episodes=300, out_path='data/pusht_dp_rollouts.pkl',
                     seed0=200000, device='cuda', max_steps=299,
                     success_coverage=0.8, policy=None, ckpt=None):
  """DP 정책 롤아웃을 굴려 STG 학습용 데이터로 저장.

  인간 데모(data/pusht_demos.pkl)와 **동일 포맷·동일 라벨 정의**:
    - 성공(coverage>=success_coverage 최초 도달) 에피소드만 저장.
    - 성공 시점에서 궤적을 자른다.
    - time_to_success[t] = (성공시점 - t),  성공 상태에서 0.
    - observation={agent_pos(2), env_state(16)}, action=픽셀좌표(2).
  각 상태의 coverage는 그 상태(관측)에 정렬해 기록한다.
  """
  import os
  import pickle
  import gymnasium as gym
  import gym_pusht  # noqa: F401
  dp = _make_policy(policy, ckpt, device)
  env = gym.make('gym_pusht/PushT-v0', obs_type='environment_state_agent_pos')

  ap_all, es_all, act_all, ttg_all, eid_all, cov_all = [], [], [], [], [], []
  kept = 0
  for e in range(n_episodes):
    obs, info = env.reset(seed=seed0 + e)
    dp.reset()
    cov = float(info.get('coverage', 0.0))          # 현재 관측의 coverage
    ap_l, es_l, act_l, cov_l = [], [], [], []
    succ_idx = -1
    for n in range(max_steps + 1):
      a = dp.act(obs).astype(np.float32)             # 이 상태에서의 액션
      ap_l.append(np.asarray(obs['agent_pos'], np.float32))
      es_l.append(np.asarray(obs['environment_state'], np.float32))
      act_l.append(a)
      cov_l.append(cov)
      if cov >= success_coverage:                    # 성공 상태 포함 후 종료
        succ_idx = n
        break
      obs, r, term, trunc, info = env.step(a)
      cov = float(info.get('coverage', r))
      if term or trunc:
        break
    if succ_idx < 0:
      continue                                       # 미성공 -> 제외 (논문과 동일)
    L = succ_idx + 1
    ttg = (L - 1 - np.arange(L)).astype(np.float32)
    ap_all.append(np.array(ap_l))
    es_all.append(np.array(es_l))
    act_all.append(np.array(act_l))
    cov_all.append(np.array(cov_l, np.float32))
    ttg_all.append(ttg)
    eid_all.append(np.full(L, kept, np.int32))
    kept += 1
  env.close()

  data = {
      'observation': {'agent_pos': np.concatenate(ap_all),
                      'env_state': np.concatenate(es_all)},
      'action': np.concatenate(act_all),
      'time_to_success': np.concatenate(ttg_all),
      'episode_id': np.concatenate(eid_all),
      'coverage': np.concatenate(cov_all),
      'meta': {'source': 'src.dp_policy DP rollouts', 'policy': ckpt or REPO_ID,
               'success_threshold': success_coverage, 'num_episodes': kept,
               'requested_episodes': n_episodes, 'seed0': seed0, 'fps': 10},
  }
  assert data['time_to_success'].max() < 300, 'STG 라벨이 bin 범위(300) 초과'
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  with open(out_path, 'wb') as fp:
    pickle.dump(data, fp)
  print(f'saved {out_path}: 성공 {kept}/{n_episodes} 에피소드, '
        f'{len(data["time_to_success"])} 스텝, '
        f'ttg max={data["time_to_success"].max():.0f}')
  return data


def evaluate(n_episodes=50, seed0=100000, device='cuda', max_steps=300,
             policy=None, ckpt=None, success_coverage=None):
  """롤아웃 평가. 반환: dict(success_rate, coverages).
  success_coverage 지정 시 그 커버리지 도달을 성공으로 집계(기본은 env is_success)."""
  import gymnasium as gym
  import gym_pusht  # noqa: F401
  dp = _make_policy(policy, ckpt, device)
  env = gym.make('gym_pusht/PushT-v0', obs_type='environment_state_agent_pos')
  succ = 0
  covs = []
  for e in range(n_episodes):
    obs, info = env.reset(seed=seed0 + e)
    dp.reset()
    done = False
    n = 0
    best = 0.0
    hit = False
    while not done and n < max_steps:
      obs, r, term, trunc, info = env.step(dp.act(obs).astype(np.float32))
      cov = float(info.get('coverage', r))
      best = max(best, cov)
      n += 1
      done = term or trunc
      if info.get('is_success', False) or (
          success_coverage is not None and cov >= success_coverage):
        hit = True
        if success_coverage is not None:
          done = True
    covs.append(best)
    succ += int(hit)
  env.close()
  covs = np.array(covs)
  return {'success_rate': succ / n_episodes, 'coverages': covs,
          'cov_mean': float(covs.mean()), 'cov_median': float(np.median(covs))}


if __name__ == '__main__':
  import argparse
  import os
  os.environ.setdefault('MUJOCO_GL', 'egl')
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', nargs='?', default='eval',
                  choices=['eval', 'collect'])
  ap.add_argument('--episodes', type=int, default=None)
  ap.add_argument('--out', default='data/pusht_dp_rollouts.pkl')
  ap.add_argument('--seed0', type=int, default=None)
  ap.add_argument('--ckpt', default=None,
                  help='로컬 lerobot 체크포인트(.../pretrained_model). 없으면 HF 사전학습.')
  args = ap.parse_args()
  if args.mode == 'collect':
    kw = dict(out_path=args.out, ckpt=args.ckpt)
    if args.episodes is not None:
      kw['n_episodes'] = args.episodes
    if args.seed0 is not None:
      kw['seed0'] = args.seed0
    collect_rollouts(**kw)
  else:
    res = evaluate(n_episodes=args.episodes or 50, ckpt=args.ckpt,
                   success_coverage=0.8)
    print(f"success_rate={res['success_rate']:.2f}  "
          f"cov_mean={res['cov_mean']:.3f}  cov_median={res['cov_median']:.3f}")
