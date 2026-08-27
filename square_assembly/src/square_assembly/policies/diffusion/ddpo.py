"""DDPO(Black et al. 2023) 핵심 부품: PyTorch/diffusers DDPMScheduler 디퓨전 액션
정책의 단계별 로그확률.

왜 필요한가: SI-EFM Stage-2는 REINFORCE로 정책을 업데이트하는데, 그러려면
`log p(a|o)`가 필요하다. `DiffusionPolicyImage.predict_action_chunk()`의 역확산
100단계(DDPM ancestral sampling)를 적분해야 닫힌 형태의 `log p(a|o)`가 나온다 —
계산이 불가능하다.

DDPO의 우회는 `src/ddpo.py`(JAX 버전, phase 4 step 0)와 동일하다: 역확산의 각
단계는 파라미터에 의존하는 평균과 파라미터 독립적인 표준편차를 갖는 등방 가우시안
전이이므로, "체인 전체"가 아니라 "단계 하나하나"를 정책 그래디언트 대상으로 삼으면
각 단계의 로그확률은 정확히 계산된다. 체인 로그확률은 단계 로그확률의 합. 마지막
단계(diffusers 타임스텝 t=0)는 노이즈를 더하지 않는 결정론적 변환이라 확률밀도가
디랙 델타이므로 합에서 제외한다.

평균·분산 공식은 짐작이 아니라 설치된 `diffusers==0.30.3`의 실제 소스를 직접 읽고
재유도했다 (`DDPMScheduler.step()`, `DDPMScheduler._get_variance()`,
`DDPMScheduler.previous_timestep()` — `python -c "import diffusers, inspect;
print(inspect.getsource(...))"`로 확인). 기본 설정(`variance_type='fixed_small'`,
`clip_sample=True`, `timestep_spacing='leading'`, `prediction_type='epsilon'`)
기준:

  prev_t = t - (num_train_timesteps // num_inference_steps)
  alpha_prod_t = alphas_cumprod[t]
  alpha_prod_t_prev = alphas_cumprod[prev_t]  (prev_t < 0 이면 1.0)
  beta_prod_t = 1 - alpha_prod_t;  beta_prod_t_prev = 1 - alpha_prod_t_prev
  current_alpha_t = alpha_prod_t / alpha_prod_t_prev;  current_beta_t = 1 - current_alpha_t

  x0 = (x - sqrt(beta_prod_t) * eps) / sqrt(alpha_prod_t)
  x0 = clip(x0, -clip_sample_range, clip_sample_range)      # clip_sample=True

  mean = [sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t] * x0
       + [sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t] * x

  variance = clip((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t, min=1e-20)
  std = sqrt(variance)     # variance_type='fixed_small' 경로 (diffusers 기본값)
  # t == 0 인 스텝만 noise를 더하지 않음 (diffusers step()의 `if t > 0:` 분기)

이 모듈은 `DiffusionPolicyImage`(square task 기본값, DDPMScheduler 경로)만
대상으로 한다. `DiffusionPolicyLowDim`은 DDIMScheduler(eta=0, 결정론적)를 쓰는
별도 경로라 이 방식이 적용되지 않는다 — 범위 밖.

`diffusion_policy_image.py`/`conditional_unet1d.py`는 수정하지 않는다 — 정책
인스턴스에서 `unet`/`inference_scheduler`/`get_global_cond(obs)` 결과를 읽기만
한다(ADR-004/005 준용).
"""

import torch


def _ddpm_posterior(scheduler, x, eps, t):
    """`DDPMScheduler.step()`이 계산하는 posterior 평균과 표준편차를, 배치 내에서
    서로 다른 timestep을 섞어서도 계산할 수 있도록 벡터화한 버전.

    From: diffusers.schedulers.scheduling_ddpm.DDPMScheduler.step /
    DDPMScheduler._get_variance / DDPMScheduler.previous_timestep
    (설치된 diffusers==0.30.3 실제 소스를 읽고 재구현 — 모듈 docstring 참고)

    Args:
      x: (B, ...) 현재 latent (posterior의 조건 x_t).
      eps: (B, ...) `unet(x, t, global_cond)`의 노이즈 예측.
      t: (B,) long tensor. diffusers 타임스텝 값 그대로(0..num_train_timesteps-1).

    Returns:
      mean: x와 같은 shape.
      std: (B,) — 파라미터 독립 상수(등방 가우시안이라 전 차원에 공통).
    """
    device, dtype = x.device, x.dtype
    num_train_timesteps = scheduler.config.num_train_timesteps
    step_size = num_train_timesteps // scheduler.num_inference_steps

    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    one = alphas_cumprod.new_tensor(1.0)

    t = t.to(device=device, dtype=torch.long)
    prev_t = t - step_size

    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = torch.where(prev_t >= 0, alphas_cumprod[prev_t.clamp(min=0)], one)
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    variance = ((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t).clamp(min=1e-20)
    std = variance.sqrt()  # variance_type == "fixed_small" (diffusers 기본값)

    # broadcast (B,) -> x.shape for the elementwise arithmetic below
    extra = (1,) * (x.dim() - 1)
    ap_t, ap_tp, bp_t, bp_tp, ca_t, cb_t = (
        v.view(-1, *extra) for v in (alpha_prod_t, alpha_prod_t_prev, beta_prod_t, beta_prod_t_prev,
                                      current_alpha_t, current_beta_t)
    )

    x0 = (x - bp_t.sqrt() * eps) / ap_t.sqrt()
    if scheduler.config.clip_sample:
        x0 = x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    pred_original_sample_coeff = ap_tp.sqrt() * cb_t / bp_t
    current_sample_coeff = ca_t.sqrt() * bp_tp / bp_t
    mean = pred_original_sample_coeff * x0 + current_sample_coeff * x

    return mean, std


def _expand_t(t, batch_size):
    """0-d 텐서(배치 전체가 같은 t, `scheduler.timesteps`를 순회할 때)를 (B,)로 편다."""
    return t.expand(batch_size) if t.dim() == 0 else t


@torch.no_grad()
def sample_with_trace(unet, scheduler, global_cond, shape, device, generator=None):
    """`policy.predict_action_chunk()`와 **수치적으로 동일한** 샘플을 내되, 중간
    latent 전체를 함께 반환한다.

    `scheduler`는 호출 전에 `set_timesteps(...)`가 이미 적용된 상태여야 한다
    (`policy.inference_scheduler`를 그대로 넘기면, `predict_action_chunk` 호출이
    설정해둔 `.timesteps`/`.num_inference_steps`를 그대로 재사용한다).

    난수 소비 순서를 `predict_action_chunk`와 정확히 맞춘다: 초기 노이즈 1회 +
    매 역확산 스텝마다(마지막 t=0 제외) 노이즈 1회 — diffusers
    `DDPMScheduler.step()`의 `if t > 0:` 분기와 동일하게, t=0에서는 노이즈를
    추가로 뽑지 않는다.

    Args:
      shape: (B, pred_horizon, action_dim).
      generator: torch.Generator(device=device) 또는 None.

    Returns:
      sample_final: (B, Tp, Da)
      xs: (n_steps+1, B, Tp, Da) — xs[0]=초기 노이즈, xs[n_steps]=sample_final,
          xs[i]는 i번째 역확산 스텝 적용 직전 latent.
    """
    b = shape[0]
    x = torch.randn(shape, generator=generator, device=device)
    xs = [x]
    for t in scheduler.timesteps:
        eps = unet(x, t, global_cond)
        t_batch = _expand_t(t, b)
        mean, std = _ddpm_posterior(scheduler, x, eps, t_batch)
        if bool((t_batch > 0).all()):
            noise = torch.randn(x.shape, generator=generator, device=device)
            x = mean + std.view(-1, *([1] * (x.dim() - 1))) * noise
        else:
            x = mean
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return x, xs


def step_logp(unet, scheduler, global_cond, x_in, x_out, timesteps):
    """(env-step, 역확산-단계) 쌍의 미니배치에 대한 한 단계 로그확률.

    x_in, x_out: (B, Tp, Da). timesteps: (B,) long — diffusers 타임스텝 값 그대로.
    `t==0`인 샘플은 결정론적 전이라 로그확률이 정의되지 않으므로 호출자가 걸러서
    절대 넣지 말 것.

    unet 파라미터에 대해 미분 가능하다(grad 필요) — `torch.no_grad()`로 감싸지 않는다.
    """
    eps = unet(x_in, timesteps, global_cond)
    mean, std = _ddpm_posterior(scheduler, x_in, eps, timesteps)

    b = x_in.shape[0]
    diff = (x_out - mean).reshape(b, -1)
    d = diff.shape[-1]
    sq = (diff / std.unsqueeze(-1)) ** 2
    return -0.5 * sq.sum(dim=-1) - d * torch.log(std) - 0.5 * d * torch.log(
        torch.tensor(2.0 * torch.pi, device=std.device, dtype=std.dtype)
    )


def chain_logp(unet, scheduler, global_cond, xs, timesteps_all):
    """한 샘플의 체인 전체 로그확률 합(t=0 제외 전부). 진단·테스트용.

    xs: (n_steps+1, B, Tp, Da) — `sample_with_trace`가 반환한 것과 같은 규약.
    timesteps_all: `scheduler.timesteps`에서 t=0을 제외한 전체 목록(길이 n_steps-1,
    `xs[i]`--(kk번째 역확산 단계)-->`xs[i+1]` 순서와 일치).
    """
    b = xs.shape[1]
    total = xs.new_zeros(b)
    for i, t in enumerate(timesteps_all):
        t_batch = _expand_t(t, b)
        total = total + step_logp(unet, scheduler, global_cond, xs[i], xs[i + 1], t_batch)
    return total
