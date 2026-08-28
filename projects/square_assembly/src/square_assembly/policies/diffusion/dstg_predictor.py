"""d(o,g) := E[steps-to-go | o] 예측기 — 논문 식(1)의 관측-only distance-to-go.

square task는 관측이 카메라 이미지(84x84 RGB, agentview+wrist)라 비전 인코더가 필요하지만,
robomimic square PH 데모(~50개)는 STG 예측기 전용 ResNet18을 처음부터 학습시키기엔 데이터가
부족하다. 그래서 이미 학습된 diffusion policy 체크포인트의 비전 인코더
(DiffusionPolicyImage.encoders, ResNet18+SpatialSoftmax)를 얼려서 그대로 재사용하고, 그 위에
STG bin을 예측하는 작은 MLP 헤드만 새로 학습시킨다(사용자 결정 — phases/5-mani-sim-ddpo/step1.md).

액션은 입력에 없다 — 논문 식(1) `d(o,g) := E[steps-to-go | o, g]`는 관측(+목표)만 받는다.
"""

import torch
import torch.nn as nn


def _global_cond_dim(frozen_policy):
    """frozen_policy.unet 내부 구조에서 global_cond_dim을 역산한다(obs_dims를 별도로
    안 받아도 되게) — ConditionalResidualBlock1d.cond_encoder의 Linear.in_features가
    cond_dim(=diffusion_step_embed_dim + global_cond_dim)이고, diffusion_step_encoder의
    SinusoidalPosEmb.dim이 diffusion_step_embed_dim이다(networks/conditional_unet1d.py 참고)."""
    cond_dim = frozen_policy.unet.down_modules[0][0].cond_encoder[1].in_features
    diffusion_step_embed_dim = frozen_policy.unet.diffusion_step_encoder[0].dim
    return cond_dim - diffusion_step_embed_dim


class DstgPredictor(nn.Module):
    """얼린 diffusion policy의 비전 인코더(policy.encoders) + lowdim 특징을 그대로
    재사용해(policy.get_global_cond와 동일한 전처리) STG 카테고리컬 분포를 예측한다.
    액션은 입력에 없다(논문 식(1) 그대로).
    """

    def __init__(self, frozen_policy, num_bins, head_hidden=(256, 256)):
        super().__init__()
        self.frozen_policy = frozen_policy
        # requires_grad_(False) + forward()의 torch.no_grad() 이중 안전판(아래) — 인코더가
        # STG 헤드 학습으로 오염되지 않게 한다(step 0의 test_chain_logp_gradient_flows_to_unet_only
        # 와 같은 원칙).
        self.frozen_policy.requires_grad_(False)
        self.frozen_policy.eval()

        global_cond_dim = _global_cond_dim(frozen_policy)
        layers = []
        in_dim = global_cond_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_bins, bias=False))
        self.head = nn.Sequential(*layers)

    def forward(self, obs):
        """obs: DiffusionPolicyImage._encode가 받는 것과 같은 형식(dict of tensors).
        반환: (B, num_bins) logits.
        """
        with torch.no_grad():
            global_cond = self.frozen_policy.get_global_cond(obs)
        return self.head(global_cond)
