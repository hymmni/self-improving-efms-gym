"""DstgReward(d/보상/성공판정 래퍼) 검증. phases/5-mani-sim-ddpo/step2.md 참고.

무거운 policy_ckpt/run_config 로딩 없이 돌게, 작은 랜덤 DiffusionPolicyImage +
DstgPredictor를 즉석에서 만들고 DstgReward._from_predictor(테스트 전용 대안
생성자)로 감싼다 — step 1의 test_dstg_predictor.py와 같은 원칙.
"""

import torch

from mani_sim.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage
from mani_sim.policies.diffusion.dstg_predictor import DstgPredictor
from mani_sim.policies.diffusion.dstg_reward import DstgReward, _tail_cvar

DEVICE = "cpu"
IMAGE_HW = (32, 32)


def _make_policy(seed=0):
    torch.manual_seed(seed)
    policy = DiffusionPolicyImage(
        rgb_keys=["cam"],
        lowdim_keys=["state"],
        obs_dims={"state": 4},
        obs_horizon=2,
        action_dim=3,
        pred_horizon=4,
        num_kp=4,
        image_hw=IMAGE_HW,
        down_dims=(32, 64),
        num_train_timesteps=10,
        num_inference_steps=10,
    )
    policy.eval()
    return policy.to(DEVICE)


def _make_obs(batch_size=2, obs_horizon=2, device=DEVICE):
    return {
        "cam": torch.rand(batch_size, obs_horizon, 3, *IMAGE_HW, device=device),
        "state": torch.randn(batch_size, obs_horizon, 4, device=device),
    }


def _make_reward(num_bins=6, statistic="mean", cvar_alpha=0.8, threshold=None, seed=0):
    policy = _make_policy(seed=seed)
    predictor = DstgPredictor(policy, num_bins=num_bins, head_hidden=(16, 16)).to(DEVICE)
    return DstgReward._from_predictor(
        predictor, num_bins, statistic=statistic, cvar_alpha=cvar_alpha,
        threshold=threshold, device=DEVICE)


# ---------------------------------------------------------- 1. shape/monotone
def test_d_shapes_and_monotone_bins():
    reward = _make_reward(num_bins=6, statistic="mean")
    obs = _make_obs(batch_size=5)
    d = reward.d(obs)
    assert d.shape == (5,)

    bin_vals = torch.arange(5, dtype=torch.float32)
    probs_low = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])   # 전부 bin 0 (가깝다)
    probs_high = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])  # 전부 bin 4 (멀다)
    d_low = (probs_low * bin_vals).sum(dim=-1)
    d_high = (probs_high * bin_vals).sum(dim=-1)
    assert float(d_high[0]) > float(d_low[0])


# --------------------------------------------------------------- 2. cvar>=mean
def test_cvar_ge_mean():
    num_bins = 12
    torch.manual_seed(3)
    logits = torch.randn(9, num_bins)
    probs = torch.softmax(logits, dim=-1)
    bin_vals = torch.arange(num_bins, dtype=torch.float32)

    for alpha in (0.5, 0.8, 0.95):
        mean = (probs * bin_vals).sum(dim=-1)
        cvar = _tail_cvar(probs, bin_vals, alpha)
        assert torch.all(cvar >= mean - 1e-4), (alpha, mean, cvar)


# ---------------------------------------------------------------- 3. 얼려짐
def test_frozen():
    reward = _make_reward(num_bins=6, statistic="mean")
    params_before = [p.detach().clone() for p in reward.predictor.parameters()]

    obs = _make_obs(batch_size=4)
    d1 = reward.d(obs)
    d2 = reward.d(obs)
    torch.testing.assert_close(d1, d2)

    params_after = list(reward.predictor.parameters())
    for before, after in zip(params_before, params_after):
        torch.testing.assert_close(before, after)
        assert not after.requires_grad
