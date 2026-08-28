"""train_si.py의 순수 함수/구조 검증 — 실제 robosuite 환경·체크포인트 없이 돈다.

phases/4-diffusion-si/step3.md의 test_si_loop.py(JAX, GraspCarry2D)와 같은 정신:
env/정책 무거운 의존성 없이도 리턴 계산, "환경 신호가 보상에 안 새는지", "옵티마이저가
인코더를 제외하는지"를 각각 독립적으로 검증한다.
"""

import inspect

import numpy as np
import torch

from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage
from square_assembly.scripts.train_si import compute_decision_reward, compute_returns

DEVICE = "cpu"
IMAGE_HW = (32, 32)


def _make_policy():
    torch.manual_seed(0)
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


def test_returns_discounting():
    # d_vals: 한 에피소드의 결정 시작/끝 관측에서 잰 d(o) 값들(가짜). rewards[t] =
    # d_vals[t] - d_vals[t+1] (compute_decision_reward과 동일한 식 2를 직접 대입).
    d_vals = np.array([9.0, 7.0, 6.0, 2.0, 0.0], dtype=np.float32)
    rewards = np.array(
        [compute_decision_reward(d_vals[t], d_vals[t + 1]) for t in range(len(d_vals) - 1)],
        dtype=np.float32,
    )

    gamma = 0.9
    returns = compute_returns(rewards, gamma)
    assert returns.shape == rewards.shape

    # R_t = sum_{i>=t} gamma^(i-t) r_i 를 순진하게(직접) 계산해 대조.
    t_len = len(rewards)
    expected = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        expected[t] = sum(gamma ** (i - t) * rewards[i] for i in range(t, t_len))
    np.testing.assert_allclose(returns, expected, atol=1e-5)

    # gamma=1이면 텔레스코핑으로 R_0 = d_vals[0] - d_vals[-1] (중간 항이 전부 상쇄됨).
    returns_g1 = compute_returns(rewards, gamma=1.0)
    assert abs(returns_g1[0] - (d_vals[0] - d_vals[-1])) < 1e-4


def test_reward_fn_has_no_ground_truth():
    """compute_decision_reward의 시그니처/구현에 env의 is_success/done/info가 흘러들지
    않는지 확인 — SI-EFM의 핵심 경계("외부 감독 없는 자기개선")가 이 함수에 걸려 있다."""
    sig = inspect.signature(compute_decision_reward)
    assert list(sig.parameters.keys()) == ["d_before", "d_after"]

    source = inspect.getsource(compute_decision_reward)
    forbidden = ["is_success", "done", "info", "outcome", "terminated", "truncated"]
    for token in forbidden:
        assert token not in source, f"compute_decision_reward에 금지된 토큰 {token!r}이 등장함"


def test_optimizer_excludes_encoder():
    """옵티마이저에 넘기는 파라미터 목록에 policy.encoders의 파라미터가 하나도 없어야 한다."""
    policy = _make_policy()
    optimizer = torch.optim.Adam(policy.unet.parameters(), lr=1e-6)

    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    encoder_ids = {id(p) for p in policy.encoders.parameters()}
    unet_ids = {id(p) for p in policy.unet.parameters()}

    assert optimized_ids == unet_ids
    assert optimized_ids.isdisjoint(encoder_ids)
    assert len(encoder_ids) > 0  # 인코더 자체가 비어있지 않은지(테스트가 공허하게 통과하지 않게)
