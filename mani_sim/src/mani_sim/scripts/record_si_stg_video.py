r"""`record_si_video.py`(장면만)에 STG 카테고리컬 분포 막대그래프를 옆에 붙인 버전.
2D의 `record_carry_stg_dist.py`/`record_carry_si_video.py`와 같은 화면 구성을 3D에
이식한 것이다.

롤아웃 루프는 직접 재구현하지 않고 `intervention_rollout.collect_episode`(배포/평가에서
이미 검증된 함수, `record_si_video.py`/`collect_square_rollouts.py`가 쓰는 것과 동일)를
그대로 재사용한다 — 처음엔 STG 계산을 끼워 넣으려고 루프를 직접 다시 짰었는데, 그
재구현판이 왜인지 성공률이 비정상적으로 낮게 나와서(8+4=12에피소드 전부 실패, 실제
학습 중 성공률 46.7%와 안 맞음) 원인을 못 찾고 이 방식으로 되돌렸다(2026-08-10).
`render_fn(obs_raw)` 콜백은 collect_episode 내부에서 `obs_history.append(obs_raw)`
**직후**에 호출되므로, 콜백 안에서 똑같은 길이의 obs_history를 별도로 유지하면
STG 계산에 필요한 관측 이력을 안전하게 재현할 수 있다(메인 롤아웃 로직은 안 건드림).

    python -m mani_sim.scripts.record_si_stg_video \
        --si-ckpt outputs/si_full_env/predictor_best.pt \
        --base-ckpt /home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt \
        --dstg-ckpt outputs/dstg/square_failaware/predictor.pt \
        --episodes 2 --hires 480 --out outputs/si_full_env/rollout_best_stg.mp4
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.policies.diffusion.dstg_reward import DstgReward
from mani_sim.runners.intervention_rollout import _predict_chunk, collect_episode
from mani_sim.runners.rollout import _build_obs_batch
from mani_sim.utils.checkpoints import load_run_config
from mani_sim.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run(si_ckpt, base_ckpt, dstg_ckpt, episodes, max_steps, out, fps, hires, xmax):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = load_run_config(base_ckpt)
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    payload = torch.load(si_ckpt, map_location=device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    print(f'로드: {si_ckpt} (epoch={payload.get("epoch")})')

    reward = DstgReward(dstg_ckpt, statistic="mean", device=device)
    fail_bin = reward.num_bins - 1  # train_dstg_failaware.py 관례(fail_bin=num_bins-1)
    if xmax is None:
        xmax = reward.num_bins

    stats_path = os.path.join(os.path.dirname(base_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []
    obs_horizon = task_cfg.get("obs_horizon", policy_cfg.obs_horizon)

    env = make_eval_env(task_cfg)

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    bin_vals = np.arange(reward.num_bins)
    fig, (ax_scene, ax_dist) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), dpi=110, gridspec_kw={"width_ratios": [1, 1.3]})
    frames = []
    successes = []

    for ep in range(episodes):
        # collect_episode가 내부에서 관리하는 obs_history와 별개로, render_fn 안에서
        # 쓸 STG 전용 obs_history를 동일 규칙(초기 obs_horizon개는 첫 프레임 반복)으로
        # 병행 유지한다. collect_episode는 obs_history.append(obs_raw) 직후에 render_fn을
        # 부르므로, 이 콜백 쪽 deque도 매 호출마다 한 번씩만 append하면 항상 동기화된다.
        stg_obs_history = {"dq": None}
        capture_ep = []

        def render_fn(obs_raw, _stg=stg_obs_history, _store=capture_ep):
            if _stg["dq"] is None:
                _stg["dq"] = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)
            else:
                _stg["dq"].append(obs_raw)

            # torch.random.fork_rng로 감싼다 — STG용 추가 forward pass가 전역 RNG
            # 상태를 건드리면(2026-08-10 실측: 이 호출을 넣기만 해도 6/6 에피소드가
            # 전부 실패로 바뀌었다 — collect_episode 자체는 안 건드렸는데도), 그
            # 이후 정책의 디퓨전 노이즈 샘플링 시퀀스가 통째로 달라져 롤아웃이
            # 나비효과로 딴 길로 샌다. fork_rng는 이 블록 안에서 RNG를 얼마든지
            # 소모해도 블록 밖(정책 샘플링)에는 전혀 영향이 없게 저장/복원해준다.
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices, enabled=True):
                with torch.no_grad():
                    obs_batch = _build_obs_batch(_stg["dq"], obs_keys, rgb_keys, normalizer, device, None)
                    logits = reward.predictor(obs_batch)
                    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            mu = float((probs * bin_vals).sum())

            if hires > 0:
                frame = env.render(mode="rgb_array", height=hires, width=hires, camera_name="agentview")
            else:
                frame = np.asarray(obs_raw["agentview_image"])
                frame = np.transpose(frame, (1, 2, 0))
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

            ax_scene.clear()
            ax_scene.imshow(np.asarray(frame))
            ax_scene.axis("off")
            ax_scene.set_title(f"ep={ep} t={len(_store)}")

            ax_dist.clear()
            xshow = min(xmax, reward.num_bins)
            ax_dist.bar(bin_vals[:xshow], probs[:xshow], width=1.0, color="tab:purple", alpha=0.8)
            if fail_bin < xshow:
                ax_dist.axvline(fail_bin, color="tab:red", ls="--", lw=1.5, label=f"fail_bin={fail_bin}")
            ax_dist.axvline(mu, color="black", lw=1.2, label=f"μ={mu:.1f}")
            ax_dist.set_xlim(0, xshow)
            ax_dist.set_ylim(0, max(0.05, float(probs[:xshow].max()) * 1.25))
            ax_dist.set_xlabel("STG")
            ax_dist.set_ylabel("Probability")
            ax_dist.set_title(f"STG Categorical Distribution  μ={mu:.1f}")
            ax_dist.legend(loc="upper right", fontsize=8)

            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = fig.get_size_inches() * fig.get_dpi()
            _store.append(np.frombuffer(canvas.buffer_rgba(), np.uint8)
                          .reshape(int(h), int(w), 4)[..., :3].copy())
            return True

        result = collect_episode(
            env, policy, normalizer, obs_keys,
            obs_horizon, policy_cfg.action_horizon, device,
            intervention_fn=lambda step, obs: None,
            max_steps=max_steps, render=False, render_fn=render_fn,
            predict_fn=predict_fn, print_diagnostics=False,
        )
        successes.append(result["success"])
        frames.extend(capture_ep)
        frames.extend([capture_ep[-1]] * fps)
        print(f'ep {ep}: steps={len(result["actions"])} success={result["success"]}')

    env.close() if hasattr(env, "close") else None
    plt.close(fig)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    print(f"성공 {sum(successes)}/{episodes}  saved {out} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--si-ckpt", required=True)
    ap.add_argument("--base-ckpt", default="/home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt")
    ap.add_argument("--dstg-ckpt", default="outputs/dstg/square_failaware/predictor.pt")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--hires", type=int, default=480)
    ap.add_argument("--xmax", type=int, default=None)
    ap.add_argument("--out", default="outputs/si_stg_video.mp4")
    args = ap.parse_args()
    run(args.si_ckpt, args.base_ckpt, args.dstg_ckpt, args.episodes, args.max_steps,
        args.out, args.fps, args.hires, args.xmax)


if __name__ == "__main__":
    main()
