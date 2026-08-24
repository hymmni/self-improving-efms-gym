r"""train_si.py가 저장한 자기개선 정책 체크포인트(predictor.pt / predictor_best.pt)를
실제로 굴려서 mp4로 저장한다.

기본은 `--hires`(robomimic EnvRobosuite.render(mode="rgb_array", camera_name="agentview")
로 매 스텝 별도 오프스크린 렌더)로 고화질 프레임을 뽑는다 — 정책이 실제로 쓰는 84x84
관측과는 별개의 렌더 호출이라 정책 입력과 화질이 다르다는 점만 유의(행동/궤적 자체는
동일 롤아웃이라 같다). `--hires 0`이면 정책이 실제로 보는 84x84 관측을 그대로(4배
업샘플만) 써서, "정책이 뭘 보고 저 행동을 했는지"를 보여줄 때 쓴다.

    python -m mani_sim.scripts.record_si_video \
        --si-ckpt outputs/si_full_env/predictor_best.pt \
        --base-ckpt /home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt \
        --episodes 3 --hires 480 --out outputs/si_full_env/rollout_best_hires.mp4
"""

import argparse
import os

import numpy as np
import torch
import imageio.v2 as imageio

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.factory import registry
from mani_sim.runners.intervention_rollout import _predict_chunk, collect_episode
from mani_sim.utils.checkpoints import load_run_config
from mani_sim.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

CAMERA_KEY = "agentview_image"
UPSCALE = 4


def run(si_ckpt, base_ckpt, episodes, max_steps, out, fps, hires):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = load_run_config(base_ckpt)
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    payload = torch.load(si_ckpt, map_location=device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    print(f'로드: {si_ckpt} (epoch={payload.get("epoch")}, '
          f'base={payload.get("si_base_ckpt")})')

    stats_path = os.path.join(os.path.dirname(base_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []

    env = make_eval_env(task_cfg)

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    frames = []
    successes = []

    def _to_video_frame(img):
        # env.step()/reset()이 돌려주는 raw obs의 이미지 키는 robomimic ObsUtils.process_obs가
        # 이미 CHW,float32,[0,1]로 바꿔둔 상태다(HWC 원본이 아님) — HWC로 되돌려야 imageio가
        # 정상적인 (H,W,3) 프레임으로 받는다.
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(np.repeat(img, UPSCALE, axis=0), UPSCALE, axis=1)

    for ep in range(episodes):
        capture_ep = []

        def capture_and_track(obs_raw, _store=capture_ep):
            if hires > 0:
                frame = env.render(mode="rgb_array", height=hires, width=hires, camera_name="agentview")
                frame = np.ascontiguousarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                _store.append(frame)
            else:
                _store.append(_to_video_frame(obs_raw[CAMERA_KEY]))
            return True

        # collect_episode(배포용 롤아웃 루프, intervention_rollout.py)를 재사용한다 —
        # 그래디언트/보상 계산이 전혀 필요 없고 단순 롤아웃+영상 캡처만 필요해서
        # train_si.py의 collect_episode_si(SI 학습 전용)가 아니라 이걸 쓴다.
        result = collect_episode(
            env, policy, normalizer, obs_keys,
            task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon, device,
            intervention_fn=lambda step, obs: None,
            max_steps=max_steps, render=False, render_fn=capture_and_track,
            predict_fn=predict_fn, print_diagnostics=False,
        )
        successes.append(result["success"])
        frames.extend(capture_ep)
        frames.extend([capture_ep[-1]] * fps)   # 에피소드 끝 1초 정지
        print(f'ep {ep}: steps={len(result["actions"])} success={result["success"]}')

    env.close() if hasattr(env, "close") else None
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    print(f'성공 {sum(successes)}/{episodes}  saved {out} ({len(frames)} frames)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--si-ckpt", required=True)
    ap.add_argument("--base-ckpt", default="/home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hires", type=int, default=480,
                    help="0이면 정책의 84x84 실제 관측 사용, N>0이면 NxN 별도 오프스크린 렌더")
    ap.add_argument("--out", default="outputs/si_video.mp4")
    args = ap.parse_args()
    run(args.si_ckpt, args.base_ckpt, args.episodes, args.max_steps, args.out, args.fps, args.hires)


if __name__ == "__main__":
    main()
