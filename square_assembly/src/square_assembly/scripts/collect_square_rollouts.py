r"""기존 정책(policy_epoch1060.pt)을 실제로 굴려서 성공+실패가 섞인 롤아웃을
robomimic 호환 HDF5로 저장한다. 2D(GraspCarry2D)에서 했던 것과 동일한 목적 —
fail-aware STG 예측기를 학습시키려면 실패 상태를 실제로 본 데이터가 필요한데,
`square_image_v15.hdf5`(PH 시연)는 전부 성공이라 실패가 아예 없다.

## HDF5 스키마 (robomimic.utils.dataset.SequenceDataset 소스 직접 확인, 2026-08-10)
`SequenceDataset.load_demo_info()`가 실제로 읽는 건 `data/{demo}.attrs['num_samples']`
(데모 길이)와 `data/{demo}/obs/{obs_keys 각각}`, `data/{demo}/{dataset_keys 각각}`
(우리는 dataset_keys=("actions",)만 씀 — RobomimicSequenceDataset 기본 extra_keys=()라
action_mode 등은 안 읽음)뿐이다. dones/rewards/states/model_file/camera_info/
stage_onehot/object는 원본 시연 hdf5엔 있지만 우리 로더 경로에선 안 읽으므로 저장
안 한다(불필요한 무거운 필드 — 특히 states/model_file은 시뮬레이터 재현용이라
우리 목적엔 안 씀).

이미지는 원본 hdf5와 동일하게 **HWC uint8**로 저장한다(`RobomimicSequenceDataset.
__getitem__`이 로드 시점에 permute+/255.0을 하므로) — `env.step()`이 돌려주는
obs_raw의 이미지 키는 robomimic ObsUtils.process_obs가 이미 CHW,float[0,1]로
바꿔둔 상태라 저장 전에 역변환(HWC,uint8)해야 한다.

    python -m square_assembly.scripts.collect_square_rollouts \
        --base-ckpt /home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt \
        --episodes 120 --max-steps 500 \
        --out data/square_rollouts_v1.hdf5
"""

import argparse
import os

import h5py
import numpy as np
import torch

from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
from square_assembly.factory import registry
from square_assembly.runners.intervention_rollout import _predict_chunk, collect_episode
from square_assembly.utils.checkpoints import load_run_config
from square_assembly.utils.task_utils import is_image_task, make_eval_env, task_obs_keys


def _to_storage(key, val, rgb_keys):
    val = np.asarray(val)
    if key in rgb_keys:
        # CHW,float[0,1] (env 출력) -> HWC,uint8 (원본 hdf5/로더가 기대하는 저장 형식)
        if val.ndim == 3 and val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[-1]:
            val = np.transpose(val, (1, 2, 0))
        val = np.clip(val * 255.0, 0, 255).astype(np.uint8)
    return val


def run(base_ckpt, episodes, max_steps, seed0, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = load_run_config(base_ckpt)
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    from square_assembly.utils.checkpoints import load_epoch_checkpoint
    load_epoch_checkpoint(base_ckpt, policy, device)
    policy.eval()

    stats_path = os.path.join(os.path.dirname(base_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []

    env = make_eval_env(task_cfg)

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    outcomes = {"success": 0, "fail": 0}

    with h5py.File(out, "w") as f:
        data_grp = f.create_group("data")
        total = 0
        for ep in range(episodes):
            obs_ep = []

            def track(obs_raw, _store=obs_ep):
                _store.append({k: _to_storage(k, obs_raw[k], rgb_keys) for k in obs_keys})
                return True

            result = collect_episode(
                env, policy, normalizer, obs_keys,
                task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon, device,
                intervention_fn=lambda step, obs: None,
                max_steps=max_steps, render=False, render_fn=track,
                predict_fn=predict_fn, print_diagnostics=False,
            )
            actions = np.asarray(result["actions"], dtype=np.float64)
            T = len(actions)
            assert T == len(obs_ep), f"obs/action 스텝 수 불일치: {len(obs_ep)} vs {T}"

            demo_grp = data_grp.create_group(f"demo_{ep}")
            demo_grp.attrs["num_samples"] = T
            demo_grp.create_dataset("actions", data=actions)
            obs_grp = demo_grp.create_group("obs")
            for k in obs_keys:
                stacked = np.stack([o[k] for o in obs_ep], axis=0)
                obs_grp.create_dataset(k, data=stacked, compression="gzip" if k in rgb_keys else None)

            is_success = bool(result["success"])
            demo_grp.attrs["is_success"] = is_success
            outcomes["success" if is_success else "fail"] += 1
            total += T
            print(f"ep {ep}: steps={T} success={is_success}  "
                  f"누적 성공 {outcomes['success']}/{ep + 1} ({outcomes['success'] / (ep + 1):.1%})",
                  flush=True)

        data_grp.attrs["total"] = total

    env.close() if hasattr(env, "close") else None
    print(f"\n수집 완료: {episodes}에피소드 (성공 {outcomes['success']}, 실패 {outcomes['fail']})  "
          f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-ckpt", default="/home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="data/square_rollouts_v1.hdf5")
    args = ap.parse_args()
    run(args.base_ckpt, args.episodes, args.max_steps, args.seed0, args.out)


if __name__ == "__main__":
    main()
