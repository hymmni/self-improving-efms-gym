"""DstgPredictor(d(o,g) := E[steps-to-go | o]) fail-aware 학습 — 성공+실패가 섞인 롤아웃
hdf5(`collect_square_rollouts.py` 산출물) 대상.

`train_dstg.py`(PH 성공만) 복사본에 실패 라벨링만 얹었다 — 원본은 안 건드림(ADR-005 원칙,
"기존 벤치마크/파이프라인 코드는 필요한 만큼만 얇게 감싸고 원본은 보존"과 동일하게 이
스크립트도 새 파일로 분리).

## 라벨링 방식 (class mode — 2D GraspCarry2D `train_carry_dstg.py --fail-mode class`와
동일한 가장 단순한 방식, deadline/부트스트랩 등 정교한 버전은 안 씀 — "빠른 효과 확인"이
목적이라 우선순위 낮음)
- 성공 데모의 transition: 기존 `get_time_to_success()`(L-1-t) 그대로.
- 실패 데모의 transition: **전부 동일한 별도 클래스(fail_bin)** — "몇 스텝 뒤에 실패했나"가
  아니라 "이 상태는 결국 실패로 간다"는 것만 배운다(2D에서도 처음엔 실패까지 남은 스텝을
  라벨링했다가 "실패까지 남은 스텝 ≠ 성공까지 남은 스텝(같은 상태에서도 다른 롤아웃이면
  성공할 수 있다)"는 범주 오류로 지적받아 고쳤던 것과 같은 이유로, 실패 transition에
  아무 스텝 라벨이나 붙이는 것 자체가 근거 없다 — 그래서 "실패 클래스"라는 별도 범주로만
  다룬다).
- fail_bin = (성공 라벨 중 최댓값) + 1, NUM_BINS = fail_bin + 1.

사용:
    python -m square_assembly.scripts.train_dstg_failaware \
        policy_ckpt=/path/to/policy_epoch1060.pt \
        hdf5_path=data/square_rollouts_v1.hdf5 \
        out=outputs/dstg/square_failaware/predictor.pt
"""

import logging
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
from square_assembly.datasets.robomimic_dataset import RobomimicSequenceDataset
from square_assembly.factory import registry
from square_assembly.policies.diffusion.dstg_predictor import DstgPredictor
from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config
from square_assembly.utils.task_utils import task_obs_keys

logger = logging.getLogger(__name__)


class _LabeledWindow(Dataset):
    def __init__(self, base, labels, indices):
        self.base = base
        self.labels = labels
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = self.base[idx]
        item["time_to_success"] = int(self.labels[idx])
        return item


def _episode_split(dataset, val_fraction, seed):
    """train_dstg.py::_episode_split과 동일(그대로 복사, 원본 안 건드림 원칙 유지)."""
    demo_ids = [dataset._seq_dataset._index_to_demo_id[i] for i in range(len(dataset))]
    unique_demos = sorted(set(demo_ids))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_demos))
    n_val = max(1, int(round(len(unique_demos) * val_fraction)))
    val_demos = {unique_demos[i] for i in perm[:n_val]}
    train_idx = [i for i, d in enumerate(demo_ids) if d not in val_demos]
    val_idx = [i for i, d in enumerate(demo_ids) if d in val_demos]
    return train_idx, val_idx, len(unique_demos) - len(val_demos), len(val_demos)


def _failaware_labels(dataset):
    """성공 transition은 기존 get_time_to_success(), 실패 transition은 전부 fail_bin.

    demo별 성공 여부는 `collect_square_rollouts.py`가 저장한 `data/{demo}.attrs['is_success']`
    를 직접 읽는다(robomimic_dataset.py에 새 메서드를 얹지 않고 여기서 한 번만 씀 — 이
    파이프라인 전용 라벨이라 공용 어댑터를 오염시키지 않기 위함)."""
    seq = dataset._seq_dataset
    ttg = dataset.get_time_to_success()
    demo_ids = np.asarray([seq._index_to_demo_id[i] for i in range(len(dataset))])

    succ_map = {d: bool(seq.hdf5_file[f"data/{d}"].attrs["is_success"]) for d in seq.demos}
    is_succ_sample = np.asarray([succ_map[d] for d in demo_ids], dtype=bool)

    succ_ttg = ttg[is_succ_sample]
    if len(succ_ttg) == 0:
        raise ValueError("성공 데모가 하나도 없다 — fail_bin을 정할 기준(성공 라벨 최댓값)이 없음.")
    fail_bin = int(succ_ttg.max()) + 1
    num_bins = fail_bin + 1

    labels = ttg.copy()
    labels[~is_succ_sample] = fail_bin
    return labels, is_succ_sample, fail_bin, num_bins


def _run_epoch(predictor, loader, device, num_bins, fail_bin, optimizer=None, epoch_label=""):
    train = optimizer is not None
    predictor.train(train)

    total_nll, total_abs_err, n = 0.0, 0.0, 0
    fail_tp, fail_fn, fail_fp, fail_tn = 0, 0, 0, 0
    bin_idx = torch.arange(num_bins, device=device, dtype=torch.float32)
    n_batches = len(loader)
    for b_i, raw_batch in enumerate(loader):
        if n_batches > 20 and (b_i % max(1, n_batches // 10) == 0 or b_i == n_batches - 1):
            print(f"  {epoch_label} 배치 {b_i + 1}/{n_batches}", flush=True)
        obs = {k: v.to(device) for k, v in raw_batch["obs"].items()}
        labels = raw_batch["time_to_success"].to(device).long()

        logits = predictor(obs)
        loss = F.cross_entropy(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            expected = (probs * bin_idx).sum(dim=-1)
            # 실패 클래스는 카테고리형이라 기댓값(스텝 수) 계산에 안 넣는다 — 성공
            # transition에서만 MAE를 잰다(train_dstg.py의 succ-only MAE와 비교 가능하게).
            is_succ_batch = labels != fail_bin
            if is_succ_batch.any():
                total_abs_err += (expected[is_succ_batch] - labels[is_succ_batch].float()).abs().sum().item()
            total_nll += loss.item() * labels.shape[0]
            n += labels.shape[0]

            pred_fail = torch.argmax(logits, dim=-1) == fail_bin
            actual_fail = labels == fail_bin
            fail_tp += int((pred_fail & actual_fail).sum())
            fail_fp += int((pred_fail & ~actual_fail).sum())
            fail_fn += int((~pred_fail & actual_fail).sum())
            fail_tn += int((~pred_fail & ~actual_fail).sum())

    mae = total_abs_err / max((n - fail_tp - fail_fn), 1)  # 성공 표본 수로 나눔(근사)
    precision = fail_tp / max(fail_tp + fail_fp, 1)
    recall = fail_tp / max(fail_tp + fail_fn, 1)
    return total_nll / n, mae, precision, recall


@hydra.main(config_path="../configs", config_name="train_dstg_failaware", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    saved = load_run_config(cfg.policy_ckpt)
    if saved is None:
        raise ValueError(f"run_config.yaml을 {cfg.policy_ckpt} 옆에서 못 찾음.")
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    OmegaConf.set_struct(task_cfg, False)
    task_cfg.hdf5_path = cfg.hdf5_path
    OmegaConf.set_struct(task_cfg, True)

    frozen_policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(cfg.policy_ckpt, frozen_policy, device)
    frozen_policy.eval()
    logger.info(f"frozen policy 복원: {cfg.policy_ckpt}")

    stats_path = os.path.join(os.path.dirname(cfg.policy_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    obs_keys = task_obs_keys(task_cfg)
    cache_mode = "all" if cfg.num_workers >= 1 else "low_dim"
    dataset = RobomimicSequenceDataset(
        hdf5_path=cfg.hdf5_path, obs_keys=obs_keys, obs_horizon=policy_cfg.obs_horizon,
        pred_horizon=1, normalizer=normalizer, rgb_keys=task_cfg.rgb_keys, hdf5_cache_mode=cache_mode,
    )
    labels, is_succ_sample, fail_bin, num_bins = _failaware_labels(dataset)
    logger.info(f"dataset len={len(dataset)} (성공 transition {int(is_succ_sample.sum())}, "
                f"실패 {int((~is_succ_sample).sum())})  fail_bin={fail_bin}  num_bins={num_bins}")

    train_idx, val_idx, n_train_demos, n_val_demos = _episode_split(dataset, cfg.val_fraction, cfg.split_seed)
    logger.info(f"episode split: train={n_train_demos} demos/{len(train_idx)} samples, "
                f"val={n_val_demos} demos/{len(val_idx)} samples")

    train_loader = DataLoader(
        _LabeledWindow(dataset, labels, train_idx), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
    )
    val_loader = DataLoader(
        _LabeledWindow(dataset, labels, val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers >= 1,
    )

    predictor = DstgPredictor(frozen_policy, num_bins, head_hidden=tuple(cfg.head_hidden)).to(device)
    optimizer = torch.optim.AdamW(predictor.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        train_nll, train_mae, train_p, train_r = _run_epoch(
            predictor, train_loader, device, num_bins, fail_bin, optimizer=optimizer,
            epoch_label=f"epoch {epoch}/{cfg.num_epochs}")
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            msg = (f"epoch {epoch} train_nll={train_nll:.4f} train_mae={train_mae:.3f} "
                   f"fail_precision={train_p:.3f} fail_recall={train_r:.3f}")
            logger.info(msg)
            print(msg, flush=True)

    val_nll, val_mae, val_p, val_r = _run_epoch(
        predictor, val_loader, device, num_bins, fail_bin, optimizer=None, epoch_label="[val]")
    logger.info(f"[val] nll={val_nll:.4f} mae={val_mae:.3f} fail_precision={val_p:.3f} "
                f"fail_recall={val_r:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)")
    print({"val_nll": val_nll, "val_mae": val_mae, "val_fail_precision": val_p, "val_fail_recall": val_r,
           "num_bins": num_bins, "fail_bin": fail_bin,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos})

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    torch.save({
        "model": predictor.head.state_dict(),
        "epoch": cfg.num_epochs,
        "num_bins": num_bins,
        "fail_bin": fail_bin,
        "policy_ckpt": os.path.abspath(cfg.policy_ckpt),
        "obs_keys": obs_keys,
        "head_hidden": list(cfg.head_hidden),
        "val_nll": val_nll,
        "val_mae": val_mae,
        "val_fail_precision": val_p,
        "val_fail_recall": val_r,
    }, cfg.out)
    logger.info(f"saved: {cfg.out}")


if __name__ == "__main__":
    main()
