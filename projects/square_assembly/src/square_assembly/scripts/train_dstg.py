"""DstgPredictor(d(o,g) := E[steps-to-go | o]) 학습 진입점 — square task, PH(성공만) 데모.

scripts/train.py 패턴을 따르되(hydra 진입점), task/policy는 hydra defaults가 아니라
policy_ckpt 옆 run_config.yaml에서 직접 복원한다(그 체크포인트가 실제로 학습된 설정이
유일한 출처 — scripts/eval.py의 _apply_run_config와 동일한 이유).

hdf5_path는 명시적으로 오버라이드해야 한다 — run_config.yaml에 박힌 학습 당시 상대경로는
이 머신에서 그대로 안 열린다(phases/5-mani-sim-ddpo/step1.md 참고).

robomimic PH 데모는 전부 성공 시연이라 이 스크립트는 succ 버전만 만든다 — fail-aware
버전(실패 bin 포함)은 실패 롤아웃 데이터가 쌓인 뒤의 다음 phase로 미룬다.

사용:
    python -m square_assembly.scripts.train_dstg \
        policy_ckpt=/path/to/policy_epoch1060.pt \
        hdf5_path=/path/to/square_image_v15.hdf5 \
        out=outputs/dstg/square_demo50_succ/predictor.pt
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
    """RobomimicSequenceDataset의 한 샘플에 time_to_success 라벨을 붙이는 얇은 어댑터
    (robomimic_dataset.py는 안 건드림 — get_time_to_success()는 전체 라벨을 한 번에 반환하는
    별도 메서드이지, __getitem__에 라벨을 얹는 게 아니다, ADR-005)."""

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
    """무작위 에피소드(데모) 단위 분할 — 같은 데모의 프레임이 train/val에 걸쳐 섞이지
    않게 한다(무작위 transition 분할이면 인접 프레임 유출로 val이 낙관적으로 나온다)."""
    # _index_to_demo_id는 {sample_index: demo_id} dict — list()로 감싸면 키(=0..N-1)만
    # 나오는 함정이 있어(demo_id별 그룹핑이 아니라 사실상 무작위 transition 분할이 되어버림),
    # 반드시 .values()로 값(demo_id)을 순서대로 꺼내야 한다.
    demo_ids = [dataset._seq_dataset._index_to_demo_id[i] for i in range(len(dataset))]
    unique_demos = sorted(set(demo_ids))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_demos))
    n_val = max(1, int(round(len(unique_demos) * val_fraction)))
    val_demos = {unique_demos[i] for i in perm[:n_val]}
    train_idx = [i for i, d in enumerate(demo_ids) if d not in val_demos]
    val_idx = [i for i, d in enumerate(demo_ids) if d in val_demos]
    return train_idx, val_idx, len(unique_demos) - len(val_demos), len(val_demos)


def _run_epoch(predictor, loader, device, num_bins, optimizer=None, epoch_label=""):
    train = optimizer is not None
    predictor.train(train)  # frozen_policy도 같이 토글됨 — VisionEncoder crop_hw가
    # self.training으로 RandomCrop/CenterCrop을 고르므로 train=RandomCrop, eval=CenterCrop
    # (원본 diffusion policy 학습과 동일한 augmentation 관례, encoders 자체 가중치는
    # requires_grad_(False)로 이미 얼어 있어 이 토글이 학습에 영향 없음).

    total_nll, total_abs_err, n = 0.0, 0.0, 0
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
            expected = (F.softmax(logits, dim=-1) * bin_idx).sum(dim=-1)
            total_abs_err += (expected - labels.float()).abs().sum().item()
            total_nll += loss.item() * labels.shape[0]
            n += labels.shape[0]

    return total_nll / n, total_abs_err / n


@hydra.main(config_path="../configs", config_name="train_dstg", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    saved = load_run_config(cfg.policy_ckpt)
    if saved is None:
        raise ValueError(f"run_config.yaml을 {cfg.policy_ckpt} 옆에서 못 찾음 — 학습 당시 설정 없이는 정책 아키텍처를 복원할 수 없다.")
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    OmegaConf.set_struct(task_cfg, False)
    task_cfg.hdf5_path = cfg.hdf5_path  # run_config.yaml의 상대경로는 이 머신에서 안 열림
    OmegaConf.set_struct(task_cfg, True)

    frozen_policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(cfg.policy_ckpt, frozen_policy, device)
    frozen_policy.eval()
    logger.info(f"frozen policy 복원: {cfg.policy_ckpt} (task={task_cfg.name} policy_name={policy_name})")

    stats_path = os.path.join(os.path.dirname(cfg.policy_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    obs_keys = task_obs_keys(task_cfg)
    cache_mode = "all" if cfg.num_workers >= 1 else "low_dim"  # h5py fork 크래시 회피(diffusion_trainer.py와 동일)
    dataset = RobomimicSequenceDataset(
        hdf5_path=cfg.hdf5_path, obs_keys=obs_keys, obs_horizon=policy_cfg.obs_horizon,
        pred_horizon=1,  # STG는 이 시점 관측 하나에 대한 라벨이지 행동 청크가 아니다
        normalizer=normalizer, rgb_keys=task_cfg.rgb_keys, hdf5_cache_mode=cache_mode,
    )
    labels = dataset.get_time_to_success()
    # 기본은 데이터에서 관측된 최대 time-to-success로 정한다(고정 상수 금지, 원래 방침).
    # cfg.num_bins_override(기본 None)가 있으면 그 값을 강제로 쓴다 — 2026-08-11: 데이터
    # 스케일링 비교 실험처럼 "여러 체크포인트가 같은 bin 공간을 써야 mu/sigma를 그대로
    # 비교할 수 있는" 상황 전용. 그 외엔 기본값(None)을 유지해 기존 동작 그대로 둔다.
    if cfg.get("num_bins_override"):
      num_bins = int(cfg.num_bins_override)
      if num_bins <= int(labels.max()):
        raise ValueError(f"num_bins_override({num_bins})가 실제 관측된 최대 time-to-success"
                         f"({int(labels.max())})보다 작거나 같다 — 라벨이 잘린다.")
    else:
      num_bins = int(labels.max()) + 1
    logger.info(f"dataset len={len(dataset)} num_bins={num_bins} (max observed time-to-success={int(labels.max())})")

    train_idx, val_idx, n_train_demos, n_val_demos = _episode_split(dataset, cfg.val_fraction, cfg.split_seed)
    logger.info(
        f"episode split: train={n_train_demos} demos/{len(train_idx)} samples, "
        f"val={n_val_demos} demos/{len(val_idx)} samples"
    )

    train_loader = DataLoader(
        _LabeledWindow(dataset, labels, train_idx), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
    )
    val_loader = DataLoader(
        _LabeledWindow(dataset, labels, val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers >= 1,
    )

    predictor = DstgPredictor(frozen_policy, num_bins, head_hidden=tuple(cfg.head_hidden)).to(device)
    # 이중 안전판: requires_grad_(False)(DstgPredictor 생성자) + 옵티마이저 파라미터 그룹을
    # head로만 좁힘(frozen_policy 전체를 넘기지 않음).
    optimizer = torch.optim.AdamW(predictor.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        train_nll, train_mae = _run_epoch(predictor, train_loader, device, num_bins, optimizer=optimizer,
                                          epoch_label=f"epoch {epoch}/{cfg.num_epochs}")
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            msg = f"epoch {epoch} train_nll={train_nll:.4f} train_mae={train_mae:.3f}"
            logger.info(msg)
            print(msg, flush=True)

    val_nll, val_mae = _run_epoch(predictor, val_loader, device, num_bins, optimizer=None, epoch_label="[val]")
    logger.info(f"[val] nll={val_nll:.4f} mae={val_mae:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)")
    print({"val_nll": val_nll, "val_mae": val_mae, "num_bins": num_bins,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos})

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    torch.save({
        "model": predictor.head.state_dict(),  # frozen_policy는 policy_ckpt에서 다시 로드하는 게 정책
        "epoch": cfg.num_epochs,
        "num_bins": num_bins,
        "policy_ckpt": os.path.abspath(cfg.policy_ckpt),
        "obs_keys": obs_keys,
        "head_hidden": list(cfg.head_hidden),
        "val_nll": val_nll,
        "val_mae": val_mae,
    }, cfg.out)
    logger.info(f"saved: {cfg.out}")


if __name__ == "__main__":
    main()
