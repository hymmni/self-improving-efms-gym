r"""얼려진 DstgPredictor 체크포인트를 SI-EFM 식(1)~(3)의 보상/성공판정으로 감싼다.

배경(코드가 아니라 여기 남기는 이유: 이 파일의 존재 이유 자체가 이론적 근거이기
때문이다) — 논문 식(5)는 `r_t = d(o_t,g) - d(o_{t+1},g)`가 잠재함수 `Phi = -d`인
PBRS(Ng et al. 1999)임을 보인다. 통상 PBRS는 "정책 불변"이 보장되는데, 그 보장은
shaping이 더해지는 **원래 보상**이 있다는 전제에서 나온다. SI-EFM에는 더해질 원래
보상이 없다 — 식(5)를 전개하면 base reward 항조차 `-(1-gamma) d(o_{t+1},g)`로 `d`
자신이 만든다. 즉 여기서는 **`d`의 정의가 곧 목적함수 자체**다. `d`를 기댓값(mean)
에서 꼬리 위험(CVaR)으로 바꾸면 목적이 "기대 잔여 스텝 최소화"에서 "최악 구간
잔여 스텝 최소화"로 바뀐다 — 정책 불변이 아니라 정책이 달라지는 것이 기대된
결과다. 이 클래스는 그 정의(통계량)를 교체 가능한 부품으로 노출한다
(phases/4-diffusion-si/step2.md, src/carry_stg_reward.py — 같은 설계를 이 서브
프로젝트의 스택(PyTorch, DstgPredictor)에 맞게 이식).

step 1(`dstg_predictor.py`/`scripts/train_dstg.py`)이 학습한 체크포인트를 그대로
쓴다: {model: head.state_dict(), num_bins, policy_ckpt, obs_keys, head_hidden, ...}.
robomimic PH 데모는 전부 성공 시연이라 fail bin 개념이 없다(phase 4의
succ/fail 두 체크포인트와 달리, 여기서는 succ 버전 하나뿐).

파라미터는 절대 업데이트하지 않는다 (논문 Algorithm 1: "Initialize and freeze a
separate Stage 1 checkpoint for reward computation and success detection").

실행:
  python -m square_assembly.policies.diffusion.dstg_reward \
      --dstg-ckpt outputs/dstg/square_demo50_succ/predictor.pt \
      --hdf5-path /home/moai/hymm_ws/square_dataset/square_image_v15.hdf5 \
      [--statistic mean|cvar]
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
from square_assembly.datasets.robomimic_dataset import RobomimicSequenceDataset
from square_assembly.factory import registry
from square_assembly.policies.diffusion.dstg_predictor import DstgPredictor
from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config


def _tail_cvar(probs, bin_vals, alpha):
    """CVaR_alpha — 최악 (1-alpha) 구간(꼬리)만 재정규화한 기댓값.

    From: src/train_carry_qstg.py의 succ_cvar (jnp.cumsum/jnp.clip 겹침-질량 트릭)을
    torch.cumsum/torch.clip으로 그대로 이식했다 — 새로 유도하지 않는다. 이미 phase 4에서
    test_cvar_ge_mean 등으로 검증된 수학이다.
    """
    cdf = torch.cumsum(probs, dim=-1)
    cdf_prev = cdf - probs
    overlap = torch.clip(cdf, alpha, 1.0) - torch.clip(cdf_prev, alpha, 1.0)
    mass = overlap.sum(dim=-1)
    cvar = (overlap * bin_vals[None, :]).sum(dim=-1) / torch.clamp(mass, min=1e-6)
    return cvar


def _episode_split(dataset, val_fraction, seed):
    """무작위 에피소드(데모) 단위 분할.

    From: square_assembly/src/square_assembly/scripts/train_dstg.py의 _episode_split — 같은
    알고리즘을 그대로 다시 쓴다. 같은 (val_fraction, split_seed)를 주면 step 1
    학습이 실제로 held-out으로 남겨둔 것과 동일한 데모 집합이 재현된다. 이 파일에
    복제한 이유는 오직 train_dstg.py를 import했을 때 hydra 진입점(main())까지
    끌려오는 걸 피하기 위해서다 — train_dstg.py 자체는 수정하지 않는다(ADR-005).
    """
    demo_ids = [dataset._seq_dataset._index_to_demo_id[i] for i in range(len(dataset))]
    unique_demos = sorted(set(demo_ids))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_demos))
    n_val = max(1, int(round(len(unique_demos) * val_fraction)))
    val_demos = {unique_demos[i] for i in perm[:n_val]}
    train_idx = [i for i, d in enumerate(demo_ids) if d not in val_demos]
    val_idx = [i for i, d in enumerate(demo_ids) if d in val_demos]
    return train_idx, val_idx, len(unique_demos) - len(val_demos), len(val_demos)


class DstgReward:
    """얼려진 DstgPredictor 체크포인트를 논문 식(1)~(3)의 d/보상/성공판정으로 감싼다.

    파라미터는 절대 업데이트하지 않는다(논문 Algorithm 1: "freeze a separate Stage 1
    checkpoint for reward computation and success detection").
    """

    def __init__(self, dstg_ckpt_path, statistic="mean", cvar_alpha=0.8,
                 threshold=None, device="cpu"):
        """dstg_ckpt_path: step 1이 저장한 체크포인트(outputs/dstg/.../predictor.pt).
        그 안의 policy_ckpt 경로로 원본 diffusion policy(비전 인코더)도 함께 복원해야
        DstgPredictor.forward가 동작한다.
        statistic: "mean" | "cvar". threshold: None이면 calibrate_threshold()로 정해서
        넣어야 한다(생성자에서 자동으로 하지 않는다 — 뭘 근거로 캘리브레이션했는지
        호출부가 명시적으로 알게 하기 위함, phase 4 step 2와 동일 원칙).
        """
        device = torch.device(device)
        ckpt = torch.load(dstg_ckpt_path, map_location=device, weights_only=False)
        policy_ckpt_path = ckpt["policy_ckpt"]

        saved = load_run_config(policy_ckpt_path)
        if saved is None:
            raise ValueError(
                f"run_config.yaml을 {policy_ckpt_path} 옆에서 못 찾음 — 정책 아키텍처를 "
                "복원할 수 없다.")
        task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

        frozen_policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
        load_epoch_checkpoint(policy_ckpt_path, frozen_policy, device)
        frozen_policy.eval()

        predictor = DstgPredictor(frozen_policy, ckpt["num_bins"],
                                   head_hidden=tuple(ckpt["head_hidden"])).to(device)
        predictor.head.load_state_dict(ckpt["model"])

        self.policy_ckpt_path = policy_ckpt_path
        self.obs_keys = list(ckpt["obs_keys"])
        self.obs_horizon = int(policy_cfg.obs_horizon)
        self.rgb_keys = list(task_cfg.rgb_keys) if "rgb_keys" in task_cfg else []
        self._init_common(predictor, ckpt["num_bins"], statistic, cvar_alpha, threshold, device)

    @classmethod
    def _from_predictor(cls, predictor, num_bins, statistic="mean", cvar_alpha=0.8,
                         threshold=None, device="cpu"):
        """테스트 전용 대안 생성자 — 무거운 policy_ckpt/run_config 로딩 없이 이미 만든
        (작은 랜덤) DstgPredictor를 직접 감싼다(square_assembly/tests/test_dstg_reward.py).
        """
        self = cls.__new__(cls)
        self.policy_ckpt_path = None
        self.obs_keys = None
        self.obs_horizon = None
        self.rgb_keys = None
        self._init_common(predictor, num_bins, statistic, cvar_alpha, threshold, device)
        return self

    def _init_common(self, predictor, num_bins, statistic, cvar_alpha, threshold, device):
        if statistic not in ("mean", "cvar"):
            raise ValueError(f"statistic must be 'mean' or 'cvar', got {statistic!r}")
        self.device = torch.device(device)
        self.predictor = predictor.to(self.device)
        self.predictor.requires_grad_(False)
        self.predictor.eval()
        self.num_bins = int(num_bins)
        self.statistic = statistic
        self.cvar_alpha = cvar_alpha
        self.threshold = threshold
        self.bin_vals = torch.arange(self.num_bins, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def d(self, obs):
        """obs: DstgPredictor.forward가 받는 형식. 반환 (B,) — 값이 클수록 나쁘다
        (목표까지 멀다). 논문 식(1)."""
        logits = self.predictor(obs)
        probs = torch.softmax(logits, dim=-1)
        if self.statistic == "mean":
            return (probs * self.bin_vals).sum(dim=-1)
        return _tail_cvar(probs, self.bin_vals, self.cvar_alpha)

    @torch.no_grad()
    def success(self, obs):
        """식(3): 1[d(o) <= threshold]. 반환 (B,) bool."""
        if self.threshold is None:
            raise ValueError(
                "threshold가 설정되지 않았다 — calibrate_threshold()로 구한 뒤 "
                "reward.threshold = s 로 설정하라.")
        return self.d(obs) <= self.threshold


def calibrate_threshold(reward, dataset, val_indices, batch_size=64):
    """held-out(에피소드 단위 분할) 데이터에서 식(3)의 s를 정한다.

    라벨: 그 샘플이 "성공 시점"인가 — time_to_success == 0인 샘플(=데모의 마지막
    프레임)을 양성, 나머지를 음성으로 한다(get_time_to_success 라벨 재사용).
    방법: d(o) 관측 범위를 스캔해 F1이 최대가 되는 s를 고른다.
    반환: (best_s, metrics) — metrics에 precision/recall/f1/best_s.
    """
    labels_all = dataset.get_time_to_success()
    val_indices = list(val_indices)
    label = labels_all[np.asarray(val_indices)] == 0

    loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    d_vals = []
    for batch in loader:
        obs = {k: v.to(reward.device) for k, v in batch["obs"].items()}
        d_vals.append(reward.d(obs).cpu().numpy())
    d_vals = np.concatenate(d_vals)

    lo, hi = float(d_vals.min()), float(d_vals.max())
    s_candidates = np.linspace(lo, hi, 400)

    best = dict(f1=-1.0, s=float(s_candidates[0]), precision=0.0, recall=0.0)
    for s in s_candidates:
        pred = d_vals <= s
        tp = float(np.sum(pred & label))
        fp = float(np.sum(pred & ~label))
        fn = float(np.sum(~pred & label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        if f1 > best["f1"]:
            best = dict(f1=f1, s=float(s), precision=precision, recall=recall)

    metrics = dict(precision=best["precision"], recall=best["recall"],
                   f1=best["f1"], best_s=best["s"])
    return best["s"], metrics


# ---------------------------------------------------------------- 검증 스크립트
def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dstg-ckpt", required=True,
                     help="outputs/dstg/square_demo50_succ/predictor.pt")
    ap.add_argument("--hdf5-path", required=True)
    ap.add_argument("--statistic", choices=["mean", "cvar"], default="mean")
    ap.add_argument("--cvar-alpha", type=float, default=0.8)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    reward = DstgReward(args.dstg_ckpt, statistic=args.statistic,
                         cvar_alpha=args.cvar_alpha, device=device)
    print(f"체크포인트: {args.dstg_ckpt}")
    print(f"  policy_ckpt={reward.policy_ckpt_path}")
    print(f"  num_bins={reward.num_bins}  statistic={args.statistic}"
          + (f"  cvar_alpha={args.cvar_alpha}" if args.statistic == "cvar" else ""))

    normalizer = MinMaxNormalizer(load_stats(
        os.path.join(os.path.dirname(reward.policy_ckpt_path), "normalization_stats.json")))
    dataset = RobomimicSequenceDataset(
        hdf5_path=args.hdf5_path, obs_keys=reward.obs_keys, obs_horizon=reward.obs_horizon,
        pred_horizon=1, normalizer=normalizer, rgb_keys=reward.rgb_keys,
    )
    train_idx, val_idx, n_train_demos, n_val_demos = _episode_split(
        dataset, args.val_fraction, args.split_seed)
    print(f"\nheld-out split: train={n_train_demos} demos/{len(train_idx)} samples  "
          f"val={n_val_demos} demos/{len(val_idx)} samples (seed={args.split_seed})")

    # ---- 1. 문턱 캘리브레이션 -------------------------------------------------
    best_s, metrics = calibrate_threshold(reward, dataset, val_idx, batch_size=args.batch_size)
    reward.threshold = best_s
    print(f"\n[1] 문턱 캘리브레이션 (held-out {n_val_demos}개 데모, {len(val_idx)}개 샘플)")
    print(f"  s={metrics['best_s']:.3f}  precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")

    # ---- 2. d 프로파일 sanity check (held-out 성공 에피소드 1개) --------------
    demo_ids = [dataset._seq_dataset._index_to_demo_id[i] for i in range(len(dataset))]
    val_demo_ids = sorted({demo_ids[i] for i in val_idx})
    sample_demo = val_demo_ids[0]
    demo_idxs = [i for i in val_idx if demo_ids[i] == sample_demo]
    demo_idxs.sort(key=lambda i: dataset._demo_id_and_index_in_demo(i)[1])

    loader = DataLoader(Subset(dataset, demo_idxs), batch_size=len(demo_idxs), shuffle=False)
    batch = next(iter(loader))
    obs = {k: v.to(reward.device) for k, v in batch["obs"].items()}
    d_trace = reward.d(obs).cpu().numpy()

    print(f"\n[2] d 프로파일 sanity check (held-out 데모={sample_demo}, "
          f"{len(d_trace)}스텝, 10스텝 간격)")
    shown = d_trace[::10]
    print("  " + "  ".join(f"t={i * 10}:{v:.2f}" for i, v in enumerate(shown)))
    diffs = np.diff(d_trace)
    frac_decreasing = float(np.mean(diffs <= 0)) if len(diffs) > 0 else float("nan")
    monotone = frac_decreasing >= 0.7
    print(f"  스텝간 비증가 비율={frac_decreasing:.1%}  대체로 단조감소={monotone}"
          + ("" if monotone else "  [경고: d가 단조 감소하지 않음 — 보상이 노이즈일 수 있음]"))


if __name__ == "__main__":
    main()
