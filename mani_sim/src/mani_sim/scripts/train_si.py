r"""SI-EFM Stage-2(Algorithm 1) — DDPO-SF로 square task 디퓨전 정책을 자기개선한다.

이 스크립트는 phase 4(GraspCarry2D, JAX)의 `train_carry_si.py`를 이 서브프로젝트의
스택(PyTorch, robomimic square)으로 이식한 것이다. 설계 원리는 원본과 동일하다:

  1. Stage-1 체크포인트를 하나 복사해 얼린다(`DstgReward` — 보상/성공 판정 전용).
  2. 현재 정책으로 on-policy 롤아웃을 모은다.
  3. r = d(o_before,g) - d(o_after,g)  (식 2)
  4. R_i = sum_{j>=i} gamma^(j-i) r_j
  5. REINFORCE: loss = -reinforce_scale * mean(R * log p(a|o,g))
  6. 그 배치는 한 번 쓰고 버린다(off-policy·부트스트래핑을 피하는 설계).

GraspCarry2D와의 핵심 차이는 "결정"의 단위다 — square task 정책은 receding-horizon
청크 실행(pred_horizon=16, action_horizon=8)이라, 이 스크립트의 "결정" 하나는 물리
스텝 하나가 아니라 "청크 예측 1회"(=역확산 100단계 체인 하나)다. 청크가 실행하는
최대 action_horizon 스텝 동안 벌어진 일의 보상을 그 결정이 통째로 받는다
(`phases/5-mani-sim-ddpo/step3.md` 참고).

`log p(a|o,g)`는 연속 디퓨전이라 닫힌 형태가 없다 — `policies/diffusion/ddpo.py`(step 0)가
역확산 100단계 각각을 가우시안 전이로 보고 단계별 로그확률의 합으로 대체한다(DDPO,
Black et al. 2023). 데이터 재사용이 없는 DDPO-SF(score function, 바닐라 REINFORCE)를
쓰는 이유도 원본과 동일 — 논문의 on-policy 설계와 정확히 맞기 때문이다.

보상/성공 판정은 `policies/diffusion/dstg_reward.py`(step 2)가 감싼 얼려진
관측-only STG 예측기(step 1)에서 나온다. 환경의 `is_success()`/`done`은 진단
로그(env_succ_rate)와 `--termination env` 모드의 종료 판정에만 쓰이고, 보상
계산에는 절대 들어가지 않는다(`compute_decision_reward`의 시그니처 자체가 그 경계다).

실행:
    python -m mani_sim.scripts.train_si \
        policy_ckpt=/home/moai/hymm_ws/square_ckpt/policy_epoch1060.pt \
        dstg_ckpt=outputs/dstg/square_demo50_succ/predictor.pt \
        iterations=1 episodes_per_iter=2 \
        out=outputs/si_smoke/predictor.pt
"""

import copy
import logging
import os
import shutil
import time
from collections import deque

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from mani_sim.datasets.normalization import MinMaxNormalizer, load_stats
from mani_sim.datasets.robomimic_dataset import RobomimicSequenceDataset
from mani_sim.factory import registry
from mani_sim.policies.diffusion import ddpo as ddpo_module
from mani_sim.policies.diffusion.dstg_reward import DstgReward, calibrate_threshold
from mani_sim.runners.rollout import _build_obs_batch
from mani_sim.utils.checkpoints import load_epoch_checkpoint, load_run_config, save_run_config
from mani_sim.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

logger = logging.getLogger(__name__)

DRIFT_SEED = 12345          # 드리프트 진단용 고정 노이즈 x 시드
DRIFT_N_OBS = 16            # 드리프트 진단용 고정 관측 배치 크기(held-out demo 프레임)


# ---------------------------------------------------------------- 순수 함수부
# (테스트가 env/정책 없이 직접 부르는 부분)

def compute_decision_reward(d_before: float, d_after: float) -> float:
    """식(2)의 청크 버전: r = d(o_before,g) - d(o_after,g).

    인자로 받는 것은 d 값 둘뿐이다 — 환경의 그 어떤 ground-truth 신호도 여기 들어오지
    않는다. 논문의 핵심 주장("외부 감독 없는 자기개선")이 이 경계에 걸려 있으므로,
    이 함수의 시그니처 자체가 그 경계다.
    """
    return float(d_before) - float(d_after)


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """R_i = sum_{j>=i} gamma^(j-i) * r_j, 역순 누적으로 계산.

    From: phases/4-diffusion-si/step3.md의 train_carry_si.py::compute_returns —
    같은 공식을 그대로 재사용(새로 유도하지 않음). rewards: (T,) -> (T,).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    t_len = rewards.shape[0]
    returns = np.zeros(t_len, dtype=np.float64)
    acc = 0.0
    for t in range(t_len - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        returns[t] = acc
    return returns.astype(np.float32)


def _tile_pair_indices(n_decisions: int, n_pairs_per_decision: int):
    """decision-major(j) x pair-index(p) 전체 나열: xs[j, p] --(timesteps_all[p])--> xs[j, p+1].

    `ddpo.chain_logp`의 규칙과 정확히 같다(step 0): xs[i]에서 timesteps_all[i]를 적용하면
    xs[i+1]이 나온다. timesteps_all은 `scheduler.timesteps[:-1]`(t=0 제외, 결정론적이라
    로그확률이 없다) — 길이가 n_steps-1이라 한 결정당 n_steps-1개의 쌍이 나온다.
    """
    j_idx = np.repeat(np.arange(n_decisions), n_pairs_per_decision)
    p_idx = np.tile(np.arange(n_pairs_per_decision), n_decisions)
    return j_idx, p_idx


def _episode_split(dataset, val_fraction, seed):
    """무작위 에피소드(데모) 단위 분할.

    From: mani_sim/src/mani_sim/policies/diffusion/dstg_reward.py의 _episode_split —
    같은 알고리즘을 그대로 다시 쓴다(그 파일 자체가 scripts/train_dstg.py에서 같은
    이유로 복제한 것 — hydra 진입점을 끌고 오지 않기 위해, ADR-005). dstg_reward.py는
    수정하지 않는다.
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


# --------------------------------------------------------------------- 롤아웃

def collect_episode_si(env, policy, ddpo_fns, dstg_reward, normalizer, obs_keys, rgb_keys,
                        obs_horizon, action_horizon, pred_horizon, action_dim, max_steps,
                        device, gamma, generator, termination):
    """한 에피소드를 실제로 굴리고, 결정(청크)별 (global_cond, xs, R) 리스트를 반환한다.

    "결정" = predict_action_chunk 1회에 대응하는 `ddpo_fns.sample_with_trace` 1회 호출
    (체인 하나). 매 결정 직전 obs로 d_before를 재고, 그 청크가 실행한 만큼(최대
    action_horizon 스텝, ground-truth 성공/max_steps로 조기 종료 가능 — rollout.py의
    청크 실행 참조 구현과 동일 조건)의 마지막 obs로 d_after를 재서
    `compute_decision_reward`로 보상을 만든다.

    env.is_success()["task"](ground truth)는 두 곳에서만 쓰인다: (a) 청크 실행 도중의
    조기 종료(물리 스텝을 아낄 뿐 보상 계산에는 안 들어간다 — reward는 항상 실제
    최종 obs의 d_after로 계산됨), (b) termination=="env"일 때의 에피소드 종료 판정.
    termination=="learned"일 때는 `dstg_reward.success(obs_after)`가 종료를 판정하고,
    ground truth는 진단용 `env_success` 반환값으로만 쓰인다(학습 신호 아님).
    """
    obs_raw = env.reset()
    obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)

    global_conds, xs_list, rewards = [], [], []
    n_env_steps = 0
    env_success = False  # 진단 전용(ground truth) — 학습 신호로 쓰지 않는다.

    while n_env_steps < max_steps:
        obs_batch = _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn=None)
        with torch.no_grad():
            global_cond = policy.get_global_cond(obs_batch)  # (1, cond_dim)
            d_before = dstg_reward.d(obs_batch).item()

        policy.inference_scheduler.set_timesteps(policy.num_inference_steps, device=device)
        shape = (1, pred_horizon, action_dim)
        x_final, xs = ddpo_fns.sample_with_trace(
            policy.unet, policy.inference_scheduler, global_cond, shape, device, generator=generator
        )
        action_chunk = normalizer.unnormalize_action(x_final[0].detach().cpu())  # (Tp, Da)

        for t in range(action_horizon):
            action = action_chunk[t].numpy()
            obs_raw, _reward, done, _info = env.step(action)
            obs_history.append(obs_raw)
            n_env_steps += 1
            if env.is_success()["task"]:
                env_success = True
            if env_success or done or n_env_steps >= max_steps:
                break

        obs_batch_after = _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn=None)
        with torch.no_grad():
            d_after = dstg_reward.d(obs_batch_after).item()
        r = compute_decision_reward(d_before, d_after)

        global_conds.append(global_cond[0].detach().cpu())
        xs_list.append(xs[:, 0].detach().cpu())  # (n_steps+1, Tp, Da), 배치축(=1) 제거
        rewards.append(r)

        if termination == "learned":
            with torch.no_grad():
                stop = bool(dstg_reward.success(obs_batch_after)[0]) or n_env_steps >= max_steps
        else:  # "env"
            stop = env_success or n_env_steps >= max_steps
        if stop:
            break

    returns = compute_returns(np.asarray(rewards, dtype=np.float32), gamma)
    decisions = list(zip(global_conds, xs_list, returns))
    return decisions, env_success, n_env_steps


# --------------------------------------------------------------------- 진단

def _load_drift_batch(calib_dataset, val_idx, device):
    idx = list(val_idx[: min(DRIFT_N_OBS, len(val_idx))])
    loader = DataLoader(Subset(calib_dataset, idx), batch_size=len(idx), shuffle=False)
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch["obs"].items()}


def _drift_metric(base_unet, cur_unet, global_cond, x_fixed, t_fixed):
    """원본(고정) unet과 현재 unet이 같은 (x, t, global_cond)에서 내는 노이즈 예측의 L2 거리
    평균 — 진단용일 뿐 손실에 넣지 않는다(reward hacking으로 정책이 무너지는 걸 조기에 보기
    위함, phase 4 step 3과 동일 목적). 전체 역확산 체인을 다시 샘플하지 않고 unet 출력만
    비교해 매 iteration 비용을 낮춘다."""
    with torch.no_grad():
        eps_base = base_unet(x_fixed, t_fixed, global_cond)
        eps_cur = cur_unet(x_fixed, t_fixed, global_cond)
    return (eps_cur - eps_base).flatten(1).norm(dim=-1).mean().item()


def _assert_encoders_frozen(before, after):
    for k, v in before.items():
        if not torch.equal(v, after[k]):
            raise RuntimeError(
                f"비전 인코더 파라미터 {k}가 DDPO 업데이트로 바뀌었다 — step_logp의 그래디언트가 "
                "policy.encoders로 새고 있다는 뜻이다. optimizer가 policy.unet.parameters()만 "
                "받는지, global_cond가 no_grad로 계산·detach됐는지 확인할 것. 즉시 중단."
            )


# ------------------------------------------------------------------------ main

@hydra.main(config_path="../configs", config_name="train_si", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed0)

    # task/policy는 policy_ckpt 옆 run_config.yaml에서 복원한다 — eval.py의 _apply_run_config와
    # 동일 이유(학습·추론 설정이 사람 실수로 어긋나는 걸 구조적으로 막음). 직접 대입으로
    # 교체한다(OmegaConf.merge는 깊은 병합이라 defaults의 square.yaml에만 있고 실제 학습
    # task엔 없는 필드가 살아남는 문제가 있음 — eval.py의 동일 주석 참고).
    saved = load_run_config(cfg.policy_ckpt)
    if saved is not None:
        cfg.task = saved.task
        cfg.policy = saved.policy
        policy_name = saved.policy_name
        logger.info(f"run_config.yaml에서 자동 적용: task={saved.task.name} policy_name={policy_name}")
    else:
        policy_name = "diffusion"
    task_cfg, policy_cfg = cfg.task, cfg.policy

    policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(cfg.policy_ckpt, policy, device)
    policy.eval()  # VisionEncoder crop을 CenterCrop으로 고정(rollout.py/eval.py와 동일 관례) —
    # 학습 내내 이 모드를 유지한다(BatchNorm은 이미 GroupNorm으로 교체돼 있어 train/eval
    # 차이가 없고, RandomCrop 같은 확률적 augmentation만 이 플래그로 갈린다).
    policy.encoders.requires_grad_(False)  # 이중 안전판 — collect 단계는 이미 no_grad로 감싸고
    # global_cond를 detach해서 저장하므로 인코더로 그래디언트가 흐를 경로 자체가 없지만,
    # DstgPredictor(step 1)와 같은 원칙으로 명시적으로도 얼려둔다.

    base_unet = copy.deepcopy(policy.unet).to(device)
    base_unet.requires_grad_(False)
    base_unet.eval()

    stats_path = os.path.join(os.path.dirname(cfg.policy_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []

    dstg_reward = DstgReward(cfg.dstg_ckpt, statistic=cfg.statistic, cvar_alpha=cfg.cvar_alpha, device=device)

    calib_dataset = RobomimicSequenceDataset(
        hdf5_path=cfg.calib_hdf5_path, obs_keys=dstg_reward.obs_keys, obs_horizon=dstg_reward.obs_horizon,
        pred_horizon=1, normalizer=normalizer, rgb_keys=dstg_reward.rgb_keys,
    )
    _, val_idx, n_train_demos, n_val_demos = _episode_split(calib_dataset, cfg.calib_val_fraction, cfg.calib_split_seed)

    if cfg.termination == "learned":
        best_s, calib_metrics = calibrate_threshold(dstg_reward, calib_dataset, val_idx)
        dstg_reward.threshold = best_s
        logger.info(
            f"[calib] threshold s={best_s:.3f} f1={calib_metrics['f1']:.3f} precision={calib_metrics['precision']:.3f} "
            f"recall={calib_metrics['recall']:.3f} (held-out {n_val_demos} demos/{len(val_idx)} samples, "
            f"train={n_train_demos} demos)"
        )
        print(f"[calib] threshold s={best_s:.3f} f1={calib_metrics['f1']:.3f}", flush=True)

    # ---- 드리프트 진단용 고정 배치(held-out demo 프레임 + 고정 노이즈 x + 고정 t) ----
    drift_obs = _load_drift_batch(calib_dataset, val_idx, device)
    with torch.no_grad():
        drift_global_cond = policy.get_global_cond(drift_obs)
    drift_gen = torch.Generator(device=device).manual_seed(DRIFT_SEED)
    drift_x = torch.randn(
        (drift_global_cond.shape[0], policy_cfg.pred_horizon, task_cfg.action_dim),
        generator=drift_gen, device=device,
    )
    policy.inference_scheduler.set_timesteps(policy.num_inference_steps, device=device)
    n_steps = len(policy.inference_scheduler.timesteps)
    drift_t = policy.inference_scheduler.timesteps[n_steps // 2].expand(drift_global_cond.shape[0])

    env = make_eval_env(task_cfg)

    optimizer = torch.optim.Adam(policy.unet.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(cfg.seed0 + 1_000_000)
    env_seed_counter = cfg.seed0

    first_update_checked = False
    n_pairs_per_decision = n_steps - 1
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)

    for it in range(1, cfg.iterations + 1):
        t_iter_start = time.time()

        all_global_cond, all_xs, all_R = [], [], []
        ep_lens, ep_decision_counts, ep_env_succ = [], [], []

        with torch.no_grad():
            for _ in range(cfg.episodes_per_iter):
                np.random.seed(env_seed_counter)  # robosuite 초기 배치 시퀀스 고정(eval.py와 동일 방식)
                env_seed_counter += 1
                decisions, env_success, n_env_steps = collect_episode_si(
                    env, policy, ddpo_module, dstg_reward, normalizer, obs_keys, rgb_keys,
                    policy_cfg.obs_horizon, policy_cfg.action_horizon, policy_cfg.pred_horizon,
                    task_cfg.action_dim, cfg.max_steps, device, cfg.gamma, generator, cfg.termination,
                )
                for gc, xs, r in decisions:
                    all_global_cond.append(gc)
                    all_xs.append(xs)
                    all_R.append(r)
                ep_lens.append(n_env_steps)
                ep_decision_counts.append(len(decisions))
                ep_env_succ.append(env_success)

        global_cond_all = torch.stack(all_global_cond)  # (N, cond_dim), CPU
        xs_all = torch.stack(all_xs)                      # (N, n_steps+1, Tp, Da), CPU
        R_all = np.asarray(all_R, dtype=np.float32)
        n_decisions = global_cond_all.shape[0]

        if cfg.advantage_norm:
            R_train = (R_all - R_all.mean()) / (R_all.std() + 1e-8)
        else:
            R_train = R_all

        # ---- (결정, 역확산-단계) 쌍 나열 → 셔플 → 미니배치 순회 (이 배치로 딱 한 번) ----
        j_idx, p_idx = _tile_pair_indices(n_decisions, n_pairs_per_decision)
        perm = np.random.default_rng(cfg.seed0 + it).permutation(len(j_idx))
        j_idx, p_idx = j_idx[perm], p_idx[perm]

        timesteps_all = policy.inference_scheduler.timesteps[:-1]  # (n_steps-1,), t=0 제외

        losses = []
        n_total_pairs = len(j_idx)
        for start in range(0, n_total_pairs, cfg.logp_batch):
            sl = slice(start, start + cfg.logp_batch)
            bj, bp = j_idx[sl], p_idx[sl]

            global_cond_b = global_cond_all[bj].to(device)
            x_in_b = xs_all[bj, bp].to(device)
            x_out_b = xs_all[bj, bp + 1].to(device)
            t_b = timesteps_all[bp].to(device)
            R_b = torch.as_tensor(R_train[bj], dtype=torch.float32, device=device)

            if not first_update_checked:
                encoders_before = {k: v.clone() for k, v in policy.encoders.state_dict().items()}

            lp = ddpo_module.step_logp(policy.unet, policy.inference_scheduler, global_cond_b, x_in_b, x_out_b, t_b)
            loss = -cfg.reinforce_scale * (R_b * lp).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if not first_update_checked:
                _assert_encoders_frozen(encoders_before, policy.encoders.state_dict())
                first_update_checked = True
                logger.info("[check] 첫 업데이트 후 비전 인코더 파라미터 불변 확인됨.")

        # ---- 로그 -----------------------------------------------------------
        drift = _drift_metric(base_unet, policy.unet, drift_global_cond, drift_x, drift_t)
        n_env_succ = sum(ep_env_succ)
        iter_time = time.time() - t_iter_start

        msg = (
            f"it={it:4d}  R_mean={R_all.mean():+8.4f}  |R|_mean={np.abs(R_all).mean():7.4f}  "
            f"env_succ_rate={n_env_succ / cfg.episodes_per_iter:5.1%}  "
            f"decisions_per_ep={np.mean(ep_decision_counts):5.1f}  ep_len_mean={np.mean(ep_lens):6.1f}  "
            f"drift_L2={drift:7.4f}  loss={np.mean(losses):+9.5f}  time={iter_time:5.1f}s"
        )
        print(msg, flush=True)
        logger.info(msg)

        if not np.isfinite(R_all).all() or not np.isfinite(drift):
            raise RuntimeError(
                f"it={it}: 리턴 또는 드리프트가 비유한(NaN/inf)이 됐다 — 중단. "
                f"R_all finite={np.isfinite(R_all).all()}  drift={drift}"
            )

    env.env.close()

    out_dir = os.path.dirname(cfg.out) or "."
    payload = {
        "model": policy.state_dict(),
        "epoch": cfg.iterations,
        "si_dstg_ckpt": cfg.dstg_ckpt,
        "si_statistic": cfg.statistic,
        "si_gamma": cfg.gamma,
        "si_reinforce_scale": cfg.reinforce_scale,
        "si_iterations": cfg.iterations,
        "si_episodes_per_iter": cfg.episodes_per_iter,
        "si_base_ckpt": cfg.policy_ckpt,
    }
    torch.save(payload, cfg.out)
    save_run_config(out_dir, task_cfg, policy_cfg, policy_name)
    shutil.copy(stats_path, os.path.join(out_dir, "normalization_stats.json"))
    logger.info(f"저장: {cfg.out} (+ run_config.yaml, normalization_stats.json)")
    print(f"저장: {cfg.out}", flush=True)


if __name__ == "__main__":
    main()
