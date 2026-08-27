# Repo Reorg (grasp_carry / pointmass / square_assembly) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repo's source code into two self-contained, `mani_sim`-style subprojects (`grasp_carry/` for the JAX/Haiku GraspCarry2D stack, and a renamed `square_assembly/` for the PyTorch/robomimic stack), plus a minimal `pointmass/` skeleton for the archived-but-not-dead pointmass work — without touching the existing Docker image/service architecture.

**Architecture:** Each subproject gets its own `src/<name>/<name>/` src-layout package (mirroring the existing `mani_sim/` pattern), its own `tests/`, and (for `grasp_carry/`) its own `data/`, `checkpoints/`, `results/`, `outputs/`. Imports move from ad-hoc root-relative (`from src.grasp_carry.x import y`, bare `from train_carry_actor import y`) to fully-qualified package imports (`from grasp_carry.x import y`, `from grasp_carry.scripts.train.train_carry_actor import y`). No new runtime dependency is introduced — `PYTHONPATH` (not editable pip install) is what makes each package importable, exactly as `mani_sim/`'s Dockerfile already does today.

**Tech Stack:** Python 3.11 (JAX/Haiku stack, root `requirements.txt`), Python 3.10 (PyTorch/robomimic stack, `square_assembly/requirements.txt`). `git mv` for all tracked-file moves, `sed`/`Edit` for import rewrites, `pytest` for regression verification.

**Spec:** `docs/superpowers/specs/2026-08-27-repo-reorg-design.md`

## Global Constraints

- Root `requirements.txt` stays the single dependency source for the JAX stack, shared by `grasp_carry/` and the future `pointmass/` revival — do **not** create a `grasp_carry/requirements.txt`.
- No editable pip install for either subproject. `PYTHONPATH` env vars make the packages importable (`grasp_carry/src`, `square_assembly/src`), matching the existing `mani_sim/torch.Dockerfile` convention.
- `scripts/` subfolders (`train/`, `collect/`, `record/`, `analyze/`) are implicit namespace packages — no `__init__.py`, matching `mani_sim/src/mani_sim/scripts/`'s existing convention.
- Do not rewrite the narrative content of `docs/ARCHITECTURE.md` or `docs/ADR.md` — only literal path/name strings that the rename breaks (e.g. `mani_sim/` → `square_assembly/`). The 1차-목표/pointmass-narrative rewrite is explicitly out of scope for this plan.
- `COMMANDS.md` full rewrite, `archive/` triage (pointmass-only vs. grasp_carry-shared), and `requirements.txt` dependency trimming are out of scope — do not touch beyond the specific line edits called out below.
- Docker **service names** (`jax`, `torch` in `docker-compose.yml`) do not change — only the paths they point at.
- The local dev environment was renamed mid-project: use conda env `self-improving-gym` (not `efms-gym`) for all JAX-stack verification commands in this plan. There is currently no local conda env for the PyTorch/robomimic stack — Group 3 verification is import/syntax-level only where noted.
- Preserve `src/grasp_carry/__init__.py`'s existing docstring (`"""GraspCarry2D — 은닉 물성 기반 파지·운반 환경 패키지 (phase 3)."""`) as `grasp_carry/src/grasp_carry/__init__.py`'s content — don't blank it out.

---

## Task 1: Scaffold `grasp_carry/` and move the library code

**Files:**
- Create: `grasp_carry/src/grasp_carry/` (via `git mv` of `src/grasp_carry/*`)
- Modify (moved + import-fixed): `grasp_carry/src/grasp_carry/{config,env,gripper,policy,ddpo,diffusion_act,conditional_unet1d,eval_carry,reward,carry_stg_reward,train_carry_dstg,train_carry_predictor,train_carry_qstg}.py`
- Create: `grasp_carry/src/grasp_carry/networks.py`
- Test: run existing `tests/` in place (not yet moved) against the new package to catch import breakage early

**Interfaces:**
- Produces: importable package `grasp_carry` (once `PYTHONPATH` includes `grasp_carry/src`), exposing `grasp_carry.config.CarryConfig`, `grasp_carry.env.GraspCarry2D`, `grasp_carry.env.FRAME_FIELDS`, `grasp_carry.gripper.Gripper`, `grasp_carry.policy.ScriptedCarryPolicy`, `grasp_carry.policy._CEILING_Y`, `grasp_carry.ddpo.build_ddpo`, `grasp_carry.ddpo._posterior_mean`, `grasp_carry.diffusion_act.{build_diffusion_act_chunk,diffusion_loss,DiffusionActNets}`, `grasp_carry.conditional_unet1d.conditional_unet1d`, `grasp_carry.eval_carry.rollout`, `grasp_carry.reward.{stepwise_reward,discounted_returns,reward_from_config}`, `grasp_carry.carry_stg_reward.{StgReward,_d_from_probs,calibrate_threshold,_val_episode_ids}`, `grasp_carry.train_carry_dstg.build_dstg_net`, `grasp_carry.train_carry_predictor.concat_obs`, `grasp_carry.train_carry_qstg.{split_success_fail,succ_mean_quantile,succ_cvar}`, `grasp_carry.networks.build_continuous_act_discrete_dist_v0`.

- [ ] **Step 1: Move the `grasp_carry` subpackage as-is**

```bash
mkdir -p grasp_carry/src
git mv src/grasp_carry grasp_carry/src/grasp_carry
```

- [ ] **Step 2: Move the flat library modules into the same package directory**

```bash
git mv src/carry_stg_reward.py grasp_carry/src/grasp_carry/carry_stg_reward.py
git mv src/conditional_unet1d.py grasp_carry/src/grasp_carry/conditional_unet1d.py
git mv src/ddpo.py grasp_carry/src/grasp_carry/ddpo.py
git mv src/diffusion_act.py grasp_carry/src/grasp_carry/diffusion_act.py
git mv src/eval_carry.py grasp_carry/src/grasp_carry/eval_carry.py
git mv src/reward.py grasp_carry/src/grasp_carry/reward.py
git mv src/train_carry_dstg.py grasp_carry/src/grasp_carry/train_carry_dstg.py
git mv src/train_carry_predictor.py grasp_carry/src/grasp_carry/train_carry_predictor.py
git mv src/train_carry_qstg.py grasp_carry/src/grasp_carry/train_carry_qstg.py
```

- [ ] **Step 3: Port the TIMER network builder from `pointmass_core.py` into `networks.py`**

Create `grasp_carry/src/grasp_carry/networks.py` with exactly this content (copied verbatim from
`pointmass_core.py` lines 322-510 — the `build_continuous_act_discrete_dist_v0` function plus the
handful of supporting types/aliases it needs, trimmed of everything else in that file, e.g.
`make_policy_fn`/`Action`/`FeedForwardPolicyWithExtra` are not included because
`build_continuous_act_discrete_dist_v0` doesn't use them):

```python
"""TIMER network builder, ported from pointmass_core.py (SI-EFM paper reproduction).

From: pointmass/pointmass_core.py (build_continuous_act_discrete_dist_v0 and its
supporting types) -- copied rather than imported so grasp_carry has no runtime
dependency on the pointmass/ subproject.
"""

import dataclasses
from typing import Any, Callable, NamedTuple, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

Params = Any
PRNGKey = Any
NetworkOutput = Any
Entropy = Any
ActDistParams = Params
LogProbFn = Any
SampleFn = Any
Observation = Any
DistanceToSuccessDistParams = Params
EntropyFn = Callable[[Params, PRNGKey], Entropy]

MIN_ACT_SCALE = 1e-2


@dataclasses.dataclass
class FeedForwardNetwork:
  """Holds a pair of pure functions defining a feed-forward network."""
  init: Callable[..., Params]
  apply: Callable[..., NetworkOutput]


class MVNDiagParams(NamedTuple):
  """Parameters for a diagonal multi-variate normal distribution."""
  loc: jnp.ndarray
  scale_diag: jnp.ndarray


class CategoricalParams(NamedTuple):
  """Parameters for a categorical distribution."""
  logits: jnp.ndarray


class TIMERNetworkOutput(NamedTuple):
  act_dist_params: ActDistParams
  dist_to_succ_dist_params: DistanceToSuccessDistParams


@dataclasses.dataclass
class TIMERNetworks:
  """Network and pure functions for the TIMER agent."""
  network: FeedForwardNetwork
  act_log_prob: LogProbFn
  sample_act: SampleFn
  dist_log_prob: LogProbFn
  sample_dist: SampleFn
  act_entropy: Optional[EntropyFn] = None
  sample_act_mode: Optional[SampleFn] = None
  dist_entropy: Optional[EntropyFn] = None
  sample_dist_mode: Optional[SampleFn] = None


def build_continuous_act_discrete_dist_v0(
    layer_sizes: Sequence[int],
    act_dim: int,
    num_dist_bins: int,
    dummy_input,
) -> TIMERNetworks:
  """Builds TIMERNetworks for continuous action and discrete distance."""

  def _network(x: Observation) -> TIMERNetworkOutput:
    h_act = hk.nets.MLP(
        output_sizes=layer_sizes,
        activation=jax.nn.relu,
        activate_final=True,)(x)
    act_loc = hk.Linear(
        act_dim,
        w_init=hk.initializers.VarianceScaling(1e-4),
        b_init=hk.initializers.Constant(0.))(h_act)
    act_scale = hk.Linear(
        act_dim,
        w_init=hk.initializers.VarianceScaling(1e-4),
        b_init=hk.initializers.Constant(0.))(h_act)
    act_scale = jax.nn.softplus(act_scale) + MIN_ACT_SCALE
    act_dist = MVNDiagParams(loc=act_loc, scale_diag=act_scale)

    h_dist = hk.nets.MLP(
        output_sizes=layer_sizes,
        activation=jax.nn.relu,
        activate_final=True,)(x)
    dist_logits = hk.Linear(num_dist_bins, with_bias=False)(h_dist)
    distance_dist = CategoricalParams(logits=dist_logits)

    return TIMERNetworkOutput(
        act_dist_params=act_dist,
        dist_to_succ_dist_params=distance_dist,)

  transformed_network = hk.without_apply_rng(hk.transform(_network))
  def init_closure(rng: PRNGKey):
    return transformed_network.init(rng, dummy_input)
  network = FeedForwardNetwork(
      init=init_closure,
      apply=transformed_network.apply,)

  def act_log_prob(params: MVNDiagParams, action):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).log_prob(action)

  def act_entropy(params: MVNDiagParams, key: PRNGKey) -> Entropy:
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).entropy()

  def sample_act(params: MVNDiagParams, key: PRNGKey):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).sample(seed=key)

  def sample_act_mode(params: MVNDiagParams, key: PRNGKey):
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).mode()

  def dist_log_prob(params: CategoricalParams, action):
    return tfd.Categorical(logits=params.logits).log_prob(action)

  def dist_entropy(params: CategoricalParams, key: PRNGKey) -> Entropy:
    del key
    return tfd.Categorical(logits=params.logits).entropy()

  def sample_dist(params: CategoricalParams, key: PRNGKey):
    return tfd.Categorical(logits=params.logits).sample(seed=key)

  def sample_dist_mode(params: CategoricalParams, key: PRNGKey):
    del key
    return tfd.Categorical(logits=params.logits).mode()

  return TIMERNetworks(
      network=network,
      act_log_prob=act_log_prob,
      sample_act=sample_act,
      dist_log_prob=dist_log_prob,
      sample_dist=sample_dist,
      act_entropy=act_entropy,
      sample_act_mode=sample_act_mode,
      dist_entropy=dist_entropy,
      sample_dist_mode=sample_dist_mode,
  )
```

Do not modify `pointmass_core.py` itself — it is being copied from, not moved yet (that happens in
Task 6), and stays byte-identical until then.

- [ ] **Step 4: Rewrite subpackage-qualified imports (`from src.grasp_carry.X` → `from grasp_carry.X`)**

```bash
grep -rl 'from src\.grasp_carry\.' grasp_carry/src | xargs -r sed -i -E 's/from src\.grasp_carry\./from grasp_carry./g'
```

- [ ] **Step 5: Rewrite flat-module imports (`from src.X` → `from grasp_carry.X`)**

Run this *after* Step 4 so it doesn't double-touch the already-fixed `src.grasp_carry.` lines:

```bash
grep -rl 'from src\.' grasp_carry/src | xargs -r sed -i -E 's/from src\./from grasp_carry./g'
```

- [ ] **Step 6: Point `train_carry_predictor.py` at the ported network function**

Edit `grasp_carry/src/grasp_carry/train_carry_predictor.py` line with
`from pointmass_core import build_continuous_act_discrete_dist_v0` →
`from grasp_carry.networks import build_continuous_act_discrete_dist_v0`.

- [ ] **Step 7: Verify the package imports cleanly**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
PYTHONPATH=grasp_carry/src python -c "
from grasp_carry.config import CarryConfig
from grasp_carry.env import GraspCarry2D, FRAME_FIELDS
from grasp_carry.gripper import Gripper
from grasp_carry.policy import ScriptedCarryPolicy
from grasp_carry.ddpo import build_ddpo
from grasp_carry.diffusion_act import build_diffusion_act_chunk
from grasp_carry.eval_carry import rollout
from grasp_carry.reward import stepwise_reward
from grasp_carry.carry_stg_reward import StgReward
from grasp_carry.train_carry_dstg import build_dstg_net
from grasp_carry.train_carry_predictor import concat_obs
from grasp_carry.train_carry_qstg import split_success_fail
from grasp_carry.networks import build_continuous_act_discrete_dist_v0
print('grasp_carry package imports OK')
"
```

Expected: `grasp_carry package imports OK`, no `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 8: Commit**

```bash
git add grasp_carry/ src/
git status  # confirm src/ shows the moved-out files, grasp_carry/ shows the new tree
git commit -m "$(cat <<'EOF'
refactor(grasp_carry): consolidate GraspCarry2D library code into grasp_carry/src/grasp_carry/

- move src/grasp_carry/ subpackage and flat src/*.py modules into grasp_carry/src/grasp_carry/
- port pointmass_core.py's TIMER network builder into grasp_carry/src/grasp_carry/networks.py
  so train_carry_predictor.py no longer depends on the root pointmass_core.py
- rewrite all src.grasp_carry.X / src.X imports to grasp_carry.X
EOF
)"
```

---

## Task 2: Move and categorize the 25 entry-point scripts

**Files:**
- Create: `grasp_carry/src/grasp_carry/scripts/{train,collect,record,analyze}/` (via `git mv` of the 25 root scripts)
- Test: `python -m py_compile` + `--help` spot checks (below)

**Interfaces:**
- Consumes: `grasp_carry.*` library modules from Task 1 (must already import cleanly)
- Produces: `grasp_carry.scripts.train.{train_carry_actor,train_carry_actor_reinforce,train_carry_si,finetune_carry_diffusion}`, `grasp_carry.scripts.collect.{collect_carry_demos,collect_carry_bc_rollouts,collect_carry_teleop_detour}`, `grasp_carry.scripts.record.{record_carry,record_carry_actor,record_carry_bc_stg_dist,record_carry_si,record_carry_si_video,record_carry_stg_dist}`, `grasp_carry.scripts.analyze.{calibrate_carry,compare_carry_selectors,eval_carry_actor,eval_carry_si,probe_carry_qstg,rollout_carry_diff_stats,run_bc_stg_guided,verify_carry_qstg_condb,analyze_mu_jump_bimodal,analyze_mu_sigma_highrisk,evaluate_stg_deadline,evaluate_stg_deadline_cdf}` — this is the exact module map Task 3 (tests) and later scripts rely on.

- [ ] **Step 1: Move each script into its category subfolder**

```bash
mkdir -p grasp_carry/src/grasp_carry/scripts/{train,collect,record,analyze}

git mv train_carry_actor.py grasp_carry/src/grasp_carry/scripts/train/train_carry_actor.py
git mv train_carry_actor_reinforce.py grasp_carry/src/grasp_carry/scripts/train/train_carry_actor_reinforce.py
git mv train_carry_si.py grasp_carry/src/grasp_carry/scripts/train/train_carry_si.py
git mv finetune_carry_diffusion.py grasp_carry/src/grasp_carry/scripts/train/finetune_carry_diffusion.py

git mv collect_carry_demos.py grasp_carry/src/grasp_carry/scripts/collect/collect_carry_demos.py
git mv collect_carry_bc_rollouts.py grasp_carry/src/grasp_carry/scripts/collect/collect_carry_bc_rollouts.py
git mv collect_carry_teleop_detour.py grasp_carry/src/grasp_carry/scripts/collect/collect_carry_teleop_detour.py

git mv record_carry.py grasp_carry/src/grasp_carry/scripts/record/record_carry.py
git mv record_carry_actor.py grasp_carry/src/grasp_carry/scripts/record/record_carry_actor.py
git mv record_carry_bc_stg_dist.py grasp_carry/src/grasp_carry/scripts/record/record_carry_bc_stg_dist.py
git mv record_carry_si.py grasp_carry/src/grasp_carry/scripts/record/record_carry_si.py
git mv record_carry_si_video.py grasp_carry/src/grasp_carry/scripts/record/record_carry_si_video.py
git mv record_carry_stg_dist.py grasp_carry/src/grasp_carry/scripts/record/record_carry_stg_dist.py

git mv calibrate_carry.py grasp_carry/src/grasp_carry/scripts/analyze/calibrate_carry.py
git mv compare_carry_selectors.py grasp_carry/src/grasp_carry/scripts/analyze/compare_carry_selectors.py
git mv eval_carry_actor.py grasp_carry/src/grasp_carry/scripts/analyze/eval_carry_actor.py
git mv eval_carry_si.py grasp_carry/src/grasp_carry/scripts/analyze/eval_carry_si.py
git mv probe_carry_qstg.py grasp_carry/src/grasp_carry/scripts/analyze/probe_carry_qstg.py
git mv rollout_carry_diff_stats.py grasp_carry/src/grasp_carry/scripts/analyze/rollout_carry_diff_stats.py
git mv run_bc_stg_guided.py grasp_carry/src/grasp_carry/scripts/analyze/run_bc_stg_guided.py
git mv verify_carry_qstg_condb.py grasp_carry/src/grasp_carry/scripts/analyze/verify_carry_qstg_condb.py
git mv analyze_mu_jump_bimodal.py grasp_carry/src/grasp_carry/scripts/analyze/analyze_mu_jump_bimodal.py
git mv analyze_mu_sigma_highrisk.py grasp_carry/src/grasp_carry/scripts/analyze/analyze_mu_sigma_highrisk.py
git mv evaluate_stg_deadline.py grasp_carry/src/grasp_carry/scripts/analyze/evaluate_stg_deadline.py
git mv evaluate_stg_deadline_cdf.py grasp_carry/src/grasp_carry/scripts/analyze/evaluate_stg_deadline_cdf.py
```

- [ ] **Step 2: Rewrite library imports inside the moved scripts (same two-pass rule as Task 1)**

```bash
grep -rl 'from src\.grasp_carry\.' grasp_carry/src/grasp_carry/scripts | xargs -r sed -i -E 's/from src\.grasp_carry\./from grasp_carry./g'
grep -rl 'from src\.' grasp_carry/src/grasp_carry/scripts | xargs -r sed -i -E 's/from src\./from grasp_carry./g'
```

- [ ] **Step 3: Rewrite bare cross-script imports to fully-qualified module paths**

These 25 scripts import each other by bare filename today (e.g. `record_carry_actor.py` does
`from record_carry import draw_env, _ee_block_dist`). The category map below is exactly the one
used in Step 1 — apply it generically so every cross-reference (verified exhaustively against the
current repo — 7 target scripts, 21 importer occurrences across scripts/ and tests/) gets fixed in
one pass, including any this plan's author might have missed:

```bash
declare -A CARRY_SCRIPT_CATEGORY=(
  [train_carry_actor]=train [train_carry_actor_reinforce]=train
  [train_carry_si]=train [finetune_carry_diffusion]=train
  [collect_carry_demos]=collect [collect_carry_bc_rollouts]=collect
  [collect_carry_teleop_detour]=collect
  [record_carry]=record [record_carry_actor]=record [record_carry_bc_stg_dist]=record
  [record_carry_si]=record [record_carry_si_video]=record [record_carry_stg_dist]=record
  [calibrate_carry]=analyze [compare_carry_selectors]=analyze [eval_carry_actor]=analyze
  [eval_carry_si]=analyze [probe_carry_qstg]=analyze [rollout_carry_diff_stats]=analyze
  [run_bc_stg_guided]=analyze [verify_carry_qstg_condb]=analyze
  [analyze_mu_jump_bimodal]=analyze [analyze_mu_sigma_highrisk]=analyze
  [evaluate_stg_deadline]=analyze [evaluate_stg_deadline_cdf]=analyze
)
for name in "${!CARRY_SCRIPT_CATEGORY[@]}"; do
  cat=${CARRY_SCRIPT_CATEGORY[$name]}
  grep -rl "^from ${name} import" grasp_carry/src/grasp_carry/scripts 2>/dev/null | \
    xargs -r sed -i -E "s/^from ${name} import/from grasp_carry.scripts.${cat}.${name} import/"
done
```

- [ ] **Step 4: Verify every moved script still parses and its imports resolve**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
for f in $(find grasp_carry/src/grasp_carry/scripts -name '*.py'); do
  PYTHONPATH=grasp_carry/src python -m py_compile "$f" || echo "SYNTAX FAIL: $f"
done
PYTHONPATH=grasp_carry/src python -c "
import importlib
mods = [
  'grasp_carry.scripts.train.train_carry_actor',
  'grasp_carry.scripts.train.train_carry_actor_reinforce',
  'grasp_carry.scripts.train.train_carry_si',
  'grasp_carry.scripts.train.finetune_carry_diffusion',
  'grasp_carry.scripts.collect.collect_carry_demos',
  'grasp_carry.scripts.collect.collect_carry_bc_rollouts',
  'grasp_carry.scripts.collect.collect_carry_teleop_detour',
  'grasp_carry.scripts.record.record_carry',
  'grasp_carry.scripts.record.record_carry_actor',
  'grasp_carry.scripts.record.record_carry_bc_stg_dist',
  'grasp_carry.scripts.record.record_carry_si',
  'grasp_carry.scripts.record.record_carry_si_video',
  'grasp_carry.scripts.record.record_carry_stg_dist',
  'grasp_carry.scripts.analyze.calibrate_carry',
  'grasp_carry.scripts.analyze.compare_carry_selectors',
  'grasp_carry.scripts.analyze.eval_carry_actor',
  'grasp_carry.scripts.analyze.eval_carry_si',
  'grasp_carry.scripts.analyze.probe_carry_qstg',
  'grasp_carry.scripts.analyze.rollout_carry_diff_stats',
  'grasp_carry.scripts.analyze.run_bc_stg_guided',
  'grasp_carry.scripts.analyze.verify_carry_qstg_condb',
  'grasp_carry.scripts.analyze.analyze_mu_jump_bimodal',
  'grasp_carry.scripts.analyze.analyze_mu_sigma_highrisk',
  'grasp_carry.scripts.analyze.evaluate_stg_deadline',
  'grasp_carry.scripts.analyze.evaluate_stg_deadline_cdf',
]
for m in mods:
    importlib.import_module(m)
print(f'{len(mods)} scripts imported OK')
"
```

Expected: no `SYNTAX FAIL` lines, then `25 scripts imported OK`. (Importing a script module as a
library runs its top-level code but not its `if __name__ == '__main__':` block, so this only
confirms every module-level import resolves — it doesn't execute each script's CLI logic. If any
module has import-time side effects that fail under plain `import` — e.g. GUI backend init — note
it here, drop that one script from the `mods` list, and fall back to `python -m py_compile` plus a
manual read-through for it instead.)

- [ ] **Step 5: Spot-check a couple of `--help` invocations end-to-end**

```bash
PYTHONPATH=grasp_carry/src python -m grasp_carry.scripts.train.train_carry_actor --help
PYTHONPATH=grasp_carry/src python -m grasp_carry.scripts.analyze.eval_carry_actor --help
```

Expected: argparse help text prints, exit code 0, no traceback.

- [ ] **Step 6: Commit**

```bash
git add grasp_carry/
git commit -m "$(cat <<'EOF'
refactor(grasp_carry): move 25 root entry-point scripts into scripts/{train,collect,record,analyze}/

- categorize by verified output type (checkpoints, datasets, mp4, stats/plots)
- rewrite bare cross-script imports (e.g. record_carry_actor -> record_carry) to
  fully-qualified grasp_carry.scripts.<category>.<name> module paths
EOF
)"
```

---

## Task 3: Move `tests/` into `grasp_carry/tests/` and get pytest green

**Files:**
- Create: `grasp_carry/tests/` (via `git mv` of `tests/*.py`)
- Modify: `grasp_carry/tests/test_si_loop.py` (bare `train_carry_si` import)

**Interfaces:**
- Consumes: everything produced by Task 1 and Task 2

- [ ] **Step 1: Move the test files**

```bash
mkdir -p grasp_carry/tests
git mv tests/test_policy.py grasp_carry/tests/test_policy.py
git mv tests/test_gripper.py grasp_carry/tests/test_gripper.py
git mv tests/test_env.py grasp_carry/tests/test_env.py
git mv tests/test_reward.py grasp_carry/tests/test_reward.py
git mv tests/test_render.py grasp_carry/tests/test_render.py
git mv tests/test_config.py grasp_carry/tests/test_config.py
git mv tests/test_ddpo.py grasp_carry/tests/test_ddpo.py
git mv tests/test_dstg.py grasp_carry/tests/test_dstg.py
git mv tests/test_si_loop.py grasp_carry/tests/test_si_loop.py
git mv tests/test_stg_reward.py grasp_carry/tests/test_stg_reward.py
```

- [ ] **Step 2: Rewrite imports (subpackage, flat, and the one bare cross-script import)**

```bash
grep -rl 'from src\.grasp_carry\.' grasp_carry/tests | xargs -r sed -i -E 's/from src\.grasp_carry\./from grasp_carry./g'
grep -rl 'from src\.' grasp_carry/tests | xargs -r sed -i -E 's/from src\./from grasp_carry./g'
```

Edit `grasp_carry/tests/test_si_loop.py` line 20, from:

```python
from train_carry_si import (compute_returns, compute_step_rewards,
```

to:

```python
from grasp_carry.scripts.train.train_carry_si import (compute_returns, compute_step_rewards,
```

(`test_render.py`'s `from record_carry import draw_env, render_frame, view_limits` was already
caught by Task 2 Step 3's loop since it scanned `tests/` too — but that ran before this task moved
the file. Re-check it here: `grep -n "^from record_carry" grasp_carry/tests/test_render.py` should
now show `from grasp_carry.scripts.record.record_carry import ...`. If it still shows the bare
form, apply the same fix as above.)

- [ ] **Step 3: Run the full test suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
PYTHONPATH=grasp_carry/src python -m pytest grasp_carry/tests/ -v
```

Expected: all tests pass (same pass/fail set as running `pytest tests/` did before this reorg
started — if any test was already failing/skipped before Task 1, it's fine for it to still be in
that state now; the bar is "no *new* failures from the move").

- [ ] **Step 4: Commit**

```bash
git add grasp_carry/tests/ tests/
git status  # confirm tests/ is now empty of tracked files
git commit -m "$(cat <<'EOF'
refactor(grasp_carry): move tests/ into grasp_carry/tests/, fix imports

- rewrite src.grasp_carry.X / src.X imports and the bare train_carry_si /
  record_carry cross-imports test_si_loop.py and test_render.py relied on
- full grasp_carry/tests/ suite passes under grasp_carry/src on PYTHONPATH
EOF
)"
```

---

## Task 4: Move data dirs, add packaging files, delete the empty root `src/`/`tests/`

**Files:**
- Create: `grasp_carry/data/`, `grasp_carry/checkpoints/`, `grasp_carry/results/`, `grasp_carry/outputs/` (via `git mv`/`mv` of the root dirs)
- Create: `grasp_carry/pyproject.toml`, `grasp_carry/README.md`
- Delete: root `src/`, `tests/` (should already be empty of tracked files after Tasks 1-3)

**Interfaces:**
- Produces: `grasp_carry` installable-package metadata other tooling (editors, future `pip install -e`) can rely on.

- [ ] **Step 1: Move the data/output directories**

These are gitignored (bulk/output data), so a plain filesystem move is enough — there's nothing
for `git mv` to track:

```bash
mv data grasp_carry/data
mv checkpoints grasp_carry/checkpoints
mv results grasp_carry/results
mv outputs grasp_carry/outputs
```

- [ ] **Step 2: Confirm root `src/` and `tests/` have no tracked files left, then remove them**

```bash
git ls-files src/ tests/   # expect empty output
rm -rf src tests
```

If this prints anything, stop and investigate — it means Tasks 1-3 missed a file.

- [ ] **Step 3: Create `grasp_carry/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "grasp_carry"
version = "0.1.0"
description = "GraspCarry2D — hidden-property-conditioned pick-and-carry environment and JAX/Haiku training stack"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

(`dependencies` stays empty on purpose — the root `requirements.txt` is the single pinned-version
source for this stack, per the Global Constraints above.)

- [ ] **Step 4: Create `grasp_carry/README.md`**

```markdown
# grasp_carry

GraspCarry2D — 은닉 물성(마찰·질량 등) 기반 2D 파지·운반 환경과 그 위의 JAX/Haiku 학습 스택.

## 구조

- `src/grasp_carry/` — 환경(`env.py`, `gripper.py`, `config.py`), 스크립트 정책(`policy.py`),
  학습 라이브러리 코드(`ddpo.py`, `diffusion_act.py`, `conditional_unet1d.py`, `reward.py`,
  `carry_stg_reward.py`, `train_carry_dstg.py`, `train_carry_predictor.py`, `train_carry_qstg.py`,
  `eval_carry.py`, `networks.py`)
- `src/grasp_carry/scripts/` — 실행 진입점. `train/`(정책 학습), `collect/`(데모/롤아웃 수집),
  `record/`(mp4 영상 녹화), `analyze/`(체크포인트 평가·비교·진단)로 기능별 분류
- `tests/` — pytest 스위트
- `data/`, `checkpoints/`, `results/`, `outputs/` — 대용량/산출물 (git 미추적)

## 실행

의존성은 레포 루트의 `requirements.txt`(JAX 스택 공유). Docker `jax` 서비스가
`PYTHONPATH=/workspace/grasp_carry/src`를 설정해두므로 별도 설치 없이 바로 import된다.

```bash
python -m grasp_carry.scripts.train.train_carry_actor --help
python -m grasp_carry.scripts.record.record_carry --help
```

로컬(비-Docker)에서 실행할 때는 `PYTHONPATH=grasp_carry/src`를 직접 설정한다.
```

- [ ] **Step 5: Verify nothing under the repo root still references the old flat paths**

```bash
grep -rln "from src\.\|from src import\|^import src\b" --include="*.py" . | grep -v '^\./archive/\|^\./references/\|^\./square_assembly/\|^\./mani_sim/'
```

Expected: empty output (Task 5's Docker/doc edits are separate and don't hit `.py` files).

- [ ] **Step 6: Commit**

```bash
git add grasp_carry/ src tests
git commit -m "$(cat <<'EOF'
refactor(grasp_carry): move data/checkpoints/results/outputs under grasp_carry/, add packaging files

- add grasp_carry/pyproject.toml (src-layout, deps left to root requirements.txt)
  and grasp_carry/README.md
- delete the now-empty root src/ and tests/
EOF
)"
```

---

## Task 5: Update Docker + DOCKER.md for `grasp_carry/`, final Group-1 verification

**Files:**
- Modify: `docker/jax.Dockerfile`
- Modify: `DOCKER.md`

**Interfaces:**
- None (infra/doc-only task; no code interface changes)

- [ ] **Step 1: Add `PYTHONPATH` to the jax image**

Edit `docker/jax.Dockerfile`. Current tail:

```dockerfile
COPY docker/jax-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
```

Insert a `PYTHONPATH` line before `ENTRYPOINT`, matching the pattern already used in
`docker/torch.Dockerfile`:

```dockerfile
COPY docker/jax-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# grasp_carry 소스는 이미지에 굽지 않음 — 런타임에 bind-mount로 들어오고, PYTHONPATH가
# 그 경로를 잡아주므로 `import grasp_carry`가 바로 됨(mani_sim/torch.Dockerfile과 동일 패턴).
ENV PYTHONPATH=/workspace/grasp_carry/src

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
```

- [ ] **Step 2: Update `DOCKER.md`'s service table and example commands**

Find this table row:

```markdown
| `jax` | 리포지토리 루트 (`train_carry_*.py`, `record_carry_*.py`, `src/grasp_carry/` 등) | JAX + Haiku + Optax |
```

Replace with:

```markdown
| `jax` | `grasp_carry/` (`src/grasp_carry/scripts/`, `src/grasp_carry/` 등) | JAX + Haiku + Optax |
```

Find this line (in the "실행 (CPU)" section):

```markdown
컨테이너 안 셸에 들어가면 평소처럼 `python train_carry_actor.py ...` 같은 명령을 그대로 실행하면 됩니다.
```

Replace with:

```markdown
컨테이너 안 셸에 들어가면 `python -m grasp_carry.scripts.train.train_carry_actor ...` 같은 명령을 실행하면 됩니다.
```

- [ ] **Step 3: Rebuild-free sanity check (no Docker daemon required)**

```bash
docker build -f docker/jax.Dockerfile -t grasp-carry-check --target base . 2>&1 | tail -5 || \
  echo "NOTE: docker not available/buildable in this environment — verify the Dockerfile change by inspection only"
```

If Docker isn't runnable here, at minimum diff-review the Dockerfile change and confirm the syntax
is valid (`ENV KEY=value`, one statement per line, no typos in the path).

- [ ] **Step 4: Full Group-1 regression check**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
PYTHONPATH=grasp_carry/src python -m pytest grasp_carry/tests/ -v
```

Expected: same pass/fail set as Task 3 Step 3 (nothing regressed from the data-dir move or Docker edits).

- [ ] **Step 5: Commit**

```bash
git add docker/jax.Dockerfile DOCKER.md
git commit -m "$(cat <<'EOF'
build(docker): point jax image PYTHONPATH at grasp_carry/src

- docker/jax.Dockerfile: add ENV PYTHONPATH=/workspace/grasp_carry/src,
  matching mani_sim/torch.Dockerfile's existing bind-mount-only convention
- DOCKER.md: update jax service table row and the example run command to the
  new python -m grasp_carry.scripts.train.train_carry_actor form
EOF
)"
```

---

## Task 6: Create the minimal `pointmass/` skeleton

**Files:**
- Create: `pointmass/pointmass_core.py` (via `git mv` of root `pointmass_core.py`)
- Create: `pointmass/pointmass_notebook.ipynb` (via `git mv` of `archive/pointmass_notebook.ipynb`)

**Interfaces:**
- None — `pointmass_core.py` is moved unmodified; nothing in the reorganized `grasp_carry/` imports
  from it anymore after Task 1 Step 6, so this move cannot break anything currently working.

- [ ] **Step 1: Confirm nothing still imports the root `pointmass_core`**

```bash
grep -rln "pointmass_core" --include="*.py" . | grep -v '^\./archive/\|^\./references/'
```

Expected: empty (Task 1 Step 6 already redirected the one real dependency to
`grasp_carry.networks`). If anything else shows up here, resolve it before moving the file.

- [ ] **Step 2: Move the files**

```bash
mkdir -p pointmass
git mv pointmass_core.py pointmass/pointmass_core.py
git mv archive/pointmass_notebook.ipynb pointmass/pointmass_notebook.ipynb
```

- [ ] **Step 3: Verify `pointmass_core.py` still parses on its own**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
python -m py_compile pointmass/pointmass_core.py && echo "pointmass_core.py OK"
```

- [ ] **Step 4: Commit**

```bash
git add pointmass/ archive/ pointmass_core.py
git commit -m "$(cat <<'EOF'
refactor(pointmass): create minimal pointmass/ skeleton for future revival

- move pointmass_core.py (root) and pointmass_notebook.ipynb (archive/) into
  pointmass/, unmodified — this is the origin-point research track, not dead
  code; see docs/superpowers/specs/2026-08-27-repo-reorg-design.md
- the rest of archive/'s pointmass diagnostic scripts stay put; triaging them
  into pointmass-only vs. grasp_carry-shared is deferred to a follow-up session
EOF
)"
```

---

## Task 7: Rename `mani_sim/` → `square_assembly/`

**Files:**
- Create: `square_assembly/` (via `git mv` of `mani_sim/`, then internal `git mv` of `src/mani_sim/` → `src/square_assembly/`)
- Modify: every `.py`/`.md`/`.yaml`/`.toml` file under the renamed tree that mentions `mani_sim`

**Interfaces:**
- Produces: importable package `square_assembly` (same public surface as `mani_sim` had, just
  renamed — this task does not change any function/class signature, only the package name).

- [ ] **Step 1: Rename the top-level directory and the inner package directory**

```bash
git mv mani_sim square_assembly
git mv square_assembly/src/mani_sim square_assembly/src/square_assembly
```

- [ ] **Step 2: Delete stale build artifacts (regenerate later, don't rename them)**

```bash
rm -rf square_assembly/src/mani_sim.egg-info
find square_assembly -name '__pycache__' -exec rm -rf {} +
```

- [ ] **Step 3: Blanket-rewrite `mani_sim` → `square_assembly` across the renamed tree**

```bash
grep -rl 'mani_sim' square_assembly --include="*.py" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.toml" | \
  xargs -r sed -i 's/mani_sim/square_assembly/g'
```

This is a plain substring replace, so `mani_sim_external` → `square_assembly_external` as a side
effect too — that's fine, Task 8 replaces that whole concept anyway.

Skip `square_assembly/mani_sim_external/` in this pass if it still exists at this point (it's
untracked/gitignored, and Task 8 handles it explicitly) — the `grep -rl` above only touches
`--include`-matched tracked-pattern files, but double check with:

```bash
git status square_assembly/mani_sim_external/   # expect: nothing (still untracked, ignored)
```

- [ ] **Step 4: Review the diff for anything the blanket replace mangled**

```bash
git diff --stat square_assembly/
git diff square_assembly/pyproject.toml square_assembly/environment.yml
```

Confirm `square_assembly/pyproject.toml` now has `name = "square_assembly"` and
`square_assembly/environment.yml` now has `name: square_assembly`. Spot-check 2-3 of the
`.py` files under `square_assembly/src/square_assembly/` to confirm `from mani_sim.` became
`from square_assembly.` cleanly (no double-replacement, no mangled unrelated text).

- [ ] **Step 5: Verify Python syntax across the renamed tree (no local torch env available)**

```bash
find square_assembly -name '*.py' -not -path '*/mani_sim_external/*' | while read -r f; do
  python3 -m py_compile "$f" || echo "SYNTAX FAIL: $f"
done
```

Expected: no `SYNTAX FAIL` lines. (This only checks syntax, not that imports resolve — there's no
local `torch`/`robosuite` install to actually import against. Note this limitation rather than
claiming full verification.)

- [ ] **Step 6: Commit**

```bash
git add square_assembly/ mani_sim
git status  # confirm mani_sim/ is gone, square_assembly/ holds everything
git commit -m "$(cat <<'EOF'
refactor(square_assembly): rename mani_sim/ to square_assembly/ (task name, not stack name)

- git mv mani_sim -> square_assembly, src/mani_sim -> src/square_assembly
- rewrite all internal `from mani_sim.X import` / `import mani_sim` references
  to square_assembly across .py/.md/.yaml/.toml under the renamed tree
- pyproject.toml and environment.yml `name` fields updated to square_assembly
  (no local conda env named mani_sim existed to rename — file-only change)
- delete stale mani_sim.egg-info / __pycache__ build artifacts
EOF
)"
```

---

## Task 8: Dissolve `mani_sim_external/` into a normal tracked module

**Files:**
- Create: `square_assembly/src/square_assembly/datasets/replay_buffer.py` (moved + rewritten from `square_assembly/mani_sim_external/piper_capstone/replay_buffer.py`)
- Modify: `square_assembly/src/square_assembly/datasets/zarr_dataset.py`, `square_assembly/src/square_assembly/datasets/normalization.py`, `square_assembly/src/square_assembly/utils/task_utils.py`
- Modify: `square_assembly/.gitignore`
- Delete: `square_assembly/mani_sim_external/`

**Interfaces:**
- Produces: `square_assembly.datasets.replay_buffer.ReplayBuffer` (same class, same
  `create_from_path(zarr_path, mode="r")` classmethod, same `.data[key]`/`.episode_ends`-shaped
  read-only interface it had at its old import location).

- [ ] **Step 1: Move the file into the package (plain `mv` — it was never git-tracked)**

```bash
mv square_assembly/mani_sim_external/piper_capstone/replay_buffer.py \
   square_assembly/src/square_assembly/datasets/replay_buffer.py
rm -rf square_assembly/mani_sim_external
```

- [ ] **Step 2: Rewrite `replay_buffer.py`'s header docstring**

The old docstring described this as a stand-in for external FLARE-vendored code the original repo
never actually committed. Since it's now a normal internal module (not framed as "external"
anymore), replace the docstring in
`square_assembly/src/square_assembly/datasets/replay_buffer.py` — keep the technical content
(what interface it satisfies, why it's read-only, the real zarr structure it was verified
against), drop the "vendoring"/"external" framing:

```python
"""Minimal read-only ReplayBuffer used by ZarrSequenceDataset
(square_assembly/src/square_assembly/datasets/zarr_dataset.py).

Reimplemented from scratch against the actual zarr layout in use
(square_dataset/square_image_v15.zarr: data/{action,action_mode,agentview_image,
robot0_eef_pos,robot0_eef_quat,robot0_eye_in_hand_image,robot0_gripper_qpos},
meta/episode_ends) rather than ported from any external package — only the narrow
interface ZarrSequenceDataset needs is implemented: `.episode_ends` (1D array of
per-episode cumulative end indices) and `.data[key]` (that key's full array across all
transitions, indexable). Write operations (append/add_episode) are intentionally not
implemented — read-only only.
"""

import zarr
```

(Keep the `class ReplayBuffer:` body below this docstring exactly as it was — only the module
docstring changes.)

- [ ] **Step 3: Fix the three call sites to use a normal import**

In `square_assembly/src/square_assembly/datasets/zarr_dataset.py`: replace

```python
_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[3] / "square_assembly_external" / "piper_capstone"
if str(_PIPER_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPER_CAPSTONE_DIR))
from replay_buffer import ReplayBuffer  # noqa: E402
```

with:

```python
from square_assembly.datasets.replay_buffer import ReplayBuffer
```

(remove the now-unused `sys` import from this file if nothing else in it uses `sys`; check with
`grep -n "sys\." square_assembly/src/square_assembly/datasets/zarr_dataset.py` before removing the
import line). Also update the module docstring's mention of
`mani_sim_external/piper_capstone/replay_buffer.py`(now `square_assembly_external/...` after
Task 7's blanket rename) to `square_assembly/src/square_assembly/datasets/replay_buffer.py`, and
drop the word "vendoring" from that sentence since it no longer is.

In `square_assembly/src/square_assembly/datasets/normalization.py`, replace the equivalent block
(lines ~57-63 pre-rename):

```python
import sys
from pathlib import Path

piper_capstone_dir = Path(__file__).resolve().parents[3] / "square_assembly_external" / "piper_capstone"
if str(piper_capstone_dir) not in sys.path:
    sys.path.insert(0, str(piper_capstone_dir))
from replay_buffer import ReplayBuffer
```

with:

```python
from square_assembly.datasets.replay_buffer import ReplayBuffer
```

(again drop the `import sys`/`from pathlib import Path` lines only if nothing else in the function
uses them — check first).

In `square_assembly/src/square_assembly/utils/task_utils.py`, apply the identical replacement to
its equivalent block (lines ~120-126 pre-rename).

- [ ] **Step 4: Remove the gitignore exception**

Edit `square_assembly/.gitignore` (currently, post-Task-7-rename, it will read
`square_assembly_external/` with a comment above it). Delete both the `square_assembly_external/`
line and its preceding comment block that explains why it used to be excluded.

- [ ] **Step 5: Verify the file is now trackable and syntax is clean**

```bash
python3 -m py_compile square_assembly/src/square_assembly/datasets/replay_buffer.py
python3 -m py_compile square_assembly/src/square_assembly/datasets/zarr_dataset.py
python3 -m py_compile square_assembly/src/square_assembly/datasets/normalization.py
python3 -m py_compile square_assembly/src/square_assembly/utils/task_utils.py
git status square_assembly/src/square_assembly/datasets/replay_buffer.py
```

Expected: no `py_compile` errors, and `git status` shows `replay_buffer.py` as untracked-and-addable
(not ignored) — confirming the `.gitignore` edit took effect.

- [ ] **Step 6: Commit**

```bash
git add square_assembly/
git commit -m "$(cat <<'EOF'
refactor(square_assembly): absorb mani_sim_external/ into a normal tracked module

- replay_buffer.py was never externally vendored (the upstream FLARE code it
  stood in for was itself gitignored in the original repo and never available
  to copy) — it's a from-scratch reimplementation of a narrow read-only
  interface, so treat it as ordinary internal code
- move it to square_assembly/src/square_assembly/datasets/replay_buffer.py,
  rewrite its docstring to drop the "vendoring" framing
- replace the sys.path-hack imports in zarr_dataset.py, normalization.py, and
  task_utils.py with a normal `from square_assembly.datasets.replay_buffer
  import ReplayBuffer`
- drop the square_assembly_external/ gitignore exception — this file is now
  git-tracked like any other source file, closing the "depended on but never
  committed" risk
EOF
)"
```

---

## Task 9: Update Docker + docs for the rename, final verification

**Files:**
- Modify: `docker-compose.yml`, `docker/torch.Dockerfile`, `.devcontainer/torch/devcontainer.json`
- Modify: `DOCKER.md`, `docs/ARCHITECTURE.md`, `docs/ADR.md`
- Modify: `grasp_carry/src/grasp_carry/scripts/train/train_carry_si.py`, `grasp_carry/src/grasp_carry/train_carry_predictor.py` (comment-only mentions of `mani_sim`)

**Interfaces:**
- None (infra/doc-only task)

- [ ] **Step 1: `docker-compose.yml`**

Find:

```yaml
  torch:
    build:
      context: .
      dockerfile: docker/torch.Dockerfile
    image: efms-gym/torch:latest
    working_dir: /workspace/mani_sim
```

Replace `working_dir: /workspace/mani_sim` with `working_dir: /workspace/square_assembly`. Leave
the `torch:` service key name and `image: efms-gym/torch:latest` untouched (service naming is
stack-based, out of scope per Global Constraints; the `efms-gym/` image-name prefix is a separate,
pre-existing staleness this plan doesn't touch).

- [ ] **Step 2: `docker/torch.Dockerfile`**

Replace:

```dockerfile
# 3D robomimic square 스택 (PyTorch + robosuite + mujoco). From: mani_sim/requirements.txt
```

with:

```dockerfile
# 3D robomimic square 스택 (PyTorch + robosuite + mujoco). From: square_assembly/requirements.txt
```

Replace:

```dockerfile
COPY mani_sim/requirements.txt ./mani_sim/requirements.txt
RUN pip install --no-cache-dir -r mani_sim/requirements.txt
```

with:

```dockerfile
COPY square_assembly/requirements.txt ./square_assembly/requirements.txt
RUN pip install --no-cache-dir -r square_assembly/requirements.txt
```

Replace:

```dockerfile
# mani_sim 소스는 이미지에 굽지 않음 — 런타임에 bind-mount로 들어오고,
# PYTHONPATH가 그 경로를 잡아주므로 `import mani_sim`이 바로 됨.
# (무거운 설치 레이어 뒤에 둬서, 이 값만 바꿀 때 위 pip install 캐시가 안 깨지게 함)
ENV PYTHONPATH=/workspace/mani_sim/src
```

with:

```dockerfile
# square_assembly 소스는 이미지에 굽지 않음 — 런타임에 bind-mount로 들어오고,
# PYTHONPATH가 그 경로를 잡아주므로 `import square_assembly`가 바로 됨.
# (무거운 설치 레이어 뒤에 둬서, 이 값만 바꿀 때 위 pip install 캐시가 안 깨지게 함)
ENV PYTHONPATH=/workspace/square_assembly/src
```

- [ ] **Step 3: `.devcontainer/torch/devcontainer.json`**

Replace:

```json
  "name": "torch (mani_sim)",
```

with:

```json
  "name": "torch (square_assembly)",
```

Replace:

```json
  "workspaceFolder": "/workspace/mani_sim",
```

with:

```json
  "workspaceFolder": "/workspace/square_assembly",
```

- [ ] **Step 4: `DOCKER.md`**

Table row — replace:

```markdown
| `torch` | `mani_sim/` | PyTorch + robosuite + mujoco |
```

with:

```markdown
| `torch` | `square_assembly/` | PyTorch + robosuite + mujoco |
```

Section 6 rule — replace:

```markdown
1. 계속 쓸 패키지라고 판단되면 → 해당 스택의 `requirements.txt`(`jax`는 루트 `requirements.txt`, `torch`는 `mani_sim/requirements.txt`)에 **버전을 명시해서** 추가
```

with:

```markdown
1. 계속 쓸 패키지라고 판단되면 → 해당 스택의 `requirements.txt`(`jax`는 루트 `requirements.txt`, `torch`는 `square_assembly/requirements.txt`)에 **버전을 명시해서** 추가
```

Any other bare `mani_sim` mentions in `DOCKER.md` (e.g. the `# PyTorch 스택 (mani_sim) 진입`
comment near the `docker compose run --rm torch bash` example) — update those to
`square_assembly` too:

```bash
grep -n "mani_sim" DOCKER.md
```

and fix each remaining hit the same way (path/name reference, not narrative rewrite).

- [ ] **Step 5: `docs/ARCHITECTURE.md`**

Replace the three-line directory-tree entry:

```
mani_sim/                        # robomimic/Diffusion Policy 서브프로젝트 (ADR-007) —
                                  #   별도 스택(PyTorch), 별도 conda env. src/mani_sim/
                                  #   구조·환경설정은 mani_sim/README.md 참고.
```

with:

```
square_assembly/                 # robomimic/Diffusion Policy 서브프로젝트 (ADR-007) —
                                  #   별도 스택(PyTorch), 별도 conda env. src/square_assembly/
                                  #   구조·환경설정은 square_assembly/README.md 참고.
```

Leave every other line in this file untouched — the pointmass 1차-목표 narrative rewrite is out
of scope for this plan (see Global Constraints).

- [ ] **Step 6: `docs/ADR.md`**

Within ADR-007's body only, replace each literal path/name mention:

- `mani_sim/` → `square_assembly/` (heading and every prose mention of the bare path)
- `github.com/Leejw221/manipulation_simulator` stays as-is (that's the *original* upstream repo
  name, not something we renamed)
- `mani_sim/environment.yml` → `square_assembly/environment.yml`
- conda env name `mani_sim` (in "독립 conda env `mani_sim`") → `square_assembly`
- `mani_sim/src/mani_sim/factory.py` → `square_assembly/src/square_assembly/factory.py`
- `envs/robomimic/factory.py` mention stays relative, no change needed
- `PyTorch(`mani_sim/`)` → `PyTorch(`square_assembly/`)`

Do not touch the surrounding rationale/tradeoff/unresolved-questions prose — only the literal
path/name tokens listed above.

```bash
grep -n "mani_sim" docs/ADR.md
```

should show zero hits when done (everything in this file's ADR-007 section was a path/name
reference — confirm no hits remain anywhere else in the file either, since `mani_sim` shouldn't
appear outside ADR-007).

- [ ] **Step 7: Comment mentions inside `grasp_carry/`**

In `grasp_carry/src/grasp_carry/scripts/train/train_carry_si.py`, find the comment near line 538
(`# iteration만 저장하면 그 정점을 잃는다 — mani_sim/train_si.py에서 겪은 뒤 고친 것과`) and
replace `mani_sim/train_si.py` with `square_assembly/train_si.py`.

In `grasp_carry/src/grasp_carry/train_carry_predictor.py`, find the comment near line 14
(`horizon 방식으로 롤아웃한다(mani_sim/3D 태스크와 같은 패턴). 2026-08-10:`) and replace
`mani_sim/3D 태스크` with `square_assembly/3D 태스크`.

- [ ] **Step 8: Confirm no dangling `mani_sim` references remain anywhere tracked**

```bash
grep -rln "mani_sim" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.toml" --include="Dockerfile*" --include="*.json" . | grep -v '^\./references/'
```

Expected: empty output. (`references/` is read-only external material and may legitimately mention
unrelated things — excluded from this check, not expected to match anyway.)

- [ ] **Step 9: Final full-repo verification**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate self-improving-gym
PYTHONPATH=grasp_carry/src python -m pytest grasp_carry/tests/ -v
find square_assembly -name '*.py' -not -path '*/mani_sim_external/*' | while read -r f; do
  python3 -m py_compile "$f" || echo "SYNTAX FAIL: $f"
done
git status
```

Expected: grasp_carry tests still green, no square_assembly syntax failures, `git status` shows a
clean tree (everything committed) with no stray untracked files besides the usual `__pycache__`/
gitignored output dirs.

- [ ] **Step 10: Write the experiments log entry**

Per `CLAUDE.md`'s process rule, log this reorg in `experiments/` using `experiments/LOG_TEMPLATE.md`
as the format. Create `experiments/2026-08-27-repo-reorg.md`:

```markdown
# Experiment Log: 2026-08-27 Repo Reorg (grasp_carry / pointmass / square_assembly)

## 1. 개요 (Overview)
- **수정 대상**: 레포 전체 소스 재배치 — `src/`, `tests/`, 루트 스크립트 25개, `pointmass_core.py`,
  `mani_sim/` 전체
- **참조 소스**: 없음 (references/ 이식 아님, 순수 내부 리팩터링)
- **목표**: GraspCarry2D 코드를 `mani_sim/`과 동일한 자기완결형 구조의 `grasp_carry/`로 통합하고,
  pointmass를 위한 최소 골격(`pointmass/`)을 만들고, `mani_sim/`을 태스크 이름
  `square_assembly/`로 리네임

## 2. 변경 내역 (Changes)
- [x] `grasp_carry/src/grasp_carry/`: 기존 `src/grasp_carry/` 서브패키지 + flat `src/*.py` 9개
  통합, `pointmass_core.py`의 TIMER 네트워크 함수를 `networks.py`로 포팅
- [x] `grasp_carry/src/grasp_carry/scripts/{train,collect,record,analyze}/`: 루트 스크립트 25개를
  실제 출력 형태(체크포인트/데이터셋/mp4/통계) 기준으로 분류해 이동
- [x] `grasp_carry/tests/`, `grasp_carry/{data,checkpoints,results,outputs}/`: 루트에서 이동
- [x] `pointmass/`: `pointmass_core.py` + `archive/pointmass_notebook.ipynb` 최소 이동
- [x] `mani_sim/` → `square_assembly/`: 폴더·내부 패키지·pyproject/environment.yml name 필드 전체 리네임
- [x] `mani_sim_external/` 해체: `replay_buffer.py`를 일반 내부 모듈로 흡수, git 추적 시작
- [x] Docker(`jax.Dockerfile`, `docker-compose.yml`, `torch.Dockerfile`,
  `.devcontainer/torch/devcontainer.json`) 및 문서(`DOCKER.md`, `ARCHITECTURE.md`, `ADR.md`) 경로 갱신

## 3. 참조 로직 (Reference Logic)
해당 없음

## 4. 가설 및 예상 결과 (Hypothesis)
동작 변경 없는 순수 재배치 — `grasp_carry/tests/` pytest 스위트가 이전과 동일하게 통과하면 성공.

## 5. 결과 기록
- `pytest grasp_carry/tests/ -v`: [실행 결과 기록]
- 후속 세션 과제: `COMMANDS.md` 재작성, `docs/ARCHITECTURE.md`/`docs/ADR.md` 내용(1차 목표 서사)
  갱신, `archive/`의 pointmass 진단 스크립트 pointmass-only vs grasp_carry-공유 분류
```

Fill in the actual pytest output in the "결과 기록" section before committing.

- [ ] **Step 11: Commit**

```bash
git add docker-compose.yml docker/torch.Dockerfile .devcontainer/torch/devcontainer.json \
        DOCKER.md docs/ARCHITECTURE.md docs/ADR.md \
        grasp_carry/src/grasp_carry/scripts/train/train_carry_si.py \
        grasp_carry/src/grasp_carry/train_carry_predictor.py \
        experiments/2026-08-27-repo-reorg.md
git commit -m "$(cat <<'EOF'
docs(infra): update Docker paths and doc references for the square_assembly rename

- docker-compose.yml, torch.Dockerfile, .devcontainer/torch: /workspace/mani_sim
  -> /workspace/square_assembly (service names jax/torch unchanged, stack-based
  naming per existing ARCHITECTURE.md policy)
- DOCKER.md, ARCHITECTURE.md, ADR-007: path/name string references only, no
  content rewrite (deferred to a follow-up session)
- fix two mani_sim comment mentions inside grasp_carry/ scripts
- log the whole reorg in experiments/2026-08-27-repo-reorg.md per CLAUDE.md
EOF
)"
```

---

## Explicitly deferred (do not attempt in this plan)

- `COMMANDS.md` full rewrite (it currently documents already-archived pointmass scripts, not
  `grasp_carry/`'s actual scripts — needs new content, not path substitution)
- `docs/ARCHITECTURE.md` / `docs/ADR.md` narrative content (1차 목표 서사, pointmass 재현 목표
  language) — only literal path strings were touched in Task 9
- `archive/`'s remaining pointmass diagnostic scripts (`archive/*.py`, `archive/src/*`) — triaging
  pointmass-only vs. grasp_carry-shared and designing a shared/common folder is a separate,
  larger piece of work
- `requirements.txt` dependency trimming — moot, since no dependency's only consumer was deleted
  (pointmass_core.py was moved, not removed)
