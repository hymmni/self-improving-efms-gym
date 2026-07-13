# Variance-Reward Experiments — Summary

Testing the claim: *using the **variance** of the steps-to-go distribution in the
reward (not just its expectation) improves self-improvement over the SI-EFM
baseline.* Priority (advisor): a clear result at small scale over a large one.

Reward definitions (`src/reward.py`):
- baseline (eq 1.1): `r = μ_t − μ_{t+1}`
- ours (eq 1.2): `r = α·(μ_t − μ_{t+1}) + β·(σ²_t − σ²_{t+1})/σ²_t`  (baseline = β=0)

Two testbeds: a **multimodal** obstacle map (left/right detour → bimodal
steps-to-go, `src/multimodal_env.py`) for the observation experiments, and the
**standard** pointmass map (calibrated dynamics, room to improve) for the reward
experiments. Predictors use bin_size=1 so E[STG] is calibrated in real steps.

Reproduce any experiment: `python -m src.run_experiment --exp {e0,e1,e1b,e2,sweeps,e3}`.

---

## Results

### E0 — bimodal distribution collapse ✅
From the decision point the predicted steps-to-go distribution is **bimodal**
(mass at the short-left ~41 and long-right ~46 routes, σ²≈5.2); once the agent
commits to a side it **collapses to a single sharp mode** (σ²→0). Both routes
verified. → *variance tracks commitment / uncertainty resolution.*

### E1 — three situations ✅
| situation | μ behaviour | σ² behaviour |
|---|---|---|
| goal approach (success) | ↓ to ~0 | ↓ to ~0 |
| near failure (bias-perturbed) | stays high | **spikes to ~500** at the struggle steps |
| multimodal collapse | ~flat then ↓ | ↓ (bimodal → unimodal) |

→ *failure is marked by a variance spike; success by variance collapse.*

### E1b — Δμ vs Δσ² independence ✅ (novelty defense)
Naive global Pearson **r = −0.91**, but this is a two-cluster artifact:
- **certain steps** (σ²≤1, the majority): **r ≈ −0.02** — Δσ²≈0 while Δμ carries
  the signal → the two are **independent**; variance is inert.
- **uncertain steps** (~2%, the decision/failure region): Δσ² is large and Δμ can
  reverse sign — variance fires exactly where the expectation is ambiguous.

→ *variance adds information rather than duplicating the expectation; it is
active precisely in the states that matter.*

### E2 — baseline vs ours (main comparison) ⚠️ no robust improvement
| condition | baseline (β=0) | ours | note |
|---|---|---|---|
| near-ceiling start (SR≈0.87) | 0.915 | **0.947** | +3pts, 5/6 seeds, paired t=1.76 (marginal) |
| degraded start, β sweep | **0.844** | 0.796 (β=.5) / 0.818 (β=1) / 0.837 (β=2) | every β>0 on/below baseline, high seed variance |

Decision items: **2.1** change_rate (0.853) ≥ ratio (0.824) — change_rate slightly
more stable. **2.2** eps ∈ {1e-6, 1e-3, 1e-1} all stable, **no NaN/divergence**.
**2.3** no β beats β=0.

### E3 — policy improvement vs predictor quality ⚠️ consistent with E2
| predictor (MAE) | baseline | ours | ours − base |
|---|---|---|---|
| f=0.10 (20.07) | 0.774 | 0.739 | −0.034 |
| f=0.30 (19.73) | 0.836 | 0.829 | −0.007 |
| f=0.50 (19.71) | 0.832 | 0.759 | −0.073 |
| f=1.00 (19.64) | 0.826 | 0.806 | −0.020 |

Across all predictor qualities ours ≤ baseline. (Caveat: on the random-goal
standard map the predictor MAE plateaus at ~20 across fractions, so the quality
axis is narrow.)

---

## Conclusion

- **Variance is a real, distinct, failure-predictive signal** (E0/E1/E1b): it
  collapses on commitment, spikes before failure, and is statistically
  independent of the expectation in the common regime while activating in the
  decision/failure states. This is the core novel finding and it is solid.
- **But adding it to the REINFORCE reward does not robustly help** (E2/E3): at
  this small scale the variance term neither reliably improves nor clearly harms
  policy learning — the effect is within seed noise and often slightly negative
  across β and predictor quality. This **empirically supports the advisor's own
  caution** ("리워드에 요소 과다 통합 시 상충·자기부정") against over-integrating
  signals into the reward.
- **Implication / next step**: the value of the variance signal is in
  **detection** — the E4 use case (μ-jump + σ²-rise as a failure / human-intervention
  trigger, and data-quality assessment) — rather than as a policy-reward term.
  A cleaner reward test would need a map where a *well-calibrated* predictor
  (like the multimodal one, MAE≈0.15) coexists with genuine room to improve;
  here the calibrated map was at ceiling and the improvable map had a weak
  predictor.

## Artifacts
`results/observe/` (E0/E1/E1b), `results/e2/`, `results/sweeps/` (2.1–2.3),
`results/e3/` — plots + CSV/JSON. Predictor checkpoints in `checkpoints/{mm,std}/`
(gitignored, regenerable via `src/train_predictor.py`).
