# Deviations from the pre-registration

Every departure from `docs/PREREGISTRATION.md` (tag `prereg-v1`) is logged here with a timestamp
and rationale, before or as it happens — never retroactively edited.

## 1. Lens-validation gate metric refined (2026-07-31)

**What changed:** the gate metric moved from pass@10 (min-over-all-35-layers) on order-ops to
pass@1/pass@5 under the companion repo's official synonym-set convention, reported on both
order-ops and multihop, with the gate = J-lens ≥ logit lens on pass@1 and pass@5 (strictly
better on at least one) on at least one eval.

**Why:** as originally implemented the metric saturated — both lenses scored an identical
0.4909, because over 35 layers nearly every intermediate appears in *some* layer's top-10 for
*either* lens (late layers read out the model's own next-token distribution for both). The
original implementation also omitted the official order-ops synonym expansion (numbers →
digit+word forms, operations → symbol+word forms), understating both lenses equally.

**When decided:** after seeing the saturated aggregate (0.4909 = 0.4909) and per-layer ranks
for one debug item; before seeing any refined aggregate results. This gate is upstream of all
experiments and hypothesis-irrelevant (it compares two readout methods, not any experimental
condition), so the refinement cannot bias H1/H2.

## 2. MATH-500 subset ordering rule specified (2026-07-31)

**What changed:** `load_math500` now orders items round-robin across levels, so every prefix
(the n=100 random arms, trim-ladder n=125/75, and any interrupted run) is level-balanced.

**Why:** the pre-registration (section 4) set the random arms at n=100 without specifying the
subset-selection rule. The original implementation took a level-sorted prefix, which would
have given the MATH-500 random-control arm zero level-5 problems (30/30/30/10/0 across levels
1–5) — confounding the H2 contrast with difficulty, the very axis under study. Caught by
adversarial code review **before any main-grid generation ran**; no data is affected.

## 3. Calibration stop-gate test: paired McNemar instead of unpaired Fisher (2026-07-31)

**What changed:** the positive-control drop significance and the J-vs-random comparison use
exact McNemar on per-item paired outcomes; the pre-registration named "Fisher exact".

**Why:** the multihop items are identical across conditions — the data is paired, and an
unpaired Fisher test discards that structure (anti-conservative or conservative depending on
correlation). Decided before any calibration run completed (the only prior run was killed
mid-flight by a laptop crash with no results written). The ≥2× drop-ratio rule is unchanged.

## 4. Contingency lens fit reduced to 32 prompts (2026-08-01)

**What changed:** the pre-registered penultimate-target refit (section 5.7: "~100 wikitext
prompts") ran with n=32.

**Why:** measured fit throughput on MPS was 10.6 min/prompt (dim_batch=16); 100 prompts
would take ~17h against an Aug 5 deadline. The paper reports lens quality beating the
logit lens from ~10 prompts. Validation of the resulting lens: it passes the lens gate on
multihop (pass@1 0.204 vs 0.183; pass@5 0.366 vs 0.353) but with thinner margins than the
stock n=479 lens and clearly worse order-ops readout — the n=32 fit is undertrained
relative to stock. **Interpretation constraint logged now, before the pen ladder runs:**
if the pen ladder fails its gate, undertraining is confounded with the
penultimate-target hypothesis; only a marked selectivity IMPROVEMENT is strong evidence
(for the final-layer-artifact explanation), and a flat/worse result leaves the
explanation candidate unresolved rather than refuted. The fit checkpoint is retained so
the fit can be extended if the selectivity direction warrants it.
