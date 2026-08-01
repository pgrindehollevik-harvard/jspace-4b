# Learnings — the experiment, explained end to end

*A running consolidation of the lessons written during the work, so the person submitting
this can genuinely explain every choice. Final version completed after the results land.*

## 1. The object under study

- **Residual stream:** each token position carries a vector (~2560 dims in Qwen3-4B)
  through 36 layers; attention/MLP blocks read it and add back into it. It is the model's
  working state.
- **Jacobian lens:** a per-layer linear map J_ℓ = E[∂h_final/∂h_ℓ] (averaged over a text
  corpus, including *future* positions) that transports a mid-layer vector into the final
  layer's basis, where the model's own unembedding can read it. It answers: *what is this
  activation disposed to make the model say, now or later?*
- **J-space:** at any position, only ~10–25 lens directions are strongly active (<10% of
  activation variance) — a small, privileged, reportable set atop a sea of automatic
  processing. The paper's claim: this plays the role Global Workspace Theory assigns to
  conscious access.
- **Why ablation:** a readout can be epiphenomenal (steam off the engine). Zeroing the
  projection onto the top-k active directions and watching *what breaks* is the causal
  test. The exclusion rule (never ablate tokens in the clean top-10 predictions) keeps
  the test aimed at internal reasoning rather than at the model's mouth.

## 2. Why the design looks like this

- **Pre-registration as a commitment device:** every dial that could be bent after seeing
  results (bands, k, sample sizes, floor rules, trim ladder, ambiguity resolutions) was
  frozen in a tagged commit *first*. The assignment grades design fairness over
  convenient conclusions.
- **The estimand is a triple difference.** P_D = [R(CoT,J) − R(direct,J)] −
  [R(CoT,random) − R(direct,random)]. Retention ratios kill baseline differences; the
  CoT−direct difference kills "ablation hurts everything"; the random-arm subtraction
  kills "you damaged the network, of course it got worse." Each layer of the estimand
  exists to execute one specific rival explanation.
- **Per-slot norm matching** is what makes the random arm a real control: same positions,
  same layers, same displacement magnitude — only the *direction* differs.
- **Hypothesis-blind calibration + stop-gate:** ablation strength was selected using only
  positive-control collapse, text-prediction selectivity, and coherence — never the
  CoT-vs-direct gap — so the intervention could not be tuned toward the hypothesis. And
  the gate has teeth: it failed, and the grid did not run.
- **Floors and power, stated up front:** AIME direct-answering was expected to be at
  floor (retention undefined); survival P(correct ablated | correct clean) is the
  floor-robust endpoint; with 60 AIME problems only collapse-sized effects are
  detectable, so the primary H2 contrast is GSM8K-vs-MATH-500 where power exists.

## 3. What the calibration actually found (the first real result)

- Clean Qwen3-4B scores **0.42** on the paper's two-hop multihop control (frontier models:
  near ceiling) — a 4B model largely lacks silent two-hop composition, so the
  pre-registered order-ops fallback control was used.
- Across four rungs (two bands × k=10, k=5, light band): the J-ablation drop on the
  control is real, significant, and — in the light band — perfectly J-specific
  (norm-matched random damage costs 0.0 points). **The lens finds causally load-bearing
  directions.**
- But *selectivity* failed everywhere: 34–42% of ordinary next-token predictions change
  under J-ablation (vs 16–27% under matched random; paper: "only mildly perturbed").
  On a small model, verbalizable directions are entangled with ordinary prediction.
- Two candidate readings, distinguishable by the pre-registered refit: (a) small models
  route routine prediction *through* their workspace (a scaling claim about the
  architecture of cognition); (b) the off-the-shelf lens (final-layer target) smears
  workspace directions with last-block artifacts (a tooling claim). The penultimate-target
  refit gives (b) its best registered shot before (a) is allowed as a conclusion.

## 4. Systems lessons (the unglamorous 60%)

- **The deadliest bugs run clean and lie.** The two criticals from adversarial review — a
  difficulty-sorted control arm and a bootstrap that deduplicated its resamples — would
  have produced beautiful, wrong conclusions. Both were invisible at the
  single-generation level; they lived in *which comparisons the data licenses*.
- **On Apple Silicon, syncs cost more than math:** removing per-token `.item()` calls and
  CPU-fallback QR bought 4.4× throughput (1.5 → 6.6 tok/s).
- **A busy laptop is a hostile scheduler:** macOS update preparation SIGKILLed the
  8-GB-wired process on ~90 s cycles regardless of sandbox, priority, or launchd
  packaging. The durable answer was not to win the fight but to make death cheap:
  stage-level checkpoints, then 64-token chunk-resumable generation with exact sampler
  and exemption-set replay, plus progress-aware restart supervision.
- **Instrument the failure, not the theory:** the breakthrough forensics were exit codes
  (137 = SIGKILL), flat memory telemetry (killing the OOM theory), and the pending-update
  discovery — each eliminating a hypothesis the previous fix had been built on.

## 5. Open questions to be answered by the final data

*(completed after the refit ladder / grid)*
