# Pre-registration: J-space ablation × chain-of-thought × problem difficulty

**Recorded before any main experiment was run.** Frozen at the git tag `prereg-v1`.
Any deviation from this document is logged in `docs/DEVIATIONS.md` with a timestamp and rationale.

- Paper being extended: Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits (Anthropic), July 2026. https://transformer-circuits.pub/2026/workspace/index.html
- Task brief: replicate and extend the paper's GSM8K chain-of-thought ablation result on Qwen3-4B across a difficulty ladder (GSM8K, MATH-500, AIME)
- Model: `Qwen/Qwen3-4B` (original hybrid checkpoint, apache-2.0)
- Lens: pre-fitted Jacobian lens `neuronpedia/jacobian-lens` → `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` (fit on this exact checkpoint per its config.yaml)
- Compute: one Apple M4 Pro, 48 GB unified memory, PyTorch MPS

## 1. Background and question

The paper shows (§3.5.2, Fig. 24) that ablating a model's J-space — zeroing the residual stream's
projection onto the k=10 most strongly activated J-lens vectors, at every token position, across a
band of mid-network layers — leaves shallow capabilities intact but collapses internal multi-step
reasoning. Critically, **GSM8K solved with explicit chain of thought is substantially more robust to
this ablation than the same problems answered directly**. The authors interpret written CoT as
externalization: the page substitutes for the internal workspace.

The paper tests this only on GSM8K (grade-school arithmetic) and only on large production models
(Sonnet 4.5). **Our question: does CoT's protective effect persist as problems get harder
(GSM8K → MATH-500 → AIME) on a small open model (Qwen3-4B)?**

## 2. Hypotheses (recorded before experiments)

**H1 (replication).** On GSM8K, J-space ablation hurts direct answering substantially more than
CoT answering, and this gap is J-space-specific: the *random-differenced protection*

> P_D = [R(CoT, J-abl) − R(direct, J-abl)] − [R(CoT, random-abl) − R(direct, random-abl)]

is positive, where R(mode, condition) = accuracy(ablated)/accuracy(clean) is retention.
Prediction: P_D(GSM8K) ≥ +0.15.

**H2 (extension, primary).** CoT protection **erodes** with difficulty: P_D declines from GSM8K to
MATH-500, and R(CoT, J-abl) declines across GSM8K → MATH-500 → AIME and across MATH-500 levels 1→5.
Mechanistic rationale (paper §9.3): externalization is bandwidth-limited — each individual step of a
harder problem itself requires more un-externalized workspace content, so writing steps out
protects less as steps get harder.
Predicted rough pattern: R(CoT, J-abl) ≈ 0.80–0.90 on GSM8K, 0.60–0.80 on MATH-500, ≤ 0.60 on AIME.

**Falsified if:** (a) P_D(GSM8K) ≤ 0 — no J-specific protection, or CoT more fragile (H1 fails);
(b) the primary H2 contrast P_D(GSM8K) − P_D(MATH-500) is ≤ 0 with its 95% CI excluding meaningful
decline, and the within-MATH level gradient of R(CoT, J-abl) is flat or rising — protection
*persists*; (c) the random control degrades as much as J-ablation everywhere (nothing J-specific
was measured). Each opposite outcome is directly visible in the headline figure (§8).

**We commit to reporting whichever outcome occurs**, including "Qwen3-4B loses coherence before any
selective workspace effect is measurable" (the paper's own small-model warning about Haiku 4.5) —
that too answers the research question and will be written up as the finding if it is what happens.

## 3. Experimental conditions

Two prompting modes × three intervention conditions per dataset, all with
`enable_thinking=False` (Qwen3 non-thinking mode), sampling T=0.7, top-p=0.8, top-k=20, min-p=0
(Qwen's recommended non-thinking settings; greedy decoding is discouraged for this model):

- **CoT mode:** "Please reason step by step, and put your final answer within \boxed{}."
- **Direct mode:** "Give only the final answer within \boxed{}. Do not show any working."
  Cap: 64 new tokens (raised from 32 so boxed LaTeX answers are never clipped).
- **Conditions:** clean (no intervention) / J-space ablation (§5) / per-slot norm-matched
  random-direction ablation (§6).

We deliberately do **not** use Qwen3's `<think>` mode: it would confound difficulty with an
inference-mode change, and its 5–30× longer traces are infeasible under ablated decode throughput.
This is a scope limitation, recorded here.

CoT token caps: GSM8K 640, MATH-500 768, AIME 1536. Any cell where >15% of *clean* generations hit
the cap is flagged cap-truncated; truncation-adjusted accuracy is reported alongside.

## 4. Datasets and sample sizes

| Dataset | Source | Sample | CoT arms (seeds/problem) | Direct arms |
|---|---|---|---|---|
| GSM8K | `openai/gsm8k` config `main`, **test** split (1319) | 150 problems, seed 0 | clean 1, J 1, random 1 (n=100 subset) | 1 each (random n=100) |
| MATH-500 | `HuggingFaceH4/MATH-500`, split `test` (500) | 150, stratified 30/level | clean 1, J 1, random 1 (n=100) | 1 each (random n=100) |
| AIME | `Maxwell-Jia/AIME_2024` (30 = 2024 I+II, MIT) + `yentinglin/aime_2025` config `default` (30 = 2025 I+II) | all 60 | clean 2, J 2, random 1 | 4 each |

The **exact AIME source** (stated per the task brief): AIME 2024 I & II from `Maxwell-Jia/AIME_2024`
(rows keyed `2024-I-k`/`2024-II-k`) and AIME 2025 I & II from `yentinglin/aime_2025` (config
`default`, split `train`). Answers are integers 0–999.

Same problems appear in every condition of a dataset (**pairing is at problem level only** — token
streams diverge at the first ablated position, so seed-sharing across conditions buys nothing and
is not claimed). Calibration pilot problems (§7) come from the GSM8K **train** split — disjoint
from all test material by construction.

**Power / MDE (stated up front).** With n=150 paired problems, a 20-point retention gap within a
dataset is detectable with ~94% power (α=.05, two-sided; paired analysis stronger). The primary H2
contrast P_D(GSM8K) − P_D(MATH-500) is a difference of differences of ratios; its SE is ≈ 11–13
points by problem-level bootstrap — we can distinguish *collapse of protection* from *persistence*,
not subtle slopes. AIME (60 problems, the entire 2024+2025 population) supports only large effects;
it is **descriptive/secondary**, entering H2 through the trend model and the survival endpoint, and
is governed by the floor rule (§9). The AIME random arm runs 1 seed/problem and enters unpaired via
problem-level cluster bootstrap (pre-specified here).

## 5. J-space ablation protocol

Faithful to §3.5.2 of the paper: *"at each token position, across a band of layers, we identify the
k=10 most strongly activated J-lens vectors and zero out the residual stream's projection onto
each"*, with the exclusion rule: *"we do not ablate any tokens that appear in the top-10 tokens of
a clean forward pass."*

The paper leaves implementation details unstated. **Pre-registered resolutions:**

1. **Activation ranking:** at each (position, band layer ℓ), lens readout logits are computed as
   `lm_head(final_rmsnorm(J_ℓ h))` — reusing the model's own norm and unembedding, never
   materializing W_U·J_ℓ. The top-10 token ids by readout logit are selected, **recomputed
   independently at every (position, layer)**.
2. **Exclusion rule:** a parallel clean forward pass (own KV cache, fed the *same* token stream as
   the ablated pass) supplies the clean top-10 next-token ids at each position; selected ids in
   that set are exempt. Sequencing (unit-tested): the clean pass's result for position t is
   available before the ablated pass's band hooks fire at position t, during both prefill and decode.
3. **Ablation directions:** v_t = J_ℓᵀ(γ ⊙ W_U[t]) where γ is the final RMSNorm gain — the linear
   part of the readout for token t.
4. **Projection:** joint orthogonal projection — QR-decompose the ≤10 direction vectors and remove
   h's component in their span (h ← h − QQᵀh). Sensitivity check (30 GSM8K problems): sequential
   per-vector projection, to show robustness to this choice.
5. **Coverage:** ablation applies to all prompt positions (prefill) and every generated token. No
   norm restoration afterward.
6. **Layer band:** the paper reports layers as percent depth; Sonnet 4.5's workspace band is
   L38–L92 (%), with "light" = first third. Percent-depth transfer to Qwen3-4B (36 layers) gives:
   **primary band candidates {14–24, 14–31}, light band 14–19**, final layers reserved as motor
   region. We will attempt the paper's §4.1 band diagnostics (lens next-token accuracy by layer,
   readout-logit kurtosis, top-1 autocorrelation, effective rank), **time-boxed to 4 hours**; if
   ambiguous, the percent-depth mapping above is adopted automatically.
7. **Known fidelity deviation, logged now:** the neuronpedia lens targets the **final** layer
   (jlens default) rather than the paper's preferred penultimate-layer variant (paper: variants
   "fairly consistent"). If the positive-control gate (§7) fails, this is the first-listed
   explanation candidate, and the contingency is fitting our own penultimate-target lens on ~100
   wikitext prompts before concluding anything.

## 6. Controls

1. **Per-slot norm-matched random ablation** (paper §A.23 protocol): at each (position, layer)
   compute the displacement the J-ablation *would* apply, then remove an equal-norm component along
   random directions at that same slot. Kills the "generic damage" explanation at trend level, not
   just per dataset. Achieved per-layer ‖Δh‖/‖h‖ logged for both arms and reported.
2. **Clean baselines** for every dataset × mode (retention denominators).
3. **Selectivity control:** teacher-forced top-1 next-token agreement with the clean model on 50
   held-out wikitext-103 sequences, under J and random ablation (paper Fig. 22 replication).
4. **Positive control / stop-gate (§7):** multi-hop reasoning must collapse under J-ablation
   before the main grid is allowed to run.
5. **Within-MATH-500 level 1→5 gradient:** difficulty axis with dataset format held constant —
   a trend there cannot be a format artifact. Pre-registered fallback difficulty axis if AIME is
   floor-gated: GSM8K → MATH L1–3 → MATH L4–5.
6. **Direct-compliance audit:** fraction of direct-mode outputs showing visible reasoning before
   the box; if compliance < 90%, results analyzed with and without non-compliant items.
7. **Artifact accounting** per cell: {no-parseable-answer, truncated-at-cap, degenerate
   (max 4-gram repetition fraction > 0.5 or no EOS), wrong-answer} — separated so ablation-induced
   format breakage cannot masquerade as the predicted gradient. No-answer scores as incorrect
   (primary), with a sensitivity analysis excluding those items.
8. **Dose-response anchor:** GSM8K light-band cells (n=75) reproduce the paper's strength ordering
   (light < primary), anchoring comparability to Fig. 22. Reported in the main text.

## 7. Calibration pilot and stop-gate (hypothesis-blind)

Before any main-grid generation, on GSM8K **train** problems (n=30) and the companion repo's eval
sets:

- **Lens validation gate:** J-lens must beat vanilla logit lens (`use_jacobian=False`) at ranking
  the annotated intermediates on `lens-eval-order-ops.json` (55 items), as in the paper.
- **Band/strength calibration:** choose the primary band from {14–24, 14–31} (k=10) to maximize
  the positive-control drop **subject to** degenerate-output rate < 10% on pilot CoT generations
  and wikitext top-1 match > 80%. The CoT-vs-direct accuracy gap is **never consulted** during
  calibration. Graduated ladder if no candidate passes: k=10→5, then light band 14–19, then
  max-drop-with-coherence, flagged.
- **Positive-control stop-gate:** on the 93-item multi-hop set (`lens-eval-multihop.json`;
  fallback: order-ops, if clean Qwen3-4B is not near-ceiling on multihop under the chat template —
  checked first), the J-ablation drop must be statistically significant **and ≥ 2× the matched
  random-control drop** (Fisher exact). If the gate fails at every ladder rung, we stop and write
  up the negative replication — the main grid does not run.

## 8. Analysis plan

- **Grading:** `math-verify` 0.9.0 on the last `\boxed{}`; GSM8K numeric fallback = last number in
  output; AIME = integer equality. AIME pass@1 = mean over samples, clustered by problem.
- **Primary endpoints:** P_D(GSM8K) > 0 (H1); P_D(GSM8K) − P_D(MATH-500) > 0 (H2 primary
  contrast). Co-primary framing: absolute-point difference-in-differences alongside retention
  ratios. 95% CIs via 10,000-resample problem-level bootstrap.
- **Secondary (floor-robust):** paired survival P(correct ablated | correct clean) with exact
  McNemar, per dataset × mode; the AIME endpoint of record.
- **Trend model:** logistic regression, correct ~ mode × condition × ordinal difficulty
  (GSM8K=1, MATH L1–5 = 2–6, AIME=7), problem-clustered SEs; the registered trend statistic is the
  J-vs-random contrast of the mode × condition × difficulty interaction.
- **Covariate check:** retention vs CoT length within dataset, separating "more ablated steps"
  from "more load per step" — reported, not used for selection.
- **Floor rule:** any cell with clean accuracy < 10% is floor-unmeasurable: excluded from ratio
  endpoints, reported via survival only. (Expected: aime-direct.)
- **Headline figure:** four retention-vs-difficulty lines — CoT/J, direct/J, CoT/random,
  direct/random. Persistence and erosion are visually distinct outcomes of the same plot.
  Main text carries this figure + one controls table; everything else goes to the appendix.

## 9. Compute budget and trim ladder

Planning throughput (to be re-anchored on day-1 measured numbers): clean decode ~15 tok/s, ablated
~5 tok/s (dual forward passes + per-band-layer readout are memory-bandwidth-bound), ablated
generations budgeted 1.4× longer than clean. Rough stage totals at those rates: GSM8K ≈ 7 h,
MATH-500 ≈ 10 h, AIME ≈ 17 h, calibration + controls ≈ 4 h — ~38 h worst case, under the 48 h hard
cap; expected less if measured throughput is better. Runs are unattended, resumable (JSONL
append), and **interleaved problem-wise across all cells within a dataset**, so an interruption
leaves balanced partial data in every arm. Priority order GSM8K → MATH-500 → AIME so H1 survives
any truncation.

**Pre-registered trim ladder** (applied in order if day-1 measured throughput projects > 40 h):
(1) drop GSM8K light-band add-on; (2) random arms n→75 (GSM8K, MATH); (3) MATH-500 → 25/level
(n=125); (4) AIME J/clean CoT arms 2→1 seed/problem.

## 10. Risks acknowledged in advance

Coherence collapse on a 4B model (reported as the finding if even the light band degenerates);
AIME floor in non-thinking mode (floor rule; fallback difficulty axis); pre-fitted lens inadequacy
(validation gate; refit contingency); MPS instability with dual KV caches (day-1 smoke tests,
resumable checkpoints); contamination of GSM8K/MATH in pretraining corpora (all inference is on
within-model retention ratios, noted as limitation); direct-mode non-compliance (audit).
