# NeurIPS 2026 workshop submission plan

**Target:** *Interpretability as a Science: Toward Rigorous Foundations for Understanding
LLMs*. **Format:** short paper, at most five pages excluding references and appendices.
**Deadline:** September 1, 2026, Anywhere on Earth.

Official sources:

- Workshop CFP: https://interpscience.github.io/cfp
- NeurIPS dates: https://neurips.cc/Conferences/2026/Dates
- Source study: https://transformer-circuits.pub/2026/workspace/index.html

## Paper promise

Working title: **When Does a Global-Workspace Intervention Become Selective? A
Preregistered Scale Test in Qwen3**.

The paper is about when an intervention is sufficiently validated to support a mechanistic
claim. The 4B stop is a first result, not a failed prelude: it shows causal effect without
ordinary-language selectivity. The fixed 8B/14B test asks whether that mismatch disappears
with scale. The paper will not imply that parameter count caused an observed difference,
and it will not present the unrun CoT difficulty grid as evidence.

## Five-page structure

1. **Introduction and claim (0.6 page):** causal interventions need positive efficacy and
   negative/selectivity controls; a preregistered stop can itself produce evidence.
2. **Method (1.3 pages):** J-space intervention, matched random control, fixed band transfer,
   models, data, and complete validity gate.
3. **Results (1.4 pages):** one scale figure and one gate/controls table, including the known
   4B baseline and all stopped stages.
4. **Interpretation (0.8 page):** scale-consistent boundary versus lens-quality alternative;
   what the experiment licenses and what it does not.
5. **Limitations and conclusion (0.9 page):** one model family, stock lenses, small number of
   scales, no CoT claim without a gate pass.

Appendix: full preregistration, exact item-level vectors, bootstrap details, environment and
artifact manifests, deviations, and the original 4B gate ladder.

## Deadline path

- **August 25:** freeze v2 preregistration/config/runner; stage pinned assets and data.
- **August 25–26:** compatibility smoke and complete 8B stages.
- **August 26–27:** compatibility smoke and complete 14B stages.
- **August 27–28:** locked analysis, figure/table, and claim audit.
- **August 28–30:** five-page draft and appendix; verify every citation against its source.
- **August 30–31:** independent reviewer-style critique, reciprocal-reviewer check, author
  review, anonymity and formatting pass.
- **September 1 AoE:** submit. Do not also place the paper under review at another NeurIPS
  workshop.

The CFP requires double-blind submission and a reciprocal reviewer unless an exception is
granted. Acceptance is non-archival, so the same work may later be expanded for an archival
venue subject to that venue's policy.
