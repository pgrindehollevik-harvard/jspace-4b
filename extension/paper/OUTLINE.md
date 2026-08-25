# Anonymous five-page paper outline

Working title: **When Does a Global-Workspace Intervention Become Selective? A
Preregistered Scale Test in Qwen3**

## Abstract (target 140–170 words)

Problem: an intervention can be causally potent yet too indiscriminate to support the
mechanistic interpretation attached to it. Prior result: at Qwen3-4B, fixed light-band
J-space ablation selectively damaged a reasoning control relative to matched random
damage, but failed ordinary-text preservation, triggering a preregistered stop. Method:
transfer that exact intervention by normalized depth to Qwen3-8B and 14B; require stock-
lens validation, two paired causal tests, a matched-damage ratio, Wikitext preservation,
and coherence as one complete gate. Result: **fill only from locked analysis**. Conclusion:
**choose the corresponding claim-ledger template**, emphasizing what the design licenses.

## 1. Introduction (0.6 page)

- Interpretability claims need both intervention efficacy and discriminant validity.
- The source study proposes the J-space as a selective global workspace and supplies a
  concrete causal intervention [verified `gurnee2026verbalizable`].
- Our 4B replication found causal structure but failed selectivity; the registered stop
  prevented a downstream CoT claim.
- Contribution: a fixed, no-tuning scale test plus a general pattern for treating a stop
  rule as scientific evidence rather than failed setup.

## 2. Method (1.3 pages)

- Qwen3 8B/14B, pinned stock lenses, historical 4B comparator.
- Fixed half-open band mapping `[14,20)/36`: 8B 14..19, 14B 15..21; `k=10`.
- Clean/J/random positive control, teacher-forced Wikitext selectivity, coherence audit.
- Full conjunctive gate and ordered stopping; primary 14B, secondary 8B.
- Exact paired tests and locked 10,000-resample intervals.

## 3. Results (1.4 pages)

- Insert `extension/analysis/scale_gate.pdf`.
- Insert compact gate table from `scale_table.csv`.
- Report every stopped stage; never hide unavailable downstream metrics.
- Use the claim ledger’s matching conclusion template.

## 4. What the evidence licenses (0.8 page)

- Separate causal efficacy, damage matching, ordinary-text preservation, and coherence.
- Discuss stock-lens quality as a rival account of scale-associated differences.
- Explain why v2 still does not answer the original CoT-protection question unless a model
  clears the measurement gate and a separate registered outcome study is run.

## 5. Limitations and conclusion (0.9 page)

- One model family, two new scales, one transferred band, stock lenses, small control set.
- Reused historical evidence is descriptive, not retroactively confirmatory.
- Threshold gates are operational criteria, not universal definitions of mechanism.
- Main takeaway should concern disciplined causal measurement, not a grand scale law.

## Appendix

Full preregistration/config digest, model/lens/data commits, item-level hit vectors,
bootstrap algorithm, environment/job records, displacement norms, coherence outputs,
deviations, and the archived 4B ladder.
