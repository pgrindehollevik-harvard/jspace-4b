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
