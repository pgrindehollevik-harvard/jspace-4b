# Claim ledger

This ledger prevents a fluent draft from outrunning the evidence. Each empirical claim
must cite an artifact field or a verified source before submission.

| Claim | Status before v2 run | Required evidence |
|---|---|---|
| The 4B light intervention has a J-specific positive-control effect. | Established historical result. | `results/calibration.json`, light-band rung; exact paired vectors in `results/calibration_state.json`. |
| The 4B intervention is not selective on ordinary text. | Established historical result. | Light-band Wikitext top-1 agreement in `results/calibration.json`. |
| The stock 8B lens is a valid measurement tool under the registered eval. | Unknown. | `extension/results/qwen3-8b/scale_gate.json:lens_validation`. |
| The stock 14B lens is a valid measurement tool under the registered eval. | Unknown. | `extension/results/qwen3-14b/scale_gate.json:lens_validation`. |
| Selective J-space causal effects emerge by 14B. | Preregistered prediction, not a fact. | Every 14B `required_checks` value true. |
| The onset lies between 4B and 8B or between 8B and 14B. | Unknown and conditional. | Complete 8B and 14B gates, interpreted exactly per preregistration. |
| Parameter scale caused any observed boundary. | Not licensed by this design. | Must not be claimed; lens quality is an unresolved alternative. |
| CoT protects reasoning at a larger Qwen3 scale. | Not tested in v2. | Must not be claimed unless a separately preregistered follow-up is run. |

## Allowed conclusion templates

- **14B pass, 8B fail:** “Under a fixed stock-lens pipeline, the complete intervention-
  validity signature appeared at 14B but not at 8B or the historical 4B baseline.”
- **8B and 14B pass:** “The registered signature appeared by 8B and replicated at 14B.”
- **Both fail after valid lenses:** “Increasing Qwen3 scale to 14B did not rescue the
  intervention’s complete validity signature.”
- **Lens failure:** “The stock lens failed the registered prerequisite, so this experiment
  does not identify whether the mechanism is absent or merely unmeasured at that scale.”

Do not replace “under this pipeline,” “associated,” or “appeared” with causal language
about scale. Do not turn a stopped stage into missing-at-random data.
