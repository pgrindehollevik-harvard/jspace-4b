# Does chain-of-thought still protect against J-space ablation when problems get hard?

CS 2881 (Harvard, Fall 2026) Homework 0. Extends the GSM8K chain-of-thought result of
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee et al., Anthropic, July 2026) to a difficulty ladder — GSM8K → MATH-500 → AIME — on
`Qwen/Qwen3-4B`, run entirely on one Apple M4 Pro (MPS).

**Report:** [`report.pdf`](report.pdf) *(in progress)*

## Question

The paper finds that J-space ablation (zeroing the residual stream's projection onto the top-10
most active J-lens directions, every position, across a mid-layer band) devastates direct
answering on GSM8K but largely spares chain-of-thought answering — written reasoning substitutes
for the internal workspace. We test whether that protection **persists, erodes, or collapses** as
problem difficulty increases.

## Repository structure

| Path | Contents |
|---|---|
| `docs/PREREGISTRATION.md` | Hypotheses, full design, analysis plan — frozen at tag `prereg-v1` **before** any experiment ran |
| `docs/DEVIATIONS.md` | Timestamped log of any deviation from the pre-registration |
| `src/hw0/` | Implementation: ablation hooks, generation loop, datasets, grading, diagnostics, grid runner |
| `tests/` | Unit tests for the intervention (exemption rule, projection correctness, clean-path identity) |
| `configs/` | Experiment grid configuration |
| `results/` | Raw per-generation JSONL records + derived tables (committed for reproducibility) |
| `report/` | Report source; built PDF is copied to `report.pdf` in the root |

## Key external artifacts

- Paper: https://transformer-circuits.pub/2026/workspace/index.html (arXiv:2607.15495)
- Companion code (lens fitting/application; **contains no ablation code** — we implement that here):
  https://github.com/anthropics/jacobian-lens
- Pre-fitted Jacobian lens for Qwen3-4B: [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  file `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` (fit on the identical checkpoint)
- Model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (non-thinking mode for all conditions)
- Datasets: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (test);
  [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (test);
  AIME 2024 I+II = [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024),
  AIME 2025 I+II = [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) (config `default`)
- Grading: [`math-verify`](https://github.com/huggingface/math-verify)

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch transformers datasets accelerate \
    'huggingface_hub[cli]' matplotlib 'math-verify[antlr4_13_2]'
git clone https://github.com/anthropics/jacobian-lens vendor/jacobian-lens
uv pip install --python .venv/bin/python -e vendor/jacobian-lens
```

Model (~8 GB) and lens (~437 MB) download from HF Hub on first run.

## Reproducing the results

Every step is resumable (stage- or chunk-level checkpoints); rerunning a completed step
loads its checkpoint. Order matters — each stage gates the next:

```bash
.venv/bin/python -m pytest tests/           # 7 intervention-correctness tests (loads model)
.venv/bin/python -u -m hw0.validate_lens    # gate 1: J-lens must beat logit lens
.venv/bin/python -u -m hw0.calibrate        # gate 2: hypothesis-blind band calibration +
                                            #   positive-control stop-gate -> results/calibration.json
.venv/bin/python -u -m hw0.fit_lens         # (contingency, prereg 5.7) penultimate-target lens refit
.venv/bin/python -u -m hw0.run_grid         # main grid (only if stop_gate == PASS), resumable JSONL
.venv/bin/python -u -m hw0.analyze          # grading, bootstrap CIs, trend model, figures
cd report && pdflatex report.tex            # rebuild the report (copy to ./report.pdf)
```

For unattended runs on a laptop, `scripts/supervisor.sh` chains calibration -> gate ->
grid with progress-aware crash restarts (see `docs/LEARNINGS.md` section 4 for why that
exists). The principal tables and figures regenerate from `results/` via `hw0.analyze`;
the calibration table comes from `results/calibration.json`.
