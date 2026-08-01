# Does chain-of-thought still protect against J-space ablation when problems get hard?

CS 2881 (Harvard, Fall 2026) Homework 0. Extends the GSM8K chain-of-thought result of
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee et al., Anthropic, July 2026) to a difficulty ladder — GSM8K → MATH-500 → AIME — on
`Qwen/Qwen3-4B`, run entirely on one Apple M4 Pro (MPS).

**Report:** [`report.pdf`](report.pdf)

**Result in one line:** a pre-registered negative replication — J-lens directions on
Qwen3-4B are causally load-bearing but not selectively separable from ordinary
prediction, so the stop-gate failed (8 rungs, 2 lenses) and the pre-registered
CoT-vs-difficulty grid was correctly never run.

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
| `docs/DEVIATIONS.md` | Timestamped log of every deviation from the pre-registration |
| `docs/LEARNINGS.md` | The experiment explained end to end, plus systems lessons |
| `src/hw0/` | Implementation: ablation hooks, resumable generation, datasets, grading, calibration, lens fitting, grid runner, analysis, figure |
| `tests/` | 7 unit tests for the intervention (exemption rule, projection correctness, clean-path identity, chunked-resume equivalence) |
| `scripts/` | Crash-tolerant supervisors for unattended runs |
| `results/` | Committed evidence: `calibration.json` + `calibration_pen.json` (the 8-rung ladders, incl. per-item hits in the `_state` files), `lens_validation*.json` (gate 1), `figures/` |
| `report/` | Report source; built PDF is copied to `report.pdf` in the root |

The main grid (`results/grid.jsonl`) intentionally does not exist: the pre-registered
stop-gate failed, so per `docs/PREREGISTRATION.md` section 7 the grid was not run. The
fitted penultimate lens (445 MB) exceeds GitHub's file limit and is not committed;
regenerate it with `hw0.fit_lens` (~5.7 h on an M4 Pro, checkpointed and resumable) —
its validation artifact is committed as `results/lens_validation_pen.json`.

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
git clone https://github.com/anthropics/jacobian-lens vendor/jacobian-lens
uv pip install --python .venv/bin/python -e . -e vendor/jacobian-lens
```

`-e .` installs the `hw0` package and every dependency (torch, transformers>=5.5,
datasets, math-verify, statsmodels, pytest, ...) from `pyproject.toml`. The model
(~8 GB) and the stock lens (~437 MB) download from HF Hub on first run.

## Reproducing the results

Every step is resumable (stage- or chunk-level checkpoints); rerunning a completed step
loads its checkpoint. Order matters — each stage gates the next. This reproduces the
submission's actual path, including the pre-registered contingency:

```bash
.venv/bin/python -m pytest tests/           # 7 intervention-correctness tests (loads model)
.venv/bin/python -u -m hw0.validate_lens    # gate 1 (stock lens) -> results/lens_validation.json
.venv/bin/python -u -m hw0.calibrate        # gate 2: hypothesis-blind ladder + stop-gate
                                            #   -> results/calibration.json  (verdict: FAIL)
.venv/bin/python -u -m hw0.fit_lens         # contingency (prereg 5.7): penultimate lens, ~5.7h
export HW0_LENS_PATH=results/lens_fit/qwen3-4b_pen_n32.pt
.venv/bin/python -u -m hw0.validate_lens    # gate 1 for the refit -> lens_validation_pen.json
.venv/bin/python -u -m hw0.calibrate        # refit ladder -> calibration_pen.json (verdict: FAIL)
unset HW0_LENS_PATH
.venv/bin/python -m hw0.fig_ladder          # the report's figure, from the two calibration JSONs
cd report && pdflatex report.tex && cp report.pdf ../report.pdf
```

`hw0.run_grid` and `hw0.analyze` implement the pre-registered main experiment; they
refuse to run / have nothing to analyze because the stop-gate verdict is FAIL — that
refusal is the submission's result. For unattended runs on a laptop,
`scripts/supervisor.sh` and `scripts/fit_supervisor.sh` chain the stages with
progress-aware crash restarts (`docs/LEARNINGS.md` section 4 explains why). The
report's table and figure regenerate from `results/calibration*.json` via `hw0.fig_ladder`.
