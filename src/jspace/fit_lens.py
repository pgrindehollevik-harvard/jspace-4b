"""Fit a penultimate-target Jacobian lens on Qwen3-4B (prereg section 5.7 contingency).

The pre-fitted neuronpedia lens targets the FINAL layer (jlens default); the paper's
preferred estimator targets the penultimate layer ("omitting the last block ... reduces
noisy artifacts"). This fit follows the paper's recipe: 100 wikitext prompts (README:
quality saturates fast; the paper's own lenses beat logit lens from ~10 prompts),
max_seq_len 128, target_layer=-2. jlens.fit checkpoints per prompt and resumes natively,
so the machine's SIGKILL regime costs at most one prompt of backward passes.

Usage: .venv/bin/python -u -m jspace.fit_lens   (writes results/lens_fit/qwen3-4b_pen_n32.pt)
"""

import os

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts

from jspace.core import MODEL_NAME

OUT_DIR = "results/lens_fit"
CKPT = os.path.join(OUT_DIR, "ckpt.pt")
OUT = os.path.join(OUT_DIR, "qwen3-4b_pen_n32.pt")
N_PROMPTS = 32  # paper: quality saturates fast, beats logit lens from ~10 prompts
DIM_BATCH = 16


def main():
    import faulthandler
    faulthandler.enable()
    os.makedirs(OUT_DIR, exist_ok=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    prompts = load_wikitext_prompts(N_PROMPTS)
    lens = jlens.fit(model, prompts, target_layer=-2, dim_batch=DIM_BATCH,
                     max_seq_len=128, checkpoint_path=CKPT, checkpoint_every=1,
                     resume=True)
    lens.save(OUT)
    print(f"LENS_FIT_COMPLETE {OUT}")


if __name__ == "__main__":
    main()
