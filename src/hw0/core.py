"""Model, tokenizer, and lens setup shared by every stage.

Layer indexing convention: lens Jacobians are keyed by decoder-block index and map the
residual stream at that block's OUTPUT to the final residual stream (jlens fits on block
outputs via ActivationRecorder). Band (a, b) means blocks a..b inclusive.
"""

from dataclasses import dataclass

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"

# Percent-depth transfer of the paper's Sonnet 4.5 workspace band (L38-L92 of 100) to
# Qwen3-4B's 36 blocks; see docs/PREREGISTRATION.md section 5.6.
BAND_PRIMARY_CANDIDATES = [(14, 24), (14, 31)]
BAND_LIGHT = (14, 19)

K_ABLATE = 10          # top-k J-lens vectors ablated per (position, layer)
K_CLEAN_EXEMPT = 10    # clean-pass top-k next-token ids exempt from ablation

# Qwen3-4B recommended non-thinking sampling (generation_config.json / model card).
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)

COT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DIRECT_INSTRUCTION = "Give only the final answer within \\boxed{}. Do not show any working."


@dataclass
class Setup:
    hf: transformers.PreTrainedModel
    tok: transformers.PreTrainedTokenizerBase
    model: "jlens.HFLensModel"
    lens: "jlens.JacobianLens"
    device: str

    @property
    def blocks(self):
        return self.model.layers

    @property
    def final_norm(self):
        return self.hf.model.norm

    @property
    def lm_head(self):
        return self.hf.lm_head


def load(device: str = "mps", dtype=torch.bfloat16) -> Setup:
    import os
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    # HW0_LENS_PATH selects a locally fitted lens (the prereg 5.7 penultimate-target
    # contingency) instead of the pre-fitted hub artifact.
    local = os.environ.get("HW0_LENS_PATH")
    if local:
        lens = jlens.JacobianLens.load(local)
    else:
        lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    return Setup(hf=hf, tok=tok, model=model, lens=lens, device=device)


def chat_prompt(setup: Setup, problem: str, mode: str) -> str:
    instruction = COT_INSTRUCTION if mode == "cot" else DIRECT_INSTRUCTION
    msgs = [{"role": "user", "content": f"{problem}\n\n{instruction}"}]
    return setup.tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
