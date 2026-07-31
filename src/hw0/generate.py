"""Dual-stream generation under intervention.

Sequencing invariant (pre-registered, unit-tested): the clean pass's top-10 next-token ids
for position t are computed BEFORE the intervened pass's band hooks fire at position t —
during prefill (clean prefill first, whole prompt) and at every decode step (clean forward
for the new token first). Both streams consume the SAME sampled tokens; only the intervened
stream's logits choose them.

Sampling is done on CPU with a seeded generator (temperature -> top-k -> top-p), identical
code path for clean and intervened runs so decoding differences cannot confound conditions.
"""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from hw0.core import Setup, SAMPLING, K_CLEAN_EXEMPT
from hw0.ablation import JSpaceAblator


@dataclass
class GenResult:
    text: str
    token_ids: list[int]
    n_new: int
    hit_cap: bool
    norm_log: dict[int, float] = field(default_factory=dict)  # per-layer mean ||dh||/||h||


def _eos_ids(tok) -> set[int]:
    # Qwen3 registers two EOS ids: <|im_end|> (151645) and <|endoftext|> (151643).
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
           tok.convert_tokens_to_ids("<|endoftext|>")}
    return {t for t in ids if t is not None}


def _sample(logits: torch.Tensor, gen: torch.Generator | None) -> int:
    if gen is None:  # greedy (used only by calibration controls, never the main grid)
        return int(logits.argmax().item())
    logits = logits.float().cpu() / SAMPLING["temperature"]
    topk = logits.topk(SAMPLING["top_k"])
    probs = torch.softmax(topk.values, dim=-1)
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(0) - sorted_probs < SAMPLING["top_p"]
    keep[0] = True
    idx = torch.multinomial(sorted_probs * keep, 1, generator=gen)
    return topk.indices[order[idx]].item()


@torch.no_grad()
def generate(setup: Setup, prompt: str, max_new_tokens: int, seed: int,
             ablator: JSpaceAblator | None = None, greedy: bool = False) -> GenResult:
    """Generate under no intervention (ablator=None) or under an installed ablator."""
    tok, hf = setup.tok, setup.hf
    ids = tok(prompt, return_tensors="pt").input_ids.to(setup.device)  # [1, S]
    gen = None if greedy else torch.Generator().manual_seed(seed)
    eos = _eos_ids(tok)

    if ablator is None:
        return _generate_single(setup, ids, max_new_tokens, gen, eos)

    main_cache, clean_cache = DynamicCache(), DynamicCache()
    new_ids: list[int] = []
    try:
        # Clean prefill (hooks silent): top-10 next-token ids at every prompt position.
        ablator.enabled = False
        out = hf(ids, past_key_values=clean_cache, use_cache=True)
        clean_top = out.logits[0].topk(K_CLEAN_EXEMPT, dim=-1).indices     # [S, 10]

        # Intervened prefill over the same prompt.
        ablator.exempt_ids = clean_top
        ablator.enabled = True
        out = hf(ids, past_key_values=main_cache, use_cache=True)
        ablator.enabled = False

        next_id = _sample(out.logits[0, -1], gen)
        while True:
            new_ids.append(next_id)
            if next_id in eos or len(new_ids) >= max_new_tokens:
                break
            step = torch.tensor([[next_id]], device=setup.device)
            # Clean stream consumes the sampled token first -> exemptions for this position.
            out_c = hf(step, past_key_values=clean_cache, use_cache=True)
            ablator.exempt_ids = out_c.logits[0].topk(K_CLEAN_EXEMPT, dim=-1).indices
            ablator.enabled = True
            out = hf(step, past_key_values=main_cache, use_cache=True)
            ablator.enabled = False
            next_id = _sample(out.logits[0, -1], gen)
    finally:
        ablator.enabled = False  # never leak an enabled hook to a later (clean) forward

    return _result(setup, new_ids, max_new_tokens, ablator, eos)


def _generate_single(setup, ids, max_new_tokens, gen, eos):
    cache = DynamicCache()
    out = setup.hf(ids, past_key_values=cache, use_cache=True)
    new_ids: list[int] = []
    next_id = _sample(out.logits[0, -1], gen)
    while True:
        new_ids.append(next_id)
        if next_id in eos or len(new_ids) >= max_new_tokens:
            break
        step = torch.tensor([[next_id]], device=setup.device)
        out = setup.hf(step, past_key_values=cache, use_cache=True)
        next_id = _sample(out.logits[0, -1], gen)
    return _result(setup, new_ids, max_new_tokens, None, eos)


def _result(setup, new_ids, cap, ablator, eos):
    stopped = bool(new_ids) and new_ids[-1] in eos
    norm_log = ablator.pop_norm_summary() if ablator is not None else {}
    return GenResult(
        text=setup.tok.decode(new_ids, skip_special_tokens=True),
        token_ids=new_ids,
        n_new=len(new_ids),
        hit_cap=len(new_ids) >= cap and not stopped,
        norm_log=norm_log,
    )
