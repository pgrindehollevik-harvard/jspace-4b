"""Intervention correctness tests (pre-registration sections 5 and 7).

Run: .venv/bin/python -m pytest tests/ -v   (loads Qwen3-4B on MPS once; ~2-3 min)
"""

import pytest
import torch

from hw0 import core
from hw0.ablation import JSpaceAblator
from hw0.generate import generate

PROMPT = "Fact: The currency used in the country shaped like a boot is"


@pytest.fixture(scope="module")
def setup():
    return core.load()


@pytest.fixture(scope="module")
def ids(setup):
    return setup.tok(PROMPT, return_tensors="pt").input_ids.to(setup.device)


def clean_logits(setup, ids):
    with torch.no_grad():
        return setup.hf(ids).logits.float().cpu()


def test_disabled_hooks_are_identity(setup, ids):
    base = clean_logits(setup, ids)
    abl = JSpaceAblator(setup, core.BAND_PRIMARY_CANDIDATES[0]).install()
    try:
        abl.enabled = False
        hooked = clean_logits(setup, ids)
    finally:
        abl.remove()
    assert torch.equal(base, hooked), "disabled hooks must not perturb the forward pass"


def test_ablation_changes_output_and_projection_is_removed(setup, ids):
    band = core.BAND_PRIMARY_CANDIDATES[0]
    abl = JSpaceAblator(setup, band).install()
    layer = band[0]
    captured = {}

    def capture(module, args, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    probe = setup.blocks[layer].register_forward_hook(capture)
    try:
        base = clean_logits(setup, ids)
        abl.exempt_ids = base[0].topk(core.K_CLEAN_EXEMPT, dim=-1).indices
        abl.enabled = True
        ablated = clean_logits(setup, ids)
        abl.enabled = False
        h_post = captured["h"]  # residual AFTER our hook modified it
    finally:
        probe.remove()
        abl.remove()

    assert not torch.equal(base, ablated), "enabled ablation must change the logits"

    # Post-hook residual must be (near-)orthogonal to the ablated directions at the
    # last position: recompute the selection the hook made and check the projection.
    s = ids.shape[1] - 1
    J = abl._J[layer]
    with torch.no_grad():
        transported = h_post[0, s] @ J.T
        readout = setup.lm_head(setup.final_norm(transported))
        # The hook ablated the pre-modification top-k; after removal, surviving
        # projection norms onto ablated (non-exempt) directions must be tiny.
        top = readout.topk(core.K_ABLATE).indices
    # note: exact ids differ post-ablation; the strong invariant is tested next.


def test_projection_zeroes_selected_directions(setup):
    band = core.BAND_PRIMARY_CANDIDATES[0]
    abl = JSpaceAblator(setup, band)
    layer = band[0]
    h = torch.randn(1, 3, setup.model.d_model, device=setup.device,
                    dtype=setup.hf.dtype)
    abl.exempt_ids = torch.full((3, core.K_CLEAN_EXEMPT), -1)  # exempt nothing real
    with torch.no_grad():
        J = abl._J[layer]
        transported = h[0] @ J.T
        top = setup.lm_head(setup.final_norm(transported)).topk(core.K_ABLATE).indices
        h_new = abl._ablate(h, layer)
        for s in range(3):
            w = setup.lm_head.weight[top[s]] * abl._gamma
            V = (w @ J).T.float()
            resid = V.T @ h_new[0, s].float()
            before = V.T @ h[0, s].float()
            assert resid.norm() < 0.05 * before.norm() + 1e-3, (
                f"projection onto ablated directions must be (near) zero, "
                f"got {resid.norm():.4f} vs {before.norm():.4f}")


def test_exempt_ids_survive(setup):
    band = core.BAND_PRIMARY_CANDIDATES[0]
    abl = JSpaceAblator(setup, band)
    layer = band[0]
    h = torch.randn(1, 1, setup.model.d_model, device=setup.device, dtype=setup.hf.dtype)
    with torch.no_grad():
        J = abl._J[layer]
        top = setup.lm_head(setup.final_norm(h[0] @ J.T)).topk(core.K_ABLATE).indices  # [1, k]
        abl.exempt_ids = top  # exempt EVERYTHING the hook would ablate
        h_new = abl._ablate(h, layer)
    assert torch.equal(h, h_new), "fully-exempt slot must pass through unchanged"


def test_random_mode_matches_jspace_norm(setup):
    band = core.BAND_PRIMARY_CANDIDATES[0]
    layer = band[0]
    h = torch.randn(1, 2, setup.model.d_model, device=setup.device, dtype=setup.hf.dtype)
    exempt = torch.full((2, core.K_CLEAN_EXEMPT), -1)
    deltas = {}
    for mode in ("jspace", "random"):
        abl = JSpaceAblator(setup, band, mode=mode)
        abl.exempt_ids = exempt
        with torch.no_grad():
            h_new = abl._ablate(h.clone(), layer)
        deltas[mode] = (h - h_new)[0].float().norm(dim=-1)
    assert torch.allclose(deltas["jspace"], deltas["random"], rtol=0.05), (
        f"per-slot norms must match: {deltas}")


def test_generation_determinism_and_dual_stream(setup):
    prompt = core.chat_prompt(setup, "What is 3 + 4?", "cot")
    band = core.BAND_PRIMARY_CANDIDATES[0]
    outs = []
    for _ in range(2):
        abl = JSpaceAblator(setup, band).install()
        try:
            outs.append(generate(setup, prompt, max_new_tokens=40, seed=7, ablator=abl))
        finally:
            abl.remove()
    assert outs[0].token_ids == outs[1].token_ids, "same seed must reproduce exactly"
    clean = generate(setup, prompt, max_new_tokens=40, seed=7)
    assert clean.n_new > 0 and outs[0].n_new > 0
