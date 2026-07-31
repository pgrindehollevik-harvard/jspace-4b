"""J-space ablation as forward hooks on decoder blocks.

Protocol (paper section 3.5.2, resolutions pre-registered in docs/PREREGISTRATION.md section 5):
at each (token position, band layer), rank J-lens readout logits computed as
lm_head(final_rmsnorm(J_l @ h)), take the top K_ABLATE token ids, drop any id in the clean
model's top-10 next-token predictions at that position, form directions
v_t = J_l^T (gamma * W_U[t]), and jointly project the residual stream out of their span.

mode="random" is the per-slot norm-matched control: it computes the displacement the J-space
ablation WOULD apply at each slot, then removes an equal-norm component along random
directions at that same slot instead (paper A.23 protocol).

The caller must set `exempt_ids` (from a clean forward pass over the same token stream)
BEFORE each hooked forward. Hooks fire only when `enabled` is True, so the same installed
ablator serves both the clean and ablated passes of the dual-stream generation loop.

Performance constraint: this runs per (position, layer) at every decode step on MPS, where
each host-device sync costs more than the math. The hot path therefore stays on-device
end-to-end — projection via a 10x10 normal-equations solve instead of QR (which falls back
to CPU LAPACK on MPS), no .item()/isin, and norm bookkeeping accumulated in device tensors
that are synced once per generation via pop_norm_summary().
"""

import torch

from hw0.core import Setup, K_ABLATE, K_CLEAN_EXEMPT

_RAND_BANK_SIZE = 4096


class JSpaceAblator:
    def __init__(self, setup: Setup, band: tuple[int, int], k: int = K_ABLATE,
                 mode: str = "jspace", rng_seed: int = 0):
        assert mode in ("jspace", "random")
        self.setup = setup
        self.band = list(range(band[0], band[1] + 1))
        self.k = k
        self.mode = mode
        self.enabled = False
        self.exempt_ids: torch.Tensor | None = None  # [seq, K_CLEAN_EXEMPT], on device
        self._handles = []
        self._rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        dtype = setup.hf.dtype
        # Band Jacobians on-device once; bf16 keeps an 11-layer band under ~150 MB.
        self._J = {l: setup.lens.jacobians[l].to(setup.device, dtype) for l in self.band}
        self._gamma = setup.final_norm.weight.detach()
        if mode == "random":
            self._rand_bank = torch.randn(
                _RAND_BANK_SIZE, setup.model.d_model, generator=self._rng
            ).to(setup.device, torch.float32)
        # On-device norm accumulators, synced once per generation.
        self._norm_sum = {l: torch.zeros((), device=setup.device) for l in self.band}
        self._norm_cnt = {l: 0 for l in self.band}

    # -- hook management -------------------------------------------------------------

    def install(self):
        for l in self.band:
            self._handles.append(
                self.setup.blocks[l].register_forward_hook(self._make_hook(l))
            )
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

    def pop_norm_summary(self) -> dict[int, float]:
        """Mean ||delta_h||/||h|| per layer since the last call. Syncs once."""
        out = {l: (self._norm_sum[l] / max(self._norm_cnt[l], 1)).item()
               for l in self.band}
        for l in self.band:
            self._norm_sum[l] = torch.zeros((), device=self.setup.device)
            self._norm_cnt[l] = 0
        return out

    # -- the intervention ------------------------------------------------------------

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            if not self.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output  # [B, S, d]
            h_new = self._ablate(h, layer)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new
        return hook

    def _ablate(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        assert self.exempt_ids is not None, "set exempt_ids before a hooked forward"
        B, S, d = h.shape
        assert B == 1, "batch size 1 only (paired clean stream tracks one sequence)"
        J = self._J[layer]
        # Readout logits at every position: lm_head(norm(J @ h)). Never materialize
        # W_U @ J (vocab x d_model per layer); the transported vector is only [S, d].
        transported = h[0] @ J.T                                          # [S, d]
        logits = self.setup.lm_head(self.setup.final_norm(transported))  # [S, vocab]
        top_ids = logits.topk(self.k, dim=-1).indices                    # [S, k]
        exempt = self.exempt_ids.to(h.device)                            # [S, 10]
        # Exemption mask without isin (MPS-native broadcast compare).
        exempt_mask = (top_ids.unsqueeze(2) == exempt.unsqueeze(1)).any(2)  # [S, k]

        # Directions for ALL top-k, exempt ones zeroed out of the projection basis:
        # a zero row contributes nothing to span/solve (with regularization), so the
        # whole batch of positions runs as one padded einsum + batched 10x10 solve.
        w = self.setup.lm_head.weight[top_ids] * self._gamma             # [S, k, d]
        V = (w.float() @ J.float())                                      # [S, k, d]
        V = V * (~exempt_mask).unsqueeze(-1)
        hs = h[0].float()                                                # [S, d]
        c = torch.einsum("skd,sd->sk", V, hs)                            # [S, k]
        G = torch.einsum("skd,smd->skm", V, V)                           # [S, k, k]
        eye = torch.eye(self.k, device=h.device)
        # Regularization scaled to G's magnitude keeps zeroed (exempt) rows inert and
        # near-collinear directions numerically safe.
        reg = 1e-4 * G.diagonal(dim1=1, dim2=2).amax(1).clamp(min=1e-8)  # [S]
        coeffs = torch.linalg.solve(G + reg[:, None, None] * eye, c.unsqueeze(-1))
        delta = torch.einsum("skd,sk->sd", V, coeffs.squeeze(-1))        # [S, d]

        if self.mode == "random":
            delta = self._random_matched(delta, hs, S, d)

        h_out = h.clone()
        h_out[0] = (hs - delta).to(h.dtype)
        dn = delta.norm(dim=-1) / (hs.norm(dim=-1) + 1e-8)               # [S]
        self._norm_sum[layer] = self._norm_sum[layer] + dn.sum()
        self._norm_cnt[layer] += S
        return h_out

    def _random_matched(self, j_delta: torch.Tensor, hs: torch.Tensor,
                        S: int, d: int) -> torch.Tensor:
        """Equal-norm removal along k random directions at each slot (A.23).

        Directions come from a fixed pre-generated bank; indices are drawn on CPU
        (seeded, no device sync) per slot.
        """
        idx = torch.randint(0, _RAND_BANK_SIZE, (S, self.k), generator=self._rng)
        R = self._rand_bank[idx.to(hs.device)]                           # [S, k, d]
        c = torch.einsum("skd,sd->sk", R, hs)
        G = torch.einsum("skd,smd->skm", R, R)
        eye = torch.eye(self.k, device=hs.device)
        reg = 1e-4 * G.diagonal(dim1=1, dim2=2).amax(1).clamp(min=1e-8)
        coeffs = torch.linalg.solve(G + reg[:, None, None] * eye, c.unsqueeze(-1))
        r = torch.einsum("skd,sk->sd", R, coeffs.squeeze(-1))            # [S, d]
        target = j_delta.norm(dim=-1, keepdim=True)
        return r * target / (r.norm(dim=-1, keepdim=True) + 1e-8)
