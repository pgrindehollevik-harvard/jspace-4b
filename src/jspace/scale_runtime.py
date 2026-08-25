"""Pinned model/lens loader used only by the version-2 extension."""

from __future__ import annotations

import os

import torch
import transformers

import jlens

from jspace import core
from jspace.profiles import LENS_REPO, LENS_REVISION, get_profile
from jspace.scale_config import confirmatory_profiles, transferred_band


def load_scale_setup(profile_key: str, device: str | None = None,
                     dtype: torch.dtype | None = None) -> core.Setup:
    if profile_key not in confirmatory_profiles():
        raise ValueError(
            f"{profile_key!r} is outside the frozen confirmatory profiles "
            f"{confirmatory_profiles()}"
        )
    if core.configured_lens_path():
        raise RuntimeError("the v2 scale test requires the pinned stock lens")

    profile = get_profile(profile_key)
    device = core.resolve_device(device)
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        dtype=dtype,
        local_files_only=offline,
    ).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        local_files_only=offline,
    )
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=profile.lens_file,
        revision=LENS_REVISION,
    )

    if len(model.layers) != profile.n_layers:
        raise RuntimeError(
            f"profile says {profile.n_layers} layers, loaded {len(model.layers)}"
        )
    if model.d_model != profile.hidden_size:
        raise RuntimeError(
            f"profile says d_model={profile.hidden_size}, loaded {model.d_model}"
        )
    band = transferred_band(profile.n_layers)
    missing = [layer for layer in range(band[0], band[1] + 1)
               if layer not in lens.jacobians]
    if missing:
        raise RuntimeError(f"pinned lens is missing registered band layers {missing}")
    return core.Setup(hf=hf, tok=tok, model=model, lens=lens, device=device)
