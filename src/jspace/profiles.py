"""Artifact identities used to stage Qwen3 models for the scaling extension.

These profiles deliberately contain *only* immutable artifact locations and model
shapes.  They do not choose an ablation band or analysis rule for a new model: those
belong in the extension's pre-registration, not in deployment plumbing.
"""

from dataclasses import dataclass


LENS_REPO = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class Qwen3Profile:
    key: str
    model_id: str
    lens_file: str
    n_layers: int
    hidden_size: int


QWEN3_PROFILES: dict[str, Qwen3Profile] = {
    "qwen3-1.7b": Qwen3Profile(
        "qwen3-1.7b", "Qwen/Qwen3-1.7B",
        "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        28, 2048,
    ),
    "qwen3-4b": Qwen3Profile(
        "qwen3-4b", "Qwen/Qwen3-4B",
        "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
        36, 2560,
    ),
    "qwen3-8b": Qwen3Profile(
        "qwen3-8b", "Qwen/Qwen3-8B",
        "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        36, 4096,
    ),
    "qwen3-14b": Qwen3Profile(
        "qwen3-14b", "Qwen/Qwen3-14B",
        "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
        40, 5120,
    ),
    "qwen3-32b": Qwen3Profile(
        "qwen3-32b", "Qwen/Qwen3-32B",
        "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt",
        64, 5120,
    ),
}


def get_profile(key: str) -> Qwen3Profile:
    try:
        return QWEN3_PROFILES[key.lower()]
    except KeyError as exc:
        choices = ", ".join(QWEN3_PROFILES)
        raise ValueError(f"unknown model profile {key!r}; choose one of: {choices}") from exc
