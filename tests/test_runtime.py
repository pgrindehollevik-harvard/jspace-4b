"""Fast, model-free checks for portable runtime configuration."""

import pytest

from jspace import core
from jspace.profiles import get_profile


def test_cpu_can_be_requested_explicitly(monkeypatch):
    monkeypatch.delenv("JSPACE_DEVICE", raising=False)
    assert core.resolve_device("cpu") == "cpu"


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="unsupported JSPACE_DEVICE"):
        core.resolve_device("tpu")


def test_stable_and_legacy_lens_names_agree(monkeypatch):
    monkeypatch.setenv("JSPACE_LENS_PATH", "results/lens.pt")
    monkeypatch.setenv("HW0_LENS_PATH", "results/lens.pt")
    assert core.configured_lens_path() == "results/lens.pt"
    assert core.using_local_lens()


def test_conflicting_lens_names_are_rejected(monkeypatch):
    monkeypatch.setenv("JSPACE_LENS_PATH", "results/current.pt")
    monkeypatch.setenv("HW0_LENS_PATH", "results/legacy.pt")
    with pytest.raises(RuntimeError, match="disagree"):
        core.configured_lens_path()


def test_qwen32_profile_has_expected_shape_and_lens():
    profile = get_profile("QWEN3-32B")
    assert profile.n_layers == 64
    assert profile.hidden_size == 5120
    assert profile.lens_file.endswith("Qwen3-32B_jacobian_lens.pt")
