"""Minimal CUDA readiness test for ORCD.

The test loads the already-staged 4B model and lens, performs one short clean
generation and one short ablated generation, and prints hardware, timing, and peak
memory.  It is an engineering smoke test, not an experimental run: it does not inspect
accuracy, update calibration artifacts, or make a scientific decision.
"""

import json
import time

import torch

from jspace import core
from jspace.ablation import JSpaceAblator
from jspace.generate import generate


def _timed_generation(*args, **kwargs):
    started = time.perf_counter()
    result = generate(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return result, elapsed


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ORCD smoke test requires an allocated CUDA GPU")

    torch.cuda.reset_peak_memory_stats()
    setup = core.load(device="cuda")
    prompt = core.chat_prompt(setup, "What is 6 times 7?", "cot")

    clean, clean_seconds = _timed_generation(
        setup, prompt, max_new_tokens=24, seed=17
    )
    ablator = JSpaceAblator(setup, core.BAND_PRIMARY_CANDIDATES[0]).install()
    try:
        ablated, ablated_seconds = _timed_generation(
            setup, prompt, max_new_tokens=24, seed=17, ablator=ablator
        )
    finally:
        ablator.remove()

    print(json.dumps({
        "device": setup.device,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "lens_layers": [min(setup.lens.jacobians), max(setup.lens.jacobians)],
        "clean_tokens": clean.n_new,
        "clean_seconds": round(clean_seconds, 3),
        "ablated_tokens": ablated.n_new,
        "ablated_seconds": round(ablated_seconds, 3),
        "peak_cuda_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "clean_preview": clean.text[:120],
        "ablated_preview": ablated.text[:120],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
