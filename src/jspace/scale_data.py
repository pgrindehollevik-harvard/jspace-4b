"""Materialize and verify the exact public-data subset for offline scale jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jspace.data import load_gsm8k, load_wikitext_heldout
from jspace.scale_config import REPO_ROOT, config_sha256, load_scale_config


DEFAULT_CACHE = REPO_ROOT / ".scale-cache" / "v2_data.json"


def cache_path() -> Path:
    return Path(os.environ.get("JSPACE_SCALE_DATA", DEFAULT_CACHE))


def payload_sha256(payload: dict) -> str:
    scientific = {
        "wikitext": payload["wikitext"],
        "coherence": payload["coherence"],
    }
    encoded = json.dumps(
        scientific, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)


def stage(path: Path | None = None) -> dict:
    path = path or cache_path()
    config = load_scale_config()
    wiki = config["data"]["wikitext"]
    coherence = config["data"]["coherence"]

    texts = load_wikitext_heldout(
        n=wiki["n"],
        min_chars=wiki["min_chars"],
        skip=wiki["skip_eligible"],
        max_chars=wiki["max_chars"],
        revision=wiki["revision"],
    )
    problems = load_gsm8k(
        n=coherence["n"],
        seed=coherence["sample_seed"],
        split=coherence["split"],
        revision=coherence["revision"],
    )
    payload = {
        "schema_version": 1,
        "config_sha256": config_sha256(),
        "wikitext": texts,
        "coherence": problems,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    _atomic_dump(path, payload)
    return payload


def load_staged_data(path: Path | None = None) -> dict:
    path = path or cache_path()
    if not path.exists():
        raise FileNotFoundError(
            f"missing staged scale data at {path}; run python -m jspace.scale_data"
        )
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("config_sha256") != config_sha256():
        raise RuntimeError("staged data was selected under a different v2 config")
    if payload.get("payload_sha256") != payload_sha256(payload):
        raise RuntimeError("staged scale-data digest mismatch")
    config = load_scale_config()
    if len(payload["wikitext"]) != config["data"]["wikitext"]["n"]:
        raise RuntimeError("wrong number of staged Wikitext sequences")
    if len(payload["coherence"]) != config["data"]["coherence"]["n"]:
        raise RuntimeError("wrong number of staged coherence problems")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cache_path())
    args = parser.parse_args()
    payload = stage(args.output)
    print(json.dumps({
        "path": str(args.output),
        "config_sha256": payload["config_sha256"],
        "payload_sha256": payload["payload_sha256"],
        "wikitext_n": len(payload["wikitext"]),
        "coherence_ids": [item["id"] for item in payload["coherence"]],
    }, indent=2))


if __name__ == "__main__":
    main()
