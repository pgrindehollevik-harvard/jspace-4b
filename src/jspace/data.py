"""Dataset loading and sampling, exactly as pre-registered (section 4).

Every item: {"id", "dataset", "problem", "answer", "level"} with answer as a string.
Sampling is seeded and deterministic; GSM8K pilot problems come from the TRAIN split.
"""

import itertools
import random

from datasets import load_dataset

CAPS = {  # max_new_tokens per dataset x mode
    ("gsm8k", "cot"): 640, ("math500", "cot"): 768, ("aime", "cot"): 1536,
    ("gsm8k", "direct"): 64, ("math500", "direct"): 64, ("aime", "direct"): 64,
}


def gsm8k_answer(sol: str) -> str:
    return sol.split("####")[-1].strip().replace(",", "")


def load_gsm8k(n=150, seed=0, split="test", revision=None):
    ds = load_dataset("openai/gsm8k", "main", split=split, revision=revision)
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [{"id": f"gsm8k-{split}-{i}", "dataset": "gsm8k", "level": 0,
             "problem": ds[i]["question"], "answer": gsm8k_answer(ds[i]["answer"])}
            for i in idx]


def load_math500(per_level=30, seed=0):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_level = {}
    for i, row in enumerate(ds):
        by_level.setdefault(row["level"], []).append(i)
    groups = []
    for level in sorted(by_level):
        take = random.Random(seed + level).sample(by_level[level],
                                                  min(per_level, len(by_level[level])))
        groups.append([{"id": f"math500-{ds[i]['unique_id']}", "dataset": "math500",
                        "level": level, "problem": ds[i]["problem"],
                        "answer": ds[i]["answer"]} for i in take])
    # Round-robin across levels so every prefix (the n=100 random arm, trim-ladder
    # n=125/75, and any interrupted run) is level-balanced; see DEVIATIONS.md entry 2.
    return [x for tier in itertools.zip_longest(*groups) for x in tier if x is not None]


def load_aime():
    a24 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    a25 = load_dataset("yentinglin/aime_2025", "default", split="train")
    items = [{"id": f"aime-{r['ID']}", "dataset": "aime", "level": 6,
              "problem": r["Problem"], "answer": str(r["Answer"]).strip()} for r in a24]
    items += [{"id": f"aime-{r['id']}", "dataset": "aime", "level": 6,
               "problem": r["problem"], "answer": str(r["answer"]).strip()} for r in a25]
    return items


def load_wikitext_heldout(n=50, min_chars=600, skip=2000, max_chars=2000,
                          revision=None):
    """Selectivity-control sequences, drawn far past the lens's fitting region."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True, revision=revision)
    out, seen = [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append(text[:max_chars])
        if len(out) == n:
            return out
    raise RuntimeError("not enough wikitext rows")
