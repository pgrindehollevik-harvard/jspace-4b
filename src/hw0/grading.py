"""Grading and artifact accounting (pre-registration sections 6.7 and 8).

Every output is classified into exactly one artifact class:
  correct / wrong-answer / no-parseable-answer / truncated-at-cap / degenerate
No-answer scores as incorrect in the primary analysis; classes are kept separate so
format breakage cannot masquerade as an accuracy gradient.
"""

import re

from math_verify import parse, verify

BOXED = re.compile(r"\\boxed\s*\{")
LAST_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_boxed(text: str) -> str | None:
    m = None
    for m_ in BOXED.finditer(text):
        m = m_
    if m is None:
        return None
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1] if depth == 0 else None


def max_ngram_repetition(token_ids: list[int], n: int = 4) -> float:
    if len(token_ids) < n * 2:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) * n / len(token_ids)


def grade(dataset: str, gold: str, text: str, token_ids: list[int],
          hit_cap: bool) -> dict:
    degenerate = max_ngram_repetition(token_ids) > 0.5
    pred = extract_boxed(text)
    if pred is None and dataset == "gsm8k":  # pre-registered numeric fallback
        nums = LAST_NUMBER.findall(text)
        pred = nums[-1].replace(",", "").replace("$", "") if nums else None

    if pred is None:
        cls = "truncated-at-cap" if hit_cap else (
            "degenerate" if degenerate else "no-parseable-answer")
        return {"correct": False, "pred": None, "class": cls}

    if dataset == "aime":
        try:
            ok = int(float(pred.replace(",", ""))) == int(gold)
        except ValueError:
            ok = False
    else:
        try:
            ok = bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
        except Exception:
            ok = False

    cls = "correct" if ok else "wrong-answer"
    if degenerate and not ok:
        cls = "degenerate"
    return {"correct": ok, "pred": pred, "class": cls,
            "hit_cap": hit_cap, "degenerate": degenerate}
