#!/usr/bin/env python3
"""Create the project-level development / held-out split.

The split is by *project*, not by patch, so a method the analysis was tuned on
cannot reappear under another patch id in the held-out set. It is created before
any tuning and is deterministic, so the held-out set stays untouched until the
analysis is frozen.

    python scripts/make_split.py
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_SHARE = 0.70
SALT = "rota-split-v1"


def bucket(project: str) -> float:
    """Stable per-project value in [0, 1); independent of corpus ordering."""
    h = hashlib.sha256((SALT + project).encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def main():
    corpus = json.load(open(os.path.join(ROOT, "data", "rota_corpus.json"), encoding="utf-8"))
    by_project = defaultdict(list)
    for r in corpus:
        by_project[r["project"]].append(r)

    assign = {p: ("dev" if bucket(p) < DEV_SHARE else "heldout") for p in by_project}
    for r in corpus:
        r["split"] = assign[r["project"]]

    counts = Counter(r["split"] for r in corpus)
    print(f"Projects: {len(by_project)}   patches: {len(corpus)}")
    for name in ("dev", "heldout"):
        rows = [r for r in corpus if r["split"] == name]
        projs = {r["project"] for r in rows}
        gt = Counter(r["gt"] for r in rows)
        tools = Counter(r["tool"] for r in rows)
        print(f"\n[{name}] {len(rows)} patches ({100*len(rows)/len(corpus):.1f}%), "
              f"{len(projs)} projects")
        print(f"   labels: {gt['incorrect']} incorrect / {gt['correct']} correct "
              f"({100*gt['incorrect']/max(len(rows),1):.1f}% incorrect)")
        print(f"   generators: {dict(tools)}")

    assert not ({r["project"] for r in corpus if r["split"] == "dev"} &
                {r["project"] for r in corpus if r["split"] == "heldout"}), \
        "project leaked across the split"

    json.dump(corpus, open(os.path.join(ROOT, "data", "rota_corpus.json"), "w", encoding="utf-8"), indent=1)
    json.dump(
        {"salt": SALT, "dev_share": DEV_SHARE, "assignment": assign},
        open(os.path.join(ROOT, "data", "rota_split.json"), "w", encoding="utf-8"), indent=1,
    )
    print(f"\nSplit written into data/rota_corpus.json and data/rota_split.json")
    print("Held-out patches must not be inspected until the analysis is frozen.")


if __name__ == "__main__":
    main()
