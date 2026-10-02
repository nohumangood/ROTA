#!/usr/bin/env python3
"""Validate the rebuilt corpus before any analysis is built on it.

Two failure modes would silently invalidate every downstream result:

1. *Selection bias* - if recovery drops incorrect patches more often than
   correct ones, the rebuilt label balance is not comparable to the original.
2. *Edit loss* - scoped replay drops hunks that fall outside the target method.
   If a dropped hunk was the actual repair, the recovered M' equals M, which
   turns a real patch into a no-op and corrupts the ground truth.

    python scripts/validate_corpus.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RELEASE = re.compile(r"\.\s*(close|disconnect|shutdown|dispose|release|free)\s*\(", re.I)


def norm(s: str) -> str:
    return " ".join((s or "").split())


def main():
    corpus = json.load(open(os.path.join(ROOT, "data", "rota_corpus.json"), encoding="utf-8"))
    uncovered = json.load(open(os.path.join(ROOT, "results", "corpus_uncovered.json"), encoding="utf-8"))
    original = [
        p for p in json.load(open(os.path.join(ROOT, "data", "eval_patches.json"), encoding="utf-8"))
        if p.get("gt") in ("correct", "incorrect") and p.get("patched_code")
    ]

    print("=" * 68)
    print("1. SELECTION BIAS")
    print("=" * 68)
    orig = Counter(p["gt"] for p in original)
    rec = Counter(r["gt"] for r in corpus)
    unc = Counter(u["gt"] for u in uncovered)
    print(f"{'':12s} {'incorrect':>10s} {'correct':>10s} {'incorrect%':>12s}")
    for name, c in (("original", orig), ("recovered", rec), ("dropped", unc)):
        n = c["incorrect"] + c["correct"]
        print(f"{name:12s} {c['incorrect']:10d} {c['correct']:10d} "
              f"{100*c['incorrect']/max(n,1):11.1f}%")
    print()
    print("Recovery rate by ground-truth label:")
    for label in ("incorrect", "correct"):
        print(f"   {label:10s} {rec[label]:5d}/{orig[label]:5d}  "
              f"({100*rec[label]/orig[label]:5.1f}%)")
    print()
    print("Recovery rate by (tool, label):")
    orig_tl = Counter((p["tool"], p["gt"]) for p in original)
    rec_tl = Counter((r["tool"], r["gt"]) for r in corpus)
    for tool in ("rlfixer", "arodnap", "amur"):
        for label in ("incorrect", "correct"):
            o, r = orig_tl[(tool, label)], rec_tl[(tool, label)]
            print(f"   {tool:9s} {label:10s} {r:4d}/{o:4d}  ({100*r/max(o,1):5.1f}%)")
    print()
    print("Drop reason by label:")
    for k, v in Counter((u["reason"], u["gt"]) for u in uncovered).most_common():
        print(f"   {k}: {v}")

    print()
    print("=" * 68)
    print("2. EDIT LOSS")
    print("=" * 68)
    identical = [r for r in corpus if norm(r["buggy_method"]) == norm(r["patched_method"])]
    print(f"Recovered pairs where M' is textually identical to M: {len(identical)} "
          f"({100*len(identical)/len(corpus):.1f}%)")
    if identical:
        print("   by strategy:", dict(Counter(r["recovery_strategy"] for r in identical)))
        print("   by gt:      ", dict(Counter(r["gt"] for r in identical)))
    print()

    dropped_hunks = [r for r in corpus if r.get("hunks_dropped", 0) > 0]
    print(f"Pairs where replay dropped at least one hunk: {len(dropped_hunks)} "
          f"({100*len(dropped_hunks)/len(corpus):.1f}%)")
    print("   dropped-hunk histogram:",
          dict(Counter(min(r["hunks_dropped"], 5) for r in dropped_hunks)))
    print()

    # A resource-leak repair should change release structure. If neither the
    # release-call count nor the try/finally structure moved, the recovered M'
    # probably lost the repair.
    def shape(code):
        return (
            len(RELEASE.findall(code or "")),
            (code or "").count("finally"),
            (code or "").count("try"),
        )

    no_shape_change = [r for r in corpus if shape(r["buggy_method"]) == shape(r["patched_method"])]
    print(f"Pairs with no change in (release count, finally, try): {len(no_shape_change)} "
          f"({100*len(no_shape_change)/len(corpus):.1f}%)")
    print("   by strategy:", dict(Counter(r["recovery_strategy"] for r in no_shape_change)))
    print("   by gt:      ", dict(Counter(r["gt"] for r in no_shape_change)))
    print()

    suspicious = [r for r in no_shape_change if r["gt"] == "correct"]
    print(f"Suspicious (labelled correct but patch changed no release structure): "
          f"{len(suspicious)}")
    print("   -> a correct resource-leak repair that adds no release is implausible,")
    print("      so these indicate lost edits rather than genuine no-op repairs.")

    json.dump(
        [{"patch_id": r["patch_id"], "gt": r["gt"], "strategy": r["recovery_strategy"]}
         for r in no_shape_change],
        open(os.path.join(ROOT, "results", "corpus_suspect_edits.json"), "w", encoding="utf-8"),
        indent=1,
    )
    print("\nWrote results/corpus_suspect_edits.json")


if __name__ == "__main__":
    main()
