"""Run every ROTA configuration over one split and write the paper's tables.

Produces a master CSV with one row per patch per configuration, a metrics
summary with bootstrap confidence intervals, McNemar tests of ROTA against each
comparison, and a per-patch error file for the qualitative analysis.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rota.baselines import APCA_BASELINES, LADDER  # noqa: E402
from rota.checker import run_rota  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "data", "rota_corpus.json")

#: Each entry disables exactly one relational component.
ABLATIONS = {
    "rota_full": {},
    "no_lineage": {"use_lineage": False},
    "no_effect_frontier": {"use_frontier": False},
    "no_conservation": {"use_conservation": False},
    "no_propagation": {"use_propagation": False},
    "no_ownership": {"use_ownership": False},
    "no_path_correspondence": {"use_path_correspondence": False},
    "no_target_condition": {"require_target": False},
}

#: Progressively fewer of the analysis inputs, to show what kappa and tau buy.
INPUT_ABLATIONS = {
    "inputs_M_Mp": {"key_var": False, "resource_type": False},
    "inputs_kappa": {"key_var": True, "resource_type": False},
    "inputs_tau": {"key_var": False, "resource_type": True},
    "inputs_kappa_tau": {"key_var": True, "resource_type": True},
}


def metrics(rows):
    # Abstentions are excluded from precision/recall and reported as coverage:
    # a certifying analysis is scored on the patches it actually decides.
    decided = [r for r in rows if r["pred"] in ("incorrect", "correct")]
    tp = sum(1 for r in decided if r["pred"] == "incorrect" and r["gt"] == "incorrect")
    fp = sum(1 for r in decided if r["pred"] == "incorrect" and r["gt"] == "correct")
    tn = sum(1 for r in decided if r["pred"] == "correct" and r["gt"] == "correct")
    fn = sum(1 for r in decided if r["pred"] == "correct" and r["gt"] == "incorrect")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    n = len(decided)
    return {"n": len(rows), "decided": n, "coverage": round(n / len(rows), 4) if rows else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(p, 4), "recall": round(rc, 4),
            "f1": round(f1, 4), "accuracy": round((tp + tn) / n, 4) if n else 0.0}


def bootstrap_ci(rows, key="f1", iters=2000, seed=20260101):
    """Percentile bootstrap over patches, resampled with replacement."""
    rng = random.Random(seed)
    n = len(rows)
    vals = []
    for _ in range(iters):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        vals.append(metrics(sample)[key])
    vals.sort()
    return round(vals[int(0.025 * iters)], 4), round(vals[int(0.975 * iters)], 4)


def mcnemar(rows_a, rows_b):
    """Exact McNemar test on the patches where the two systems disagree."""
    by_b = {r["patch_id"]: r for r in rows_b}
    b_only = c_only = 0
    for r in rows_a:
        o = by_b.get(r["patch_id"])
        if o is None:
            continue
        if r["pred"] == "abstain" or o["pred"] == "abstain":
            continue
        ra = r["pred"] == r["gt"]
        rb = o["pred"] == o["gt"]
        if ra and not rb:
            b_only += 1
        elif rb and not ra:
            c_only += 1
    n = b_only + c_only
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    # Two-sided exact binomial test with p = 0.5.
    from math import comb

    k = min(b_only, c_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b": b_only, "c": c_only, "p_value": round(min(1.0, 2 * tail), 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--out", default="results/rota")
    args = ap.parse_args()

    corpus = json.load(open(CORPUS, encoding="utf-8"))
    data = corpus if args.split == "all" else [p for p in corpus if p.get("split") == args.split]
    print(f"{args.split}: {len(data)} patches, "
          f"{len({p['project'] for p in data})} projects")

    runs = collections.OrderedDict()

    for name, flags in ABLATIONS.items():
        runs[name] = [_rota_row(p, flags) for p in data]
    for name, which in INPUT_ABLATIONS.items():
        runs[name] = [_rota_row(p, {}, inputs=which) for p in data]
    for name, fn in {**LADDER, **APCA_BASELINES}.items():
        runs[name] = [_baseline_row(p, fn) for p in data]

    outdir = os.path.join(ROOT, os.path.dirname(args.out))
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(ROOT, args.out) + f"_{args.split}"

    with open(stem + "_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "configuration", "patch_id", "project", "tool", "gt", "pred",
            "reason", "failure_family", "resource_type", "parse_ok",
        ])
        w.writeheader()
        for name, rows in runs.items():
            for r in rows:
                w.writerow({"configuration": name, **r})

    summary = {}
    full = runs["rota_full"]
    for name, rows in runs.items():
        m = metrics(rows)
        lo, hi = bootstrap_ci(rows)
        m["f1_ci95"] = [lo, hi]
        if name != "rota_full":
            m["mcnemar_vs_rota_full"] = mcnemar(full, rows)
        summary[name] = m
    summary["per_project"] = _per_project(full)
    summary["per_family"] = _per_family(full)
    json.dump(summary, open(stem + "_metrics.json", "w", encoding="utf-8"), indent=1)

    with open(stem + "_errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "patch_id", "project", "tool", "gt", "pred", "error_type",
            "reason", "failure_family", "resource_type",
        ])
        w.writeheader()
        for r in full:
            if r["pred"] == r["gt"]:
                continue
            w.writerow({**{k: r[k] for k in (
                "patch_id", "project", "tool", "gt", "pred", "reason",
                "failure_family", "resource_type")},
                "error_type": "FP" if r["pred"] == "incorrect" else "FN"})

    _report(summary, runs)
    print(f"\nWrote {stem}_predictions.csv, _metrics.json, _errors.csv")


def _rota_row(p, flags, inputs=None):
    inputs = inputs or {"key_var": True, "resource_type": True}
    out = run_rota(
        p["buggy_method"], p["patched_method"],
        key_var=p.get("key_var", "") if inputs["key_var"] else "",
        resource_type=p.get("resource_type", "") if inputs["resource_type"] else "",
        **flags,
    )
    cert = out.certificate
    pred = {"INCORRECT": "incorrect", "CORRECT": "correct"}.get(cert.verdict, "abstain")
    return _row(p, pred, cert.reason, cert.parse_ok)


def _baseline_row(p, fn):
    pred = fn(p["buggy_method"], p["patched_method"],
              key_var=p.get("key_var", ""), resource_type=p.get("resource_type", ""))
    return _row(p, "incorrect" if pred.verdict == "INCORRECT" else "correct",
                pred.reason, pred.parse_ok)


def _row(p, pred, reason, parse_ok):
    return {
        "patch_id": p["patch_id"], "project": p.get("project", ""),
        "tool": p.get("tool", ""), "gt": p["gt"], "pred": pred,
        "reason": reason,
        "failure_family": (p.get("failure_reason") or "correct").split(":")[0],
        "resource_type": p.get("resource_type", ""), "parse_ok": parse_ok,
    }


def _per_project(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["project"]].append(r)
    f1s = []
    accs = []
    for proj, rs in by.items():
        m = metrics(rs)
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
    f1s.sort()
    accs.sort()
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else 0.0  # noqa: E731
    return {
        "n_projects": len(by),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "macro_accuracy": round(sum(accs) / len(accs), 4) if accs else 0.0,
        "median_accuracy": q(accs, 0.5),
        "q1_accuracy": q(accs, 0.25), "q3_accuracy": q(accs, 0.75),
    }


def _per_family(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["failure_family"]].append(r)
    return {fam: metrics(rs) for fam, rs in sorted(by.items())}


def _report(summary, runs):
    print()
    print(f"{'configuration':<26}{'cov':>6}{'P':>7}{'R':>7}{'F1':>7}{'Acc':>7}"
          f"{'F1 95% CI':>18}{'McNemar p':>12}")
    for name in runs:
        m = summary[name]
        ci = f"[{m['f1_ci95'][0]:.3f}, {m['f1_ci95'][1]:.3f}]"
        mc = m.get("mcnemar_vs_rota_full", {}).get("p_value", "")
        mc = f"{mc:.2e}" if isinstance(mc, float) else "-"
        print(f"{name:<26}{m['coverage']:>6.2f}{m['precision']:>7.3f}{m['recall']:>7.3f}"
              f"{m['f1']:>7.3f}{m['accuracy']:>7.3f}{ci:>18}{mc:>12}")
    print()
    print("ROTA by failure family:")
    for fam, m in summary["per_family"].items():
        print(f"   {fam:<20} n={m['n']:<5} acc={m['accuracy']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f}")
    print()
    print("project-level:", summary["per_project"])


if __name__ == "__main__":
    main()
