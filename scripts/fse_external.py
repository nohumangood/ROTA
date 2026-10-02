"""External APCA baselines and class-balance-robust metrics.

External baselines (NOT ROTA ablations; a separate cfg_lifecycle substrate):
  - INVALIDATOR (Le et al.): semantic overfit + syntactic suspiciousness.
  - ODS (Ye et al.): descriptor-feature rule score.
  - Majority: always predict the development-set majority class.

We report P/R/F1/Acc/MCC/balanced-accuracy on the held-out set and the exact
McNemar test against ROTA on the patches ROTA decides. We also recompute the
leave-one-generator-out table with balanced accuracy and F1 so the extreme
per-tool class imbalance cannot flatter any system.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baselines.invalidator_jleaks import evaluate_invalidator  # noqa: E402
from baselines.ods import evaluate_ods  # noqa: E402
from rota.checker import run_rota  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "data", "rota_corpus.json")


def rota_verdict(p):
    o = run_rota(p["buggy_method"], p["patched_method"],
                 key_var=p.get("key_var", ""), resource_type=p.get("resource_type", ""))
    return {"INCORRECT": "incorrect", "CORRECT": "correct"}.get(o.certificate.verdict,
                                                               "abstain")


def inv_verdict(p):
    return "incorrect" if evaluate_invalidator(
        p["buggy_method"], p["patched_method"], key_var=p.get("key_var", "")
    )["verdict"] == "INCORRECT" else "correct"


def ods_verdict(p):
    return "incorrect" if evaluate_ods(
        p["buggy_method"], p["patched_method"])["verdict"] == "INCORRECT" else "correct"


def metrics(rows):
    """rows: list of (pred, gt) with pred in correct/incorrect (decided only)."""
    tp = sum(1 for pr, gt in rows if pr == "incorrect" and gt == "incorrect")
    fp = sum(1 for pr, gt in rows if pr == "incorrect" and gt == "correct")
    tn = sum(1 for pr, gt in rows if pr == "correct" and gt == "correct")
    fn = sum(1 for pr, gt in rows if pr == "correct" and gt == "incorrect")
    n = tp + fp + tn + fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bacc = (r + tnr) / 2
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den else 0.0
    return {"n": n, "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            "Acc": round((tp + tn) / n, 3) if n else 0.0,
            "BalAcc": round(bacc, 3), "MCC": round(mcc, 3)}


def mcnemar(a, b):
    """a,b: dict patch_id->pred(correct/incorrect); gt via GT. exact two-sided."""
    pass


def exact_mcnemar(pairs):
    """pairs: list of (a_correct_bool, b_correct_bool). Return exact 2-sided p."""
    b = sum(1 for a, bb in pairs if a and not bb)
    c = sum(1 for a, bb in pairs if bb and not a)
    k, tot = min(b, c), b + c
    if tot == 0:
        return 1.0, b, c
    p = min(1.0, 2 * sum(comb(tot, i) for i in range(k + 1)) / 2 ** tot)
    return p, b, c


def main():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    held = [p for p in corpus if p.get("split") == "heldout"]
    dev = [p for p in corpus if p.get("split") == "dev"]
    maj = "incorrect" if sum(1 for p in dev if p["gt"] == "incorrect") > len(dev) / 2 \
        else "correct"

    # per-patch predictions
    rv = {p["patch_id"]: rota_verdict(p) for p in held}
    iv = {p["patch_id"]: inv_verdict(p) for p in held}
    ov = {p["patch_id"]: ods_verdict(p) for p in held}
    mv = {p["patch_id"]: maj for p in held}
    gt = {p["patch_id"]: p["gt"] for p in held}

    out = {"majority_class": maj, "heldout": {}}
    for name, pred in (("INVALIDATOR", iv), ("ODS", ov), ("Majority", mv)):
        rows = [(pred[pid], gt[pid]) for pid in gt]
        m = metrics(rows)
        # McNemar vs ROTA on patches ROTA decides
        dec = [pid for pid in gt if rv[pid] != "abstain"]
        pairs = [(pred[pid] == gt[pid], rv[pid] == gt[pid]) for pid in dec]
        p, b, c = exact_mcnemar(pairs)
        m["McNemar_vs_ROTA"] = round(p, 4)
        m["b_base>rota"] = b
        m["c_rota>base"] = c
        out["heldout"][name] = m
    # ROTA for reference (decided set)
    rrows = [(rv[pid], gt[pid]) for pid in gt if rv[pid] != "abstain"]
    out["heldout"]["ROTA(decided)"] = metrics(rrows)

    # LOGO with balanced accuracy / F1
    out["logo_balanced"] = {}
    for tool in ("rlfixer", "arodnap", "amur"):
        sub = [p for p in corpus if p["tool"] == tool]
        gtt = {p["patch_id"]: p["gt"] for p in sub}
        rvt = {p["patch_id"]: rota_verdict(p) for p in sub}
        ivt = {p["patch_id"]: inv_verdict(p) for p in sub}
        ovt = {p["patch_id"]: ods_verdict(p) for p in sub}
        majt = {p["patch_id"]: maj for p in sub}
        row = {"n": len(sub),
               "pct_incorrect": round(sum(1 for p in sub if p["gt"] == "incorrect")
                                      / len(sub), 3)}
        for name, pred in (("ROTA", rvt), ("INVALIDATOR", ivt), ("ODS", ovt),
                           ("Majority", majt)):
            rows = [(pred[pid], gtt[pid]) for pid in gtt if pred[pid] != "abstain"]
            m = metrics(rows)
            row[name] = {"Acc": m["Acc"], "BalAcc": m["BalAcc"], "F1": m["F1"]}
        out["logo_balanced"][tool] = row

    json.dump(out, open(os.path.join(ROOT, "results", "fse_external.json"), "w",
                        encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
