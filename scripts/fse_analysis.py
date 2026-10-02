"""Reviewer-driven strengthening analyses for the FSE submission.

Everything here is computed from the real corpus and the real ROTA code; no
number is hand-entered. Produces results/fse_analysis.json and prints a report.

Analyses
  A. Per-condition attribution (which refinement condition produced each
     INCORRECT verdict) + partial-fix attribution  -> resolves the
     "conservation changes nothing yet is credited for partial fixes" paradox.
  B. Baseline disagreement case analysis (ROTA vs independent_diff / patch_only).
  C. Challenge subsets (wrapper / multi-resource / caller-owned / rename-rebind
     / exceptional) with per-subset metrics for ROTA and baselines.
  D. Leave-one-generator-out + per-generator generalization.
  E. kappa/tau input anomaly (kappa-only vs kappa+tau disagreements).
  F. Automatic lineage accuracy on oracle-determined correspondences.
  G. Abstention: decided / abstain-as-error / common-coverage / selective risk.
  H. Project-cluster bootstrap + hyperparameter sensitivity (theta, radius).
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rota.checker as checker  # noqa: E402
from rota.baselines import (  # noqa: E402
    independent_diff, patch_only_lifecycle, unary_patched_only, warning_removal,
)
from rota.checker import run_rota  # noqa: E402
from rota.ir import Ownership, build_ir  # noqa: E402
from rota.lineage import compute_lineage, target_resources  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "data", "rota_corpus.json")

COND = {
    "PATCH_INDUCED_LIFECYCLE_INVALIDATION": "C1_invalidate",
    "PATCH_INTRODUCED_NEW_OBLIGATION": "C1_introduce",
    "ORPHANED_OBLIGATION": "C1_orphaned",
    "TARGET_OBLIGATION_NOT_DISCHARGED": "C2_target_not_discharged",
    "TARGET_RELEASE_NOT_EXCEPTION_SAFE": "C2_target_weak",
    "COUPLED_OBLIGATION_LEFT_PENDING": "C3_conservation",
    "NO_REPAIRING_TRANSFORMATION": "C4_no_repair",
}


# --------------------------------------------------------------------------- #
# metrics (abstention-aware), replicated so this file is standalone
# --------------------------------------------------------------------------- #
def metrics(rows):
    dec = [r for r in rows if r["pred"] in ("incorrect", "correct")]
    tp = sum(1 for r in dec if r["pred"] == "incorrect" and r["gt"] == "incorrect")
    fp = sum(1 for r in dec if r["pred"] == "incorrect" and r["gt"] == "correct")
    tn = sum(1 for r in dec if r["pred"] == "correct" and r["gt"] == "correct")
    fn = sum(1 for r in dec if r["pred"] == "correct" and r["gt"] == "incorrect")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    n = len(dec)
    return {"n": len(rows), "decided": n,
            "coverage": round(n / len(rows), 4) if rows else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(p, 4), "recall": round(rc, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / n, 4) if n else 0.0}


def acc_abstain_wrong(rows):
    n = len(rows)
    ok = sum(1 for r in rows if r["pred"] == r["gt"])
    return round(ok / n, 4) if n else 0.0


def mcnemar(a, b):
    by = {r["patch_id"]: r for r in b}
    bo = co = 0
    for r in a:
        o = by.get(r["patch_id"])
        if o is None or r["pred"] == "abstain" or o["pred"] == "abstain":
            continue
        ra, rb = r["pred"] == r["gt"], o["pred"] == o["gt"]
        if ra and not rb:
            bo += 1
        elif rb and not ra:
            co += 1
    n = bo + co
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    from math import comb
    k = min(bo, co)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": bo, "c": co, "p_value": round(min(1.0, 2 * tail), 6)}


# --------------------------------------------------------------------------- #
# per-patch computation
# --------------------------------------------------------------------------- #
def rota_pred(p, **flags):
    out = run_rota(p["buggy_method"], p["patched_method"],
                   key_var=p.get("key_var", ""),
                   resource_type=p.get("resource_type", ""), **flags)
    c = out.certificate
    pred = {"INCORRECT": "incorrect", "CORRECT": "correct"}.get(c.verdict, "abstain")
    return pred, c.reason


def rota_pred_inputs(p, use_k, use_t):
    out = run_rota(p["buggy_method"], p["patched_method"],
                   key_var=p.get("key_var", "") if use_k else "",
                   resource_type=p.get("resource_type", "") if use_t else "")
    c = out.certificate
    return {"INCORRECT": "incorrect", "CORRECT": "correct"}.get(c.verdict, "abstain")


def base_pred(fn, p):
    r = fn(p["buggy_method"], p["patched_method"],
           key_var=p.get("key_var", ""), resource_type=p.get("resource_type", ""))
    return "incorrect" if r.verdict == "INCORRECT" else "correct"


def structural_flags(p):
    """Structure of the patch, read from the IR, used to slice the corpus."""
    kv, rt = p.get("key_var", ""), p.get("resource_type", "")
    ir_b = build_ir(p["buggy_method"], key_var=kv, resource_type=rt)
    ir_p = build_ir(p["patched_method"], key_var=kv, resource_type=rt)
    f = {"parse": ir_b.parse_ok and ir_p.parse_ok}
    if not f["parse"]:
        return {**f, "wrapper": False, "multi": False, "caller": False,
                "rename": False, "exceptional": False, "splitmerge": False}
    obl_b = [r for r in ir_b.resources.values() if r.carries_obligation]
    obl_p = [r for r in ir_p.resources.values() if r.carries_obligation]
    f["wrapper"] = any(r.wraps or r.wrapped_by
                       for ir in (ir_b, ir_p) for r in ir.resources.values())
    f["multi"] = max(len(obl_b), len(obl_p)) >= 2
    f["caller"] = any(r.ownership == Ownership.CALLER
                      for ir in (ir_b, ir_p) for r in ir.resources.values())
    lin = compute_lineage(ir_b, ir_p)
    f["splitmerge"] = bool(lin.split or lin.merged)
    rename = False
    for brid, prid in lin.forward.items():
        bs = ir_b.resources[brid].symbols
        ps = ir_p.resources[prid].symbols
        if bs and ps and not (bs & ps):
            rename = True
    f["rename"] = rename or f["splitmerge"]
    gb = {e.guarded for e in ir_b.events if e.guarded != "none"}
    gp = {e.guarded for e in ir_p.events if e.guarded != "none"}
    f["exceptional"] = bool(gp - gb)
    return f


# --------------------------------------------------------------------------- #
def main():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    held = [p for p in corpus if p.get("split") == "heldout"]
    allc = corpus
    print(f"held-out {len(held)}  full {len(allc)}")

    def rows_for(data, predfn):
        out = []
        for p in data:
            pred, reason = predfn(p)
            out.append({"patch_id": p["patch_id"], "project": p["project"],
                        "tool": p["tool"], "gt": p["gt"], "pred": pred,
                        "reason": reason,
                        "family": (p.get("failure_reason") or "correct").split(":")[0]})
        return out

    R = {}  # system -> rows on held-out
    R["rota"] = rows_for(held, lambda p: rota_pred(p))
    R["idiff"] = rows_for(held, lambda p: (base_pred(independent_diff, p), ""))
    R["polc"] = rows_for(held, lambda p: (base_pred(patch_only_lifecycle, p), ""))
    R["wrem"] = rows_for(held, lambda p: (base_pred(warning_removal, p), ""))
    R["unary"] = rows_for(held, lambda p: (base_pred(unary_patched_only, p), ""))

    res = {}

    # ---- A. per-condition attribution ----------------------------------- #
    attr = collections.Counter()
    fam_cond = collections.defaultdict(collections.Counter)
    for p, r in zip(held, R["rota"]):
        if r["pred"] == "incorrect":
            attr[COND.get(r["reason"], r["reason"])] += 1
        fam_cond[r["family"]][
            COND.get(r["reason"], r["reason"]) if r["pred"] == "incorrect"
            else ("ABSTAIN" if r["pred"] == "abstain" else "accept")] += 1
    res["A_condition_attribution"] = dict(attr)
    res["A_partial_fix_attribution"] = dict(fam_cond["partial_fix"])
    res["A_family_condition"] = {k: dict(v) for k, v in fam_cond.items()}

    # ---- B. disagreement analysis --------------------------------------- #
    def disagree(sysrows):
        by = {r["patch_id"]: r for r in sysrows}
        rota_only = collections.Counter()   # rota right, baseline wrong
        base_only = collections.Counter()   # baseline right, rota wrong
        for p, rr in zip(held, R["rota"]):
            if rr["pred"] == "abstain":
                continue
            br = by[rr["patch_id"]]
            ra, rb = rr["pred"] == rr["gt"], br["pred"] == br["gt"]
            fam = rr["family"]
            if ra and not rb:
                rota_only[fam] += 1
            elif rb and not ra:
                base_only[fam] += 1
        return {"rota_right_base_wrong": dict(rota_only),
                "base_right_rota_wrong": dict(base_only),
                "n_rota_only": sum(rota_only.values()),
                "n_base_only": sum(base_only.values())}
    res["B_vs_independent_diff"] = disagree(R["idiff"])
    res["B_vs_patch_only_lifecycle"] = disagree(R["polc"])

    # ---- C. challenge subsets ------------------------------------------- #
    flags = {p["patch_id"]: structural_flags(p) for p in held}
    subsets = {
        "wrapper_dependent": lambda f: f["wrapper"],
        "multi_resource": lambda f: f["multi"],
        "caller_owned": lambda f: f["caller"],
        "rename_rebind_split": lambda f: f["rename"],
        "exceptional_cleanup": lambda f: f["exceptional"],
    }
    chrows = {s: {sys: [] for sys in ("rota", "idiff", "polc", "wrem", "unary")}
              for s in subsets}
    for s, pred in subsets.items():
        ids = {pid for pid, f in flags.items() if f["parse"] and pred(f)}
        for sys in ("rota", "idiff", "polc", "wrem", "unary"):
            chrows[s][sys] = [r for r in R[sys] if r["patch_id"] in ids]
    res["C_challenge_subsets"] = {
        s: {"n": len(chrows[s]["rota"]),
            **{sys: metrics(chrows[s][sys]) for sys in
               ("rota", "idiff", "polc", "wrem", "unary")}}
        for s in subsets}

    # ---- D. leave-one-generator-out ------------------------------------- #
    logo = {}
    for tool in ("rlfixer", "arodnap", "amur"):
        unseen = [p for p in allc if p["tool"] == tool]
        rr = rows_for(unseen, lambda p: rota_pred(p))
        ir = rows_for(unseen, lambda p: (base_pred(independent_diff, p), ""))
        pr = rows_for(unseen, lambda p: (base_pred(patch_only_lifecycle, p), ""))
        logo[tool] = {"n": len(unseen),
                      "rota": metrics(rr), "idiff": metrics(ir),
                      "polc": metrics(pr),
                      "rota_abstain_wrong": acc_abstain_wrong(rr)}
    res["D_leave_one_generator_out"] = logo

    # ---- E. kappa/tau anomaly ------------------------------------------- #
    konly = rows_for(held, lambda p: (rota_pred_inputs(p, True, False), ""))
    ktau = rows_for(held, lambda p: (rota_pred_inputs(p, True, True), ""))
    konly_by = {r["patch_id"]: r for r in konly}
    flip = collections.Counter()
    flip_fam = collections.Counter()
    for p, rk in zip(held, konly):
        rt = next(r for r in ktau if r["patch_id"] == rk["patch_id"])
        k_ok = rk["pred"] == rk["gt"]
        t_ok = rt["pred"] == rt["gt"]
        if k_ok and not t_ok:
            flip[(p.get("resource_type", "") or "?").split(".")[-1]] += 1
            flip_fam[rk["family"]] += 1
    res["E_kappa_only_right_kappatau_wrong"] = {
        "count": sum(flip.values()),
        "by_resource_type": dict(flip),
        "by_family": dict(flip_fam),
        "kappa_only": metrics(konly), "kappa_tau": metrics(ktau)}

    # ---- F. lineage accuracy on oracle correspondences ------------------ #
    oracle_total = oracle_hit = 0
    amb = 0
    for p in allc:
        f = structural_flags(p)  # cheap re-parse
        kv, rt = p.get("key_var", ""), p.get("resource_type", "")
        ir_b = build_ir(p["buggy_method"], key_var=kv, resource_type=rt)
        ir_p = build_ir(p["patched_method"], key_var=kv, resource_type=rt)
        if not (ir_b.parse_ok and ir_p.parse_ok):
            continue
        ob = [(rid, r) for rid, r in ir_b.resources.items() if r.carries_obligation]
        op = [(rid, r) for rid, r in ir_p.resources.items() if r.carries_obligation]
        lin = compute_lineage(ir_b, ir_p)
        # Oracle 1: exactly one obligation each side -> must correspond.
        if len(ob) == 1 and len(op) == 1:
            oracle_total += 1
            if lin.forward.get(ob[0][0]) == op[0][0]:
                oracle_hit += 1
            continue
        # Oracle 2: name-preserved obligations (unique shared primary symbol).
        for brid, br in ob:
            cand = [prid for prid, pr in op if (br.symbols & pr.symbols)]
            if len(cand) == 1:
                oracle_total += 1
                if lin.forward.get(brid) == cand[0]:
                    oracle_hit += 1
            elif len(cand) > 1:
                amb += 1
    res["F_lineage_oracle"] = {
        "oracle_pairs": oracle_total,
        "correct": oracle_hit,
        "accuracy": round(oracle_hit / oracle_total, 4) if oracle_total else 0.0,
        "ambiguous_skipped": amb}

    # ---- G. abstention variants ----------------------------------------- #
    rota_decided_ids = {r["patch_id"] for r in R["rota"] if r["pred"] != "abstain"}
    common = {sys: [r for r in R[sys] if r["patch_id"] in rota_decided_ids]
              for sys in R}
    res["G_abstention"] = {
        "decided_only": metrics(R["rota"]),
        "abstain_as_wrong_accuracy": acc_abstain_wrong(R["rota"]),
        "common_coverage": {sys: metrics(common[sys]) for sys in R},
        "selective_point": {"coverage": metrics(R["rota"])["coverage"],
                            "risk": round(1 - metrics(R["rota"])["accuracy"], 4)}}

    # ---- H1. project-cluster bootstrap ---------------------------------- #
    def cluster_bootstrap(rows, iters=2000, seed=7):
        rng = random.Random(seed)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["project"]].append(r)
        projs = list(by)
        vals = []
        for _ in range(iters):
            samp = []
            for _ in range(len(projs)):
                samp += by[projs[rng.randrange(len(projs))]]
            vals.append(metrics(samp)["f1"])
        vals.sort()
        return [round(vals[int(0.025 * iters)], 4), round(vals[int(0.975 * iters)], 4)]
    res["H_project_cluster_bootstrap_f1_ci"] = cluster_bootstrap(R["rota"])

    # ---- H2. hyperparameter sensitivity --------------------------------- #
    orig_lineage = checker.compute_lineage
    orig_radius = checker.FRONTIER_RADIUS
    theta_sens = {}
    for theta in (1.0, 1.5, 2.0, 2.5, 3.0):
        checker.compute_lineage = (
            lambda a, b, _t=theta, **k: orig_lineage(a, b, min_score=_t))
        rr = rows_for(held, lambda p: rota_pred(p))
        theta_sens[theta] = metrics(rr)["accuracy"]
    checker.compute_lineage = orig_lineage
    radius_sens = {}
    for rad in (1, 2, 3):
        checker.FRONTIER_RADIUS = rad
        rr = rows_for(held, lambda p: rota_pred(p))
        radius_sens[rad] = metrics(rr)["accuracy"]
    checker.FRONTIER_RADIUS = orig_radius
    res["H_theta_sensitivity_acc"] = theta_sens
    res["H_radius_sensitivity_acc"] = radius_sens

    out = os.path.join(ROOT, "results", "fse_analysis.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)
    print(json.dumps({k: res[k] for k in (
        "A_condition_attribution", "A_partial_fix_attribution",
        "B_vs_independent_diff", "B_vs_patch_only_lifecycle",
        "D_leave_one_generator_out", "E_kappa_only_right_kappatau_wrong",
        "F_lineage_oracle", "H_theta_sensitivity_acc",
        "H_radius_sensitivity_acc", "H_project_cluster_bootstrap_f1_ci")},
        indent=1))


if __name__ == "__main__":
    main()
