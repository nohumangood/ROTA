"""Independent certificate checker.

A ROTA certificate is machine-checkable: its verdict is a deterministic function
of named, reproducible facts (the obligation transformation of each resource on
the effect frontier). This script demonstrates that by re-deriving the verdict
*without* calling the production :func:`rota.checker.check`. It instead

  1. rebuilds the per-version substrate and the cross-version lineage from
     <M, M', kappa, tau> alone (reproducing the transformations), and
  2. re-applies the four refinement conditions with an independent copy of the
     decision logic written here,

and then confirms the recomputed <verdict, reason, target> equals the one the
certificate claims. A certificate that fails either step is rejected.

Run over a split, it reports the fraction of certificates that an independent
party can replay and confirm. This is what "machine-checkable certificate"
means, and is the property an opaque classifier cannot offer.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rota.checker import run_rota  # noqa: E402
from rota.ir import build_ir  # noqa: E402
from rota.lineage import compute_lineage, target_resources  # noqa: E402
from rota.relational import ObligationState, Transformation, analyze  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "data", "rota_corpus.json")

FRONTIER_RADIUS = 2
BLOCKING = {Transformation.INVALIDATE, Transformation.INTRODUCE,
            Transformation.ORPHANED}
REPAIRING = {Transformation.DISCHARGE, Transformation.ESCAPE_TRANSFER,
             Transformation.TRANSFER, Transformation.SPLIT, Transformation.MERGE}
PENDING = {ObligationState.UNSATISFIED, ObligationState.SATISFIED_WEAK}
LOCALLY_OWNED = {ObligationState.UNSATISFIED, ObligationState.SATISFIED_WEAK,
                 ObligationState.SATISFIED_SAFE}
REASON = {Transformation.INVALIDATE: "PATCH_INDUCED_LIFECYCLE_INVALIDATION",
          Transformation.INTRODUCE: "PATCH_INTRODUCED_NEW_OBLIGATION",
          Transformation.ORPHANED: "ORPHANED_OBLIGATION"}


def independent_verdict(result):
    """Re-derive <verdict, reason, target> from a transformation set.

    A standalone reimplementation of the refinement relation, kept separate from
    rota.checker so that a defect there would surface as a mismatch.
    """
    vs = result.verdicts
    target = ""
    tv = [v for v in vs if v.is_target]
    if tv:
        best = min(tv, key=lambda v: 0 if v.transformation in REPAIRING else 1)
        target = best.name
    # C1: no blocking transformation on the frontier. When several are present
    # the certificate names the last in frontier order, so match that.
    blocking = [v for v in vs if v.transformation in BLOCKING]
    if blocking:
        return "INCORRECT", REASON[blocking[-1].transformation], target
    # C2: the reported obligation is discharged (and safely). Charged strictly
    # against the target set (kappa and its tau-typed siblings); non-target
    # frontier siblings belong to C3, so the two conditions stay disjoint.
    pend = [v for v in vs if v.state_before in PENDING and v.is_target]
    if pend:
        unfixed = [v for v in pend if v.state_after == ObligationState.UNSATISFIED]
        weak = [v for v in pend if v.state_after == ObligationState.SATISFIED_WEAK]
        if unfixed:
            return "INCORRECT", "TARGET_OBLIGATION_NOT_DISCHARGED", target
        if weak:
            return "INCORRECT", "TARGET_RELEASE_NOT_EXCEPTION_SAFE", target
    # C3: obligation conservation (non-target coupled sibling left pending).
    if tv:
        stranded = [v for v in vs if not v.is_target
                    and v.state_after == ObligationState.UNSATISFIED
                    and v.state_before in PENDING
                    and v.frontier_distance <= FRONTIER_RADIUS]
        if stranded:
            return "INCORRECT", "COUPLED_OBLIGATION_LEFT_PENDING", target
    # C4: the patch must act on the obligation structure.
    if any(v.state_before in LOCALLY_OWNED for v in vs) and not any(
            v.transformation in REPAIRING for v in vs):
        return "INCORRECT", "NO_REPAIRING_TRANSFORMATION", target
    return "CORRECT", "obligation refinement holds", target


def verify(p):
    kv, rt = p.get("key_var", ""), p.get("resource_type", "")
    cert = run_rota(p["buggy_method"], p["patched_method"],
                    key_var=kv, resource_type=rt).certificate
    if cert.reason in ("UNPARSED", "NO_OBLIGATION_MODELLED"):
        return "abstain", True
    # Independent replay from <M, M', kappa, tau> only.
    ir_b = build_ir(p["buggy_method"], key_var=kv, resource_type=rt)
    ir_p = build_ir(p["patched_method"], key_var=kv, resource_type=rt)
    lineage = compute_lineage(ir_b, ir_p, min_score=2.0)
    result = analyze(ir_b, ir_p, lineage, set(target_resources(ir_b, kv, rt)))
    v, r, t = independent_verdict(result)
    decision_ok = (v == cert.verdict and r == cert.reason)
    full_ok = decision_ok and t == cert.target
    return "decided", decision_ok, full_ok


def main():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    for split in ("heldout", "all"):
        data = corpus if split == "all" else [p for p in corpus
                                              if p.get("split") == split]
        checked = dec = full = 0
        for p in data:
            res = verify(p)
            if res[0] == "abstain":
                continue
            checked += 1
            dec += int(res[1])
            full += int(res[2])
        print(f"{split:>8}: verdict+reason {dec}/{checked} "
              f"({dec / checked:.4f}); +target {full}/{checked} "
              f"({full / checked:.4f})")


if __name__ == "__main__":
    main()
