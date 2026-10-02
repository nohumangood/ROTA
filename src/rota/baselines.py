"""Reference points that isolate what the relational formulation contributes.

The ladder below is the paper's central comparison. Each rung is given the same
inputs and the same substrate as ROTA; they differ only in how much of the
relation between the two versions they are allowed to see. If a unary or
differential formulation could reach ROTA's accuracy, the relational domain
would not be earning its place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from rota.ir import MethodIR, build_ir
from rota.relational import (
    ObligationState,
    obligation_state,
    propagated_releases,
    use_after_release,
)


@dataclass
class Prediction:
    verdict: str              # "CORRECT" | "INCORRECT"
    reason: str
    parse_ok: bool = True


def _warnings(ir: MethodIR) -> Set[str]:
    """Resource-leak warnings a single-version analysis would raise.

    Keyed by allocation origin and type rather than by internal id, so the two
    versions' warnings are comparable as sets - which is precisely the move a
    differential baseline makes, and precisely what loses the identity of an
    obligation when the patch renames or rebinds it.
    """
    prop = propagated_releases(ir)
    uar = {rid for rid, _, _ in use_after_release(ir)}
    out: Set[str] = set()
    for rid, res in ir.resources.items():
        state = obligation_state(ir, rid, prop)
        if state == ObligationState.UNSATISFIED:
            out.add(f"leak:{res.simple_type}:{res.from_call}")
        elif state == ObligationState.SATISFIED_WEAK:
            out.add(f"epath:{res.simple_type}:{res.from_call}")
        if rid in uar:
            out.add(f"uar:{res.simple_type}:{res.from_call}")
    return out


def _irs(buggy: str, patched: str, key_var: str, resource_type: str):
    ir_b = build_ir(buggy, key_var=key_var, resource_type=resource_type)
    ir_p = build_ir(patched, key_var=key_var, resource_type=resource_type)
    return ir_b, ir_p


def unary_patched_only(buggy: str, patched: str, *, key_var: str = "",
                       resource_type: str = "") -> Prediction:
    """Rung 1: look only at M'. The buggy version is never consulted."""
    ir_p = build_ir(patched, key_var=key_var, resource_type=resource_type)
    if not ir_p.parse_ok:
        return Prediction("CORRECT", "UNPARSED", parse_ok=False)
    w = _warnings(ir_p)
    return Prediction("INCORRECT", f"warning:{sorted(w)[0]}") if w \
        else Prediction("CORRECT", "no warning in patched version")


def independent_analysis(buggy: str, patched: str, *, key_var: str = "",
                         resource_type: str = "") -> Prediction:
    """Rung 2: analyse both versions, accept when M' is clean.

    Reading M does not help here because the two analyses never meet; this rung
    exists to show that merely *having* both versions is not the same as
    relating them.
    """
    ir_b, ir_p = _irs(buggy, patched, key_var, resource_type)
    if not ir_b.parse_ok or not ir_p.parse_ok:
        return Prediction("CORRECT", "UNPARSED", parse_ok=False)
    wb, wp = _warnings(ir_b), _warnings(ir_p)
    if not wb:
        return Prediction("CORRECT", "no obligation reported in buggy version")
    return Prediction("INCORRECT", f"warning:{sorted(wp)[0]}") if wp \
        else Prediction("CORRECT", "patched version analyses clean")


def independent_diff(buggy: str, patched: str, *, key_var: str = "",
                     resource_type: str = "") -> Prediction:
    """Rung 3: run the analysis twice and subtract the warning sets.

    This is the strongest formulation available without a cross-version notion
    of obligation identity, and the one ROTA has to beat to justify lineage.
    """
    ir_b, ir_p = _irs(buggy, patched, key_var, resource_type)
    if not ir_b.parse_ok or not ir_p.parse_ok:
        return Prediction("CORRECT", "UNPARSED", parse_ok=False)
    wb, wp = _warnings(ir_b), _warnings(ir_p)
    new = wp - wb
    if new:
        return Prediction("INCORRECT", f"new warning:{sorted(new)[0]}")
    if wb - wp:
        return Prediction("CORRECT", "reported warnings removed")
    return Prediction("INCORRECT", "no warning removed")


def warning_removal(buggy: str, patched: str, *, key_var: str = "",
                    resource_type: str = "") -> Prediction:
    """APCA baseline: accept exactly when the reported warning disappears."""
    ir_b, ir_p = _irs(buggy, patched, key_var, resource_type)
    if not ir_b.parse_ok or not ir_p.parse_ok:
        return Prediction("CORRECT", "UNPARSED", parse_ok=False)
    wb, wp = _warnings(ir_b), _warnings(ir_p)
    removed = {w for w in wb if w.startswith(("leak:", "epath:"))} - wp
    return Prediction("CORRECT", "reported warning removed") if removed \
        else Prediction("INCORRECT", "reported warning still present")


def patch_only_lifecycle(buggy: str, patched: str, *, key_var: str = "",
                         resource_type: str = "") -> Prediction:
    """APCA baseline: a lifecycle check restricted to the patched version's
    target resource, with no reference to the buggy version at all."""
    ir_p = build_ir(patched, key_var=key_var, resource_type=resource_type)
    if not ir_p.parse_ok:
        return Prediction("CORRECT", "UNPARSED", parse_ok=False)
    from rota.lineage import target_resources

    prop = propagated_releases(ir_p)
    tgt = target_resources(ir_p, key_var, resource_type)
    if not tgt:
        return Prediction("CORRECT", "no target resource in patched version")
    states = [obligation_state(ir_p, rid, prop) for rid in tgt]
    if any(s == ObligationState.UNSATISFIED for s in states):
        return Prediction("INCORRECT", "target still unreleased")
    if any(s == ObligationState.SATISFIED_WEAK for s in states):
        return Prediction("INCORRECT", "target released on the normal path only")
    return Prediction("CORRECT", "target lifecycle complete")


LADDER = {
    "unary_patched_only": unary_patched_only,
    "independent_analysis": independent_analysis,
    "independent_diff": independent_diff,
}

APCA_BASELINES = {
    "warning_removal": warning_removal,
    "patch_only_lifecycle": patch_only_lifecycle,
}
