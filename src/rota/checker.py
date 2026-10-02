"""Obligation Refinement Checking: the verdict and its certificate.

A patch is correct when it is a *refinement* of the buggy program's resource
obligations, not when the patched program independently satisfies a resource
property. Refinement here is deliberately asymmetric: the patch must change
behaviour (that is the repair), so equivalence is the wrong relation. What must
hold is

``TargetDischarged``
    the reported obligation is discharged, legally transferred, or escaped;
``NoInvalidTransformation``
    nothing previously valid was invalidated and no new violation introduced;
``ObligationConserved``
    every obligation present in M is accounted for in M' - discharged,
    preserved, transferred, or escaped - and none merely vanished.

Each failing condition yields a witness, so an INCORRECT verdict always names a
concrete obligation and the transformation that broke it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rota.ir import Ownership, build_ir
from rota.lineage import compute_lineage, target_resources
from rota.relational import (
    ObligationState,
    RelationalResult,
    Transformation,
    analyze,
)

#: Transformations that, on any obligation, make the patch incorrect.
#: How far from the patched resources an obligation may sit and still be the
#: patch's responsibility, in wrapper/alias/ownership hops.
FRONTIER_RADIUS = 2

#: States in M that mean the method itself held an obligation. A method that
#: only borrows a caller's resource has nothing to repair, so the absence of a
#: repairing transformation there says nothing about the patch.
LOCALLY_OWNED = {
    ObligationState.UNSATISFIED,
    ObligationState.SATISFIED_WEAK,
    ObligationState.SATISFIED_SAFE,
}

#: States in M that leave an obligation outstanding on at least one path.
PENDING_BEFORE = {
    ObligationState.UNSATISFIED,
    ObligationState.SATISFIED_WEAK,
}

BLOCKING = {
    Transformation.INVALIDATE,
    Transformation.INTRODUCE,
    Transformation.ORPHANED,
}

#: Transformations that discharge the reported defect.
REPAIRING = {
    Transformation.DISCHARGE,
    Transformation.ESCAPE_TRANSFER,
    Transformation.TRANSFER,
    Transformation.SPLIT,
    Transformation.MERGE,
}


@dataclass
class Certificate:
    """Why the verdict is what it is."""

    verdict: str                       # CORRECT | INCORRECT
    reason: str
    target: str = ""
    target_transformation: str = ""
    witnesses: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    n_resources: int = 0
    n_frontier: int = 0
    parse_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "target": self.target,
            "target_transformation": self.target_transformation,
            "witnesses": self.witnesses,
            "transformations": self.transformations,
            "n_resources": self.n_resources,
            "n_frontier": self.n_frontier,
            "parse_ok": self.parse_ok,
        }


def check(result: RelationalResult, *, require_target: bool = True,
          conservation: bool = True) -> Certificate:
    """Apply the refinement conditions to a set of transformation verdicts."""
    cert = Certificate(verdict="CORRECT", reason="obligation refinement holds")
    cert.n_resources = len(result.verdicts)
    cert.n_frontier = len(result.frontier)
    cert.transformations = [
        f"{v.name}:{v.state_before.value}->{v.state_after.value}={v.transformation.value}"
        for v in result.verdicts
    ]

    targets = [v for v in result.verdicts if v.is_target]
    if targets:
        # Report the strongest repair evidence available for the target.
        best = min(targets, key=lambda v: 0 if v.transformation in REPAIRING else 1)
        cert.target = best.name
        cert.target_transformation = best.transformation.value

    # Condition 1: no invalid transformation anywhere in the effect frontier.
    for v in result.verdicts:
        if v.transformation in BLOCKING:
            cert.verdict = "INCORRECT"
            cert.reason = _reason_for(v.transformation)
            cert.witnesses.append(
                f"{v.name}: {v.state_before.value} -> {v.state_after.value} "
                f"= {v.transformation.value}"
                + (f" ({v.witness})" if v.witness else "")
            )
    if cert.verdict == "INCORRECT":
        return cert

    # Condition 2: the reported obligation must actually be discharged.
    # This condition is charged strictly against the *target* obligations - the
    # resource kappa names and its tau-typed siblings, which target_resources
    # pulls into the target set. Non-target frontier siblings are deliberately
    # excluded here: they are the responsibility of Condition 3 (conservation),
    # so that the two conditions are disjoint by construction rather than one
    # silently subsuming the other. If no target carried a pending obligation in
    # M, the obligation ROTA located is not the one the report refers to, and
    # charging the patch for it would turn a localisation miss into a verdict.
    #
    # An obligation counts as pending in M when it is unsatisfied *or* released
    # only on the normal path. Excluding the latter would drop every defect
    # whose buggy version already closes the resource and leaks it only when an
    # exception escapes.
    pending_targets = [
        v for v in result.verdicts
        if v.state_before in PENDING_BEFORE and v.is_target
    ]
    if require_target and pending_targets:
        unfixed = [v for v in pending_targets
                   if v.state_after == ObligationState.UNSATISFIED]
        # A release reached only on the normal path still leaks whenever an
        # exception escapes, so it does not discharge the obligation. This is a
        # judgement about release strength, charged only against target
        # obligations.
        weak = [v for v in pending_targets
                if v.state_after == ObligationState.SATISFIED_WEAK]
        if unfixed:
            cert.verdict = "INCORRECT"
            cert.reason = "TARGET_OBLIGATION_NOT_DISCHARGED"
            cert.witnesses.append(
                f"{unfixed[0].name}: still {unfixed[0].state_after.value} after patch"
            )
        elif weak:
            cert.verdict = "INCORRECT"
            cert.reason = "TARGET_RELEASE_NOT_EXCEPTION_SAFE"
            cert.witnesses.append(
                f"{weak[0].name}: released on the normal path only"
            )
    if cert.verdict == "INCORRECT":
        return cert

    # Condition 3, obligation conservation. A repair that discharges the target
    # obligation but leaves a coupled *non-target* one outstanding has not
    # transformed the defect, only part of it. This condition now owns the
    # non-target frontier siblings outright: Condition 2 no longer touches them,
    # so a cross-type coupled leak (a stranded sibling whose type is not tau's
    # family, hence not a target) is caught here and nowhere else. The effect
    # frontier is what makes this checkable: it bounds the obligations the patch
    # is answerable for, so the condition does not degenerate into "the method
    # must be leak-free".
    if require_target and targets and conservation:
        # A sibling obligation the patch left unsatisfied is a partial fix. This
        # is the relational reading of "M' still leaks": unlike a warning diff
        # it fires only for an obligation that was pending in M and that M'
        # genuinely strands, not for a borrowed resource the method never had to
        # close.
        stranded = [
            v for v in result.verdicts
            if not v.is_target
            and v.state_after == ObligationState.UNSATISFIED
            and v.state_before in PENDING_BEFORE
            and v.frontier_distance <= FRONTIER_RADIUS
        ]
        if stranded:
            cert.verdict = "INCORRECT"
            cert.reason = "COUPLED_OBLIGATION_LEFT_PENDING"
            cert.witnesses.append(
                f"{stranded[0].name}: unsatisfied at frontier distance "
                f"{stranded[0].frontier_distance}"
            )
    if cert.verdict == "INCORRECT":
        return cert

    # Condition 4. A repair must *do* something to the obligation structure. If
    # every obligation is carried across unchanged, the patch edited code that
    # has no bearing on the defect, however plausible the edit looks. Only the
    # paired view can state this: in either version alone the obligations are
    # simply whatever they are.
    had_local_obligation = any(
        v.state_before in LOCALLY_OWNED for v in result.verdicts
    )
    if require_target and had_local_obligation and not any(
        v.transformation in REPAIRING for v in result.verdicts
    ):
        cert.verdict = "INCORRECT"
        cert.reason = "NO_REPAIRING_TRANSFORMATION"
        cert.witnesses.append(
            f"{len(result.verdicts)} obligations, none transformed into a discharge"
        )

    return cert


def _reason_for(t: Transformation) -> str:
    return {
        Transformation.INVALIDATE: "PATCH_INDUCED_LIFECYCLE_INVALIDATION",
        Transformation.INTRODUCE: "PATCH_INTRODUCED_NEW_OBLIGATION",
        Transformation.ORPHANED: "ORPHANED_OBLIGATION",
    }[t]


# --------------------------------------------------------------------------- #
# end-to-end entry point
# --------------------------------------------------------------------------- #

@dataclass
class RotaOutput:
    certificate: Certificate
    result: Optional[RelationalResult] = None


def run_rota(
    buggy_method: str,
    patched_method: str,
    *,
    key_var: str = "",
    resource_type: str = "",
    use_lineage: bool = True,
    use_frontier: bool = True,
    use_conservation: bool = True,
    use_propagation: bool = True,
    use_ownership: bool = True,
    use_path_correspondence: bool = True,
    require_target: bool = True,
) -> RotaOutput:
    """Analyse one patch. Inputs are exactly <M, M', kappa, tau>.

    The ablation flags disable individual relational components so that each
    one's contribution can be measured; they never admit extra inputs.
    """
    ir_b = build_ir(buggy_method, key_var=key_var, resource_type=resource_type,
                    path_correspondence=use_path_correspondence)
    ir_p = build_ir(patched_method, key_var=key_var, resource_type=resource_type,
                    path_correspondence=use_path_correspondence)
    if not ir_b.parse_ok or not ir_p.parse_ok:
        return RotaOutput(Certificate("CORRECT", "UNPARSED", parse_ok=False))

    if not use_ownership:
        # Without the ownership distinction every resource looks locally owned,
        # so closing a caller's resource no longer reads as an invalidation.
        for ir in (ir_b, ir_p):
            for res in ir.resources.values():
                res.ownership = Ownership.LOCAL
                res.is_parameter = False

    if not use_propagation:
        for ir in (ir_b, ir_p):
            for res in ir.resources.values():
                res.wraps = None
                res.wrapped_by = None

    lineage = compute_lineage(ir_b, ir_p, min_score=2.0 if use_lineage else 10 ** 6)
    targets = set(target_resources(ir_b, key_var, resource_type))
    result = analyze(ir_b, ir_p, lineage, targets)

    if not use_frontier:
        result.verdicts = [v for v in result.verdicts if v.is_target]
    if not use_conservation:
        result.verdicts = [
            v for v in result.verdicts if v.transformation != Transformation.ORPHANED
        ]

    # A certifying analysis states what it cannot certify. When neither version
    # exposes a resource obligation, ROTA has nothing to reason about and says
    # so, rather than defaulting to a verdict the evidence does not support. The
    # abstention rate is a reported coverage figure, not a hidden guess.
    if require_target and not result.verdicts:
        return RotaOutput(Certificate("ABSTAIN", "NO_OBLIGATION_MODELLED",
                                      parse_ok=True), result)

    cert = check(result, require_target=require_target, conservation=use_conservation)
    return RotaOutput(cert, result)
