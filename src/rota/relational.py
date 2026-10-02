"""The relational core of ROTA: paired obligation states and their transformation.

A single-version analysis asks whether a program satisfies a resource property.
ROTA asks a different question: what did the patch *do* to each obligation. The
unit of analysis is therefore not a state but a pair of states plus the lineage
that says the two belong to the same obligation.

The distinction is load-bearing rather than cosmetic. The same state delta means
opposite things depending on the resource's role: ``LIVE -> RELEASED`` on a
locally owned leaking resource is the repair, while the identical delta on a
caller-owned parameter destroys a lifetime the caller still depends on. Only the
paired view, carrying ownership and lineage, can tell those apart - which is why
running a leak checker twice and subtracting cannot express this property.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rota.ir import (
    AbstractResource,
    EventKind,
    MethodIR,
    Ownership,
    close_propagates,
)
from rota.lineage import Lineage


class ObligationState(Enum):
    """State of one resource obligation within a single program version."""

    NONE = "none"                      # no obligation (no-op type, or borrowed)
    UNSATISFIED = "unsatisfied"        # acquired, never released -> leak
    SATISFIED_SAFE = "satisfied_safe"  # released under finally / try-with-resources
    SATISFIED_WEAK = "satisfied_weak"  # released on the normal path only
    TRANSFERRED = "transferred"        # returned or stored; obligation left the method
    CALLER_LIVE = "caller_live"        # caller-owned and still usable here
    CALLER_CLOSED = "caller_closed"    # caller-owned but released here


class Transformation(Enum):
    """What the patch did to one obligation. This is ROTA's semantic domain."""

    PRESERVE = "preserve"
    DISCHARGE = "discharge"
    WEAK_DISCHARGE = "weak_discharge"
    UNFIXED = "unfixed"
    INTRODUCE = "introduce"
    INVALIDATE = "invalidate"
    TRANSFER = "transfer"
    SPLIT = "split"
    MERGE = "merge"
    ESCAPE_TRANSFER = "escape_transfer"
    ORPHANED = "orphaned"


@dataclass
class ResourceVerdict:
    """One obligation, its two states, and the transformation between them."""

    buggy_rid: Optional[str]
    patch_rid: Optional[str]
    name: str
    state_before: ObligationState
    state_after: ObligationState
    transformation: Transformation
    is_target: bool = False
    ownership: Ownership = Ownership.UNKNOWN
    witness: str = ""
    frontier_distance: int = 0


@dataclass
class RelationalResult:
    verdicts: List[ResourceVerdict] = field(default_factory=list)
    frontier: Set[str] = field(default_factory=set)
    aligned: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# per-version obligation state
# --------------------------------------------------------------------------- #

def _releases(ir: MethodIR, rid: str) -> List:
    return [e for e in ir.events if e.resource == rid and e.kind == EventKind.RELEASE]


def propagated_releases(ir: MethodIR) -> Dict[str, List[Tuple[str, int, str]]]:
    """Resources released indirectly because something wrapping them was closed.

    Returns ``delegate_rid -> [(wrapper_rid, line, guard)]``. Wrapper closes are
    not a special-cased rule here: they are an obligation-propagation relation
    that feeds the same state machine as a direct release.
    """
    out: Dict[str, List[Tuple[str, int, str]]] = {}
    for rid, res in ir.resources.items():
        if not res.wraps:
            continue
        if close_propagates(res) is False:
            continue
        for ev in _releases(ir, rid):
            chain, cur = [], res.wraps
            # Closing a wrapper reaches every delegate down the chain.
            while cur:
                chain.append(cur)
                nxt = ir.resources.get(cur)
                cur = nxt.wraps if nxt else None
            for delegate in chain:
                out.setdefault(delegate, []).append((rid, ev.line, ev.guarded))
    return out


def use_after_release(ir: MethodIR) -> List[Tuple[str, int, str]]:
    """Uses of a resource that occur after it has been released.

    Covers propagated releases, which is the case a patch most often gets wrong:
    the patch closes a wrapper it created and thereby invalidates a stream the
    method still reads from.
    """
    prop = propagated_releases(ir)
    out: List[Tuple[str, int, str]] = []
    for rid in ir.resources:
        rel_lines = [e.line for e in _releases(ir, rid)]
        rel_lines += [line for _, line, _ in prop.get(rid, [])]
        if not rel_lines:
            continue
        first = min(rel_lines)
        for ev in ir.events:
            if ev.resource != rid or ev.kind != EventKind.USE:
                continue
            if ev.line > first:
                via = next((w for w, line, _ in prop.get(rid, []) if line <= ev.line), None)
                out.append((rid, ev.line, f"use {ev.text} after release{f' via {via}' if via else ''}"))
    return out


def obligation_state(ir: MethodIR, rid: str, prop: Dict[str, List]) -> ObligationState:
    """Collapse a resource's events and guards into one obligation state."""
    res = ir.resources.get(rid)
    if res is None:
        return ObligationState.NONE

    direct = _releases(ir, rid)
    indirect = prop.get(rid, [])

    if res.is_parameter or res.ownership == Ownership.CALLER:
        return ObligationState.CALLER_CLOSED if (direct or indirect) else ObligationState.CALLER_LIVE

    if not res.carries_obligation:
        return ObligationState.NONE

    if res.ownership in (Ownership.ESCAPED, Ownership.FIELD):
        return ObligationState.TRANSFERRED

    guards = [e.guarded for e in direct] + [g for _, _, g in indirect]
    if any(g in ("finally", "try-with-resources") for g in guards):
        return ObligationState.SATISFIED_SAFE
    if direct or indirect:
        return ObligationState.SATISFIED_WEAK
    return ObligationState.UNSATISFIED


_DIVERTS = ("return", "throw", "break", "continue")


def _no_diverting_path(ir: MethodIR, acquire_line: int, release_line: int) -> bool:
    """Whether control must reach ``release_line`` once ``acquire_line`` runs.

    Conservative: any try region, branch, or abrupt transfer in between is
    enough to admit a path that skips the release.
    """
    if release_line < acquire_line:
        return False
    for n in ir.nodes:
        if not (acquire_line <= n.line < release_line):
            continue
        if n.in_try or n.in_finally or n.kind in ("try", "catch", "finally", "branch"):
            return False
        if n.text.lstrip().startswith(_DIVERTS):
            return False
    return True


_SATISFIED = (ObligationState.SATISFIED_SAFE, ObligationState.SATISFIED_WEAK)


def classify(
    before: ObligationState,
    after: ObligationState,
    ownership: Ownership,
    *,
    invalidated: bool,
) -> Transformation:
    """Map a paired state onto the transformation lattice.

    Role matters as much as the states: the caller-owned branch below is the
    reason this cannot be reduced to a state diff.
    """
    if invalidated:
        return Transformation.INVALIDATE

    if before in (ObligationState.CALLER_LIVE, ObligationState.CALLER_CLOSED):
        if before == ObligationState.CALLER_LIVE and after == ObligationState.CALLER_CLOSED:
            # The method took on releasing something it does not own.
            return Transformation.INVALIDATE
        return Transformation.PRESERVE

    if before == ObligationState.UNSATISFIED:
        if after == ObligationState.SATISFIED_SAFE:
            return Transformation.DISCHARGE
        if after == ObligationState.SATISFIED_WEAK:
            return Transformation.WEAK_DISCHARGE
        if after == ObligationState.TRANSFERRED:
            return Transformation.ESCAPE_TRANSFER
        if after == ObligationState.NONE:
            return Transformation.ORPHANED
        return Transformation.UNFIXED

    if before in _SATISFIED:
        if after == ObligationState.UNSATISFIED:
            return Transformation.INTRODUCE
        if after == ObligationState.NONE:
            return Transformation.ORPHANED
        if before == ObligationState.SATISFIED_WEAK and after == ObligationState.SATISFIED_SAFE:
            # Moving a normal-path release under finally or try-with-resources
            # discharges the obligation on the exceptional paths too, which is
            # the repair for a noCloseEPath defect.
            return Transformation.DISCHARGE
        if before == ObligationState.SATISFIED_SAFE and after == ObligationState.SATISFIED_WEAK:
            return Transformation.INTRODUCE
        return Transformation.PRESERVE

    if before == ObligationState.TRANSFERRED:
        if after == ObligationState.UNSATISFIED:
            return Transformation.INTRODUCE
        return Transformation.PRESERVE

    # before == NONE
    if after == ObligationState.UNSATISFIED:
        return Transformation.INTRODUCE
    return Transformation.PRESERVE


# --------------------------------------------------------------------------- #
# effect frontier
# --------------------------------------------------------------------------- #

def effect_frontier(ir_p: MethodIR, seeds: Set[str], *, max_depth: int = 3) -> Dict[str, int]:
    """Resources reachable from the patched resources by obligation-carrying edges.

    The analysis scope is not the syntactic patch region. Adding one
    ``close()`` can invalidate a resource several wrapper or alias hops away, so
    the frontier follows wrapper, alias, and ownership edges outward from the
    resources the patch touched and records the hop distance.
    """
    dist: Dict[str, int] = {s: 0 for s in seeds if s in ir_p.resources}
    frontier = list(dist)
    while frontier:
        cur = frontier.pop(0)
        d = dist[cur]
        if d >= max_depth:
            continue
        res = ir_p.resources[cur]
        neighbours: Set[str] = set()
        if res.wraps:
            neighbours.add(res.wraps)
        if res.wrapped_by:
            neighbours.add(res.wrapped_by)
        for other_rid, other in ir_p.resources.items():
            if other_rid == cur:
                continue
            if other.symbols & res.symbols:          # alias
                neighbours.add(other_rid)
            if other.wraps == cur or res.wraps == other_rid:
                neighbours.add(other_rid)
        for n in neighbours:
            if n in ir_p.resources and n not in dist:
                dist[n] = d + 1
                frontier.append(n)
    return dist


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def analyze(
    ir_b: MethodIR,
    ir_p: MethodIR,
    lineage: Lineage,
    target_b: Set[str],
) -> RelationalResult:
    """Produce one transformation verdict per obligation."""
    result = RelationalResult()
    prop_b = propagated_releases(ir_b)
    prop_p = propagated_releases(ir_p)
    uar_b = {rid for rid, _, _ in use_after_release(ir_b)}
    uar_p = {rid: w for rid, _, w in use_after_release(ir_p)}

    seeds = set()
    for brid in target_b:
        prid = lineage.image(brid)
        if prid:
            seeds.add(prid)
    seeds |= set(lineage.added)
    for lst in lineage.split.values():
        seeds |= set(lst)
    dist = effect_frontier(ir_p, seeds)
    result.frontier = set(dist)

    # Matched obligations: the paired states.
    for brid, prid in lineage.forward.items():
        bres, pres = ir_b.resources[brid], ir_p.resources[prid]
        before = obligation_state(ir_b, brid, prop_b)
        after = obligation_state(ir_p, prid, prop_p)
        # Only an invalidation the patch *introduced* counts against it.
        invalidated = prid in uar_p and brid not in uar_b
        tr = classify(before, after, pres.ownership, invalidated=invalidated)

        if brid in lineage.split:
            tr = Transformation.SPLIT if tr == Transformation.PRESERVE else tr
        if prid in lineage.merged:
            tr = Transformation.MERGE if tr == Transformation.PRESERVE else tr

        result.verdicts.append(ResourceVerdict(
            brid, prid, _label(bres, pres), before, after, tr,
            is_target=brid in target_b,
            ownership=pres.ownership,
            witness=uar_p.get(prid, ""),
            frontier_distance=dist.get(prid, 99),
        ))

    # Obligations the patch removed. A resource that no longer exists carries no
    # obligation, so its disappearance is not itself a conservation violation;
    # ORPHANED is reserved for obligations stranded in M' (below), where some
    # resource is still acquired and nothing can reach it to release it.
    for brid in lineage.dropped:
        bres = ir_b.resources[brid]
        before = obligation_state(ir_b, brid, prop_b)
        tr = (
            Transformation.DISCHARGE
            if before == ObligationState.UNSATISFIED
            else Transformation.PRESERVE
        )
        result.verdicts.append(ResourceVerdict(
            brid, None, _label(bres, None), before, ObligationState.NONE, tr,
            is_target=brid in target_b, ownership=bres.ownership,
            witness=f"{bres.from_call or bres.simple_type} disappeared in M'",
        ))

    # Obligations the patch created.
    for prid in lineage.added:
        pres = ir_p.resources[prid]
        after = obligation_state(ir_p, prid, prop_p)
        invalidated = prid in uar_p
        if invalidated:
            tr = Transformation.INVALIDATE
        elif after == ObligationState.UNSATISFIED:
            # Nothing names this resource, so no statement can ever close it.
            # The obligation is not merely unmet, it is unmeetable, which is the
            # conservation violation the principle is about.
            tr = Transformation.ORPHANED if _unreachable(pres) \
                else Transformation.INTRODUCE
        else:
            tr = Transformation.PRESERVE
        result.verdicts.append(ResourceVerdict(
            None, prid, _label(None, pres), ObligationState.NONE, after, tr,
            ownership=pres.ownership, witness=uar_p.get(prid, ""),
            frontier_distance=dist.get(prid, 99),
        ))

    # Resources introduced as part of a split inherit the origin's target role.
    for origin, members in lineage.split.items():
        for prid in members:
            pres = ir_p.resources[prid]
            after = obligation_state(ir_p, prid, prop_p)
            invalidated = prid in uar_p
            tr = Transformation.INVALIDATE if invalidated else (
                Transformation.INTRODUCE if after == ObligationState.UNSATISFIED
                else Transformation.SPLIT
            )
            result.verdicts.append(ResourceVerdict(
                origin, prid, _label(None, pres), ObligationState.NONE, after, tr,
                is_target=origin in target_b, ownership=pres.ownership,
                witness=uar_p.get(prid, ""), frontier_distance=dist.get(prid, 99),
            ))

    return result


def _unreachable(res: AbstractResource) -> bool:
    """Whether no symbol in the method denotes this resource."""
    return (
        bool(res.symbols)
        and all(s.startswith("<anon@") for s in res.symbols)
        and res.wrapped_by is None
    )


def _label(bres: Optional[AbstractResource], pres: Optional[AbstractResource]) -> str:
    src = pres or bres
    if src is None:
        return "?"
    sym = sorted(src.symbols)[0] if src.symbols else src.rid
    return f"{sym}:{src.simple_type or '?'}"
