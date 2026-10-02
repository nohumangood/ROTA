"""Cross-version resource lineage: which resource in M' *is* which resource in M.

This is the first step that has no counterpart in single-version analysis. A
patch is free to rename a variable, replace an allocation with a wrapper, move
an acquisition into a try-with-resources header, split one resource into two, or
merge two into one. None of those are identity changes at the obligation level,
yet all of them break variable-name matching.

Lineage therefore matches *abstract resource objects* across versions using
allocation evidence rather than syntax: the producing expression, the declared
type, the symbols the resource is reachable through, the wrapper chain it sits
in, and its position in the acquisition order. The resulting map mu is what lets
:mod:`rota.relational` talk about one obligation having two states.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rota.ir import AbstractResource, MethodIR


@dataclass
class Lineage:
    """The map mu between buggy-side and patched-side resources."""

    forward: Dict[str, str] = field(default_factory=dict)    # buggy rid -> patch rid
    backward: Dict[str, str] = field(default_factory=dict)   # patch rid -> buggy rid
    scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    #: Buggy resources with no counterpart; candidates for an orphaned obligation.
    dropped: List[str] = field(default_factory=list)
    #: Patched resources with no antecedent; candidates for an introduced one.
    added: List[str] = field(default_factory=list)
    #: buggy rid -> several patch rids (one obligation split in two).
    split: Dict[str, List[str]] = field(default_factory=dict)
    #: several buggy rids -> one patch rid (obligations merged under one owner).
    merged: Dict[str, List[str]] = field(default_factory=dict)

    def image(self, rid: str) -> Optional[str]:
        return self.forward.get(rid)


def _type_affinity(a, b) -> float:
    ta = a if isinstance(a, str) else a.simple_type
    tb = b if isinstance(b, str) else b.simple_type
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    # A patch that wraps a stream keeps the family even though the type changes.
    for suffix in ("InputStream", "OutputStream", "Reader", "Writer",
                   "Connection", "Statement", "ResultSet", "Socket", "Channel"):
        if ta.endswith(suffix) and tb.endswith(suffix):
            return 0.5
    return 0.0


_ORIGIN = re.compile(r"new\s+([\w.]+)|^([\w]+)$")


def _origin(res: AbstractResource) -> str:
    """Allocation origin of a resource, normalised across spellings.

    The same allocation is recorded as ``new FileInputStream`` when a
    declaration names it and as the full expression text when it is inlined, so
    comparing the raw strings would miss the correspondence.
    """
    m = _ORIGIN.search(res.from_call or "")
    if m:
        return (m.group(1) or m.group(2) or "").split(".")[-1]
    return (res.from_call or "").strip()


def _score(a: AbstractResource, b: AbstractResource, order_a: int, order_b: int) -> float:
    """Evidence that patched resource ``b`` continues buggy resource ``a``."""
    score = 0.0
    if a.symbols & b.symbols:
        score += 3.0
    oa, ob = _origin(a), _origin(b)
    if oa and oa == ob:
        score += 3.0
    score += 2.0 * _type_affinity(a, b)
    if a.is_parameter and b.is_parameter:
        score += 4.0 if a.symbols & b.symbols else 0.0
    elif a.is_parameter != b.is_parameter:
        score -= 4.0
    # Acquisition order is weak evidence, used only to break ties.
    score += 0.5 * (1.0 / (1.0 + abs(order_a - order_b)))
    return score


def compute_lineage(ir_b: MethodIR, ir_p: MethodIR, *, min_score: float = 2.0) -> Lineage:
    """Match resources across versions by allocation evidence, greedily."""
    lin = Lineage()
    b_items = list(ir_b.resources.items())
    p_items = list(ir_p.resources.items())
    order_b = {rid: i for i, (rid, _) in enumerate(b_items)}
    order_p = {rid: i for i, (rid, _) in enumerate(p_items)}

    pairs: List[Tuple[float, str, str]] = []
    for brid, bres in b_items:
        for prid, pres in p_items:
            s = _score(bres, pres, order_b[brid], order_p[prid])
            lin.scores[(brid, prid)] = s
            if s >= min_score:
                pairs.append((s, brid, prid))
    pairs.sort(key=lambda t: -t[0])

    used_b: Set[str] = set()
    used_p: Set[str] = set()
    for s, brid, prid in pairs:
        if brid in used_b or prid in used_p:
            continue
        lin.forward[brid] = prid
        lin.backward[prid] = brid
        used_b.add(brid)
        used_p.add(prid)

    lin.dropped = [rid for rid, _ in b_items if rid not in used_b]
    lin.added = [rid for rid, _ in p_items if rid not in used_p]

    _detect_split_merge(ir_b, ir_p, lin, pairs)
    return lin


def _detect_split_merge(ir_b: MethodIR, ir_p: MethodIR, lin: Lineage, pairs) -> None:
    """Attribute unmatched resources to a split or a merge where the wrapper
    chain explains them.

    A patch that replaces ``new FileInputStream(f)`` with a buffered wrapper adds
    a resource whose obligation is not new: it is the same obligation, now owned
    one level up. Recording that as SPLIT rather than INTRODUCE is what keeps
    ROTA from reporting a correct wrapper-introducing repair as a regression.
    """
    for prid in list(lin.added):
        pres = ir_p.resources[prid]
        if pres.wraps and pres.wraps in lin.backward:
            origin = lin.backward[pres.wraps]
            lin.split.setdefault(origin, []).append(prid)
            lin.added.remove(prid)

    for brid in list(lin.dropped):
        bres = ir_b.resources[brid]
        # The delegate disappeared but its wrapper survived: obligations merged.
        if bres.wrapped_by and bres.wrapped_by in lin.forward:
            lin.merged.setdefault(lin.forward[bres.wrapped_by], []).append(brid)
            lin.dropped.remove(brid)


def target_resources(ir: MethodIR, key_var: str, resource_type: str = "") -> List[str]:
    """Resources denoted by the reported leaked variable kappa.

    Falls back through progressively weaker evidence. The ordering matters: a
    wider fallback holds more of the defect but also charges the patch for
    obligations the report never mentioned.
    """
    syms = _symbols(key_var)
    out = [rid for rid, res in ir.resources.items() if res.symbols & syms]

    # The reported defect is a *kind* of resource, not only the one variable the
    # report happened to name. When tau is a specific resource type, every
    # obligation-carrying resource of that type is part of the same defect, so a
    # patch that discharges one named instance while a sibling of the same type
    # leaks is a partial fix - the dominant incorrect class.
    tau = (resource_type or "").split(".")[-1].split("<")[0]
    if tau:
        typed = [
            rid for rid, r in ir.resources.items()
            if r.carries_obligation and _type_affinity(r.simple_type, tau) > 0
        ]
        out = list(dict.fromkeys(out + typed))
    if out:
        return out

    return [rid for rid, r in ir.resources.items() if r.carries_obligation]


def _symbols(key_var: str) -> Set[str]:
    import re

    return {
        m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\b", key_var or "")
        if m.group(1) not in ("new", "null", "return", "try", "finally", "this")
    }
