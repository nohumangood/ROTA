"""Invalidator-style baseline adapted to JLeaks method-level patches.

Implements the published semantic + syntactic reasoning at method granularity
(Daikon invariants → cfg_lifecycle obligation states; CodeBERT similarity → token overlap).
"""
from __future__ import annotations

import re
from typing import Dict, Set

from legacy.cfg_lifecycle import detect_scope_errors, extract_rlds, get_code_warnings


def _token_set(code: str) -> Set[str]:
    return set(re.findall(r"\w+", code or ""))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _resource_names(rlds) -> Set[str]:
    return {r.name for r in rlds if not r.is_parameter}


def _leaked_names(rlds) -> Set[str]:
    return {r.name for r in rlds if r.state == "Leaked" and not r.is_parameter}


def _safe_names(rlds) -> Set[str]:
    return {r.name for r in rlds if r.state == "Safe"}


def semantic_overfit(buggy: str, patched: str, fixed: str = "") -> bool:
    """Invalidator semantic checks at method level (M + M' only)."""
    bug_r = extract_rlds(buggy or "")
    pat_r = extract_rlds(patched or "")

    bug_leak = _leaked_names(bug_r)
    pat_leak = _leaked_names(pat_r)

    if bug_leak and (bug_leak & pat_leak):
        return True
    return False


def syntactic_suspicious(buggy: str, patched: str, fixed: str = "") -> bool:
    """Patch retains buggy structure with residual structural warnings."""
    if detect_scope_errors(patched):
        return True
    if any("reassigned" in w for w in get_code_warnings(patched)):
        return True
    pat_leak = _leaked_names(extract_rlds(patched))
    sim_bp = _jaccard(patched, buggy)
    if pat_leak and sim_bp > 0.72:
        return True
    return sim_bp < 0.55 and bool(pat_leak)


def evaluate_invalidator(
    buggy_code: str,
    patched_code: str,
    *,
    fixed_code: str = "",
    key_var: str = "",
) -> Dict:
    sem = semantic_overfit(buggy_code, patched_code, fixed_code)
    syn = syntactic_suspicious(buggy_code, patched_code, fixed_code)
    verdict = "INCORRECT" if (sem or syn) else "CORRECT"
    return {
        "verdict": verdict,
        "pipeline": "invalidator_jleaks",
        "semantic_hit": sem,
        "syntactic_hit": syn,
    }
