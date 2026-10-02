"""Rebuild a parseable evaluation corpus from upstream JLeaks plus ARPA patches.

The ARPA artifact stores each program version as a damaged source-file window,
which cannot support an AST/CFG-based analysis. Upstream JLeaks ships the clean
defect method for essentially every bug, but it does not ship the repair-tool
patches. This module recovers a clean patched method by lifting the edit out of
the damaged window pair and replaying it against the clean upstream method.

Recovery proceeds as a cascade, strongest evidence first:

``direct``
    The patched window already contains a brace-complete method whose name
    matches the upstream defect method.
``scoped_replay``
    The clean method is aligned into the buggy window, the window-level edit is
    projected onto that aligned region, and the surviving hunks are replayed.
``repaired_replay``
    As above, but the replayed text needed brace repair before it parsed.

Anything that still fails to parse is reported as uncovered rather than guessed
at, so coverage stays an explicit, auditable number.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rota.javasrc import (
    JavaMethod,
    blank_noncode,
    extract_methods,
    parse_method,
    select_target_method,
)


@dataclass
class RecoveredPair:
    """A buggy/patched method pair that both parse, plus provenance."""

    buggy: str
    patched: str
    strategy: str
    method_name: str = ""
    hunks_applied: int = 0
    hunks_dropped: int = 0
    notes: List[str] = field(default_factory=list)


def _norm(line: str) -> str:
    return " ".join(line.split())


def _brace_delta(src: str) -> int:
    b = blank_noncode(src)
    return b.count("{") - b.count("}")


def _repair_braces(src: str) -> Optional[str]:
    """Close or open the minimum number of braces to balance a method body."""
    d = _brace_delta(src)
    if d == 0:
        return None
    if d > 0:
        return src.rstrip() + "\n" + "}\n" * d
    # Too many closers: drop trailing closers rather than inventing openers.
    out = src.rstrip()
    for _ in range(-d):
        idx = out.rfind("}")
        if idx < 0:
            return None
        out = out[:idx] + out[idx + 1:]
        out = out.rstrip()
    return out


def _index_map(src_lines: List[str], dst_lines: List[str]) -> Dict[int, int]:
    """Map dst line indices onto src line indices via matching blocks."""
    sm = difflib.SequenceMatcher(
        None, [_norm(x) for x in src_lines], [_norm(x) for x in dst_lines], autojunk=False
    )
    out: Dict[int, int] = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            out[b + k] = a + k
    return out


def _span_extract(clean_buggy: str, dam_patched: str) -> Optional[Tuple[str, float]]:
    """Locate the patched counterpart of the clean method inside the window.

    Rather than diffing the two damaged windows against each other - they are
    independently extracted and frequently misaligned - this anchors the clean
    method directly in the patched window. Lines the repair tool inserted appear
    as gaps between matching blocks and are carried along, so the extracted span
    is the patched method including its edit.

    Returns ``(text, coverage)`` where coverage is the fraction of clean-method
    lines that anchored, which gauges how trustworthy the span is.
    """
    clean_lines = clean_buggy.splitlines()
    dam_lines = dam_patched.splitlines()
    if not clean_lines or not dam_lines:
        return None

    sm = difflib.SequenceMatcher(
        None,
        [_norm(x) for x in clean_lines],
        [_norm(x) for x in dam_lines],
        autojunk=False,
    )
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    anchored = sum(b.size for b in blocks)
    coverage = anchored / len(clean_lines)

    lo = blocks[0].b
    hi = blocks[-1].b + blocks[-1].size

    # Extend forward until the span's braces balance, so the method closes.
    span = "\n".join(dam_lines[lo:hi])
    guard = 0
    while _brace_delta(span) > 0 and hi < len(dam_lines) and guard < 200:
        hi += 1
        span = "\n".join(dam_lines[lo:hi])
        guard += 1
    # Extend backward if the signature line was cut off above the first anchor.
    guard = 0
    while _brace_delta(span) < 0 and lo > 0 and guard < 200:
        lo -= 1
        span = "\n".join(dam_lines[lo:hi])
        guard += 1

    return span, coverage


def _target_name(defect_method: str) -> str:
    """Method name from a JLeaks 'path/File.java:name' defect-method spec."""
    return (defect_method or "").split(":")[-1].strip()


def _pick(methods: List[JavaMethod], name: str, key_var: str, rtype: str) -> Optional[JavaMethod]:
    """Prefer the declared defect method, else fall back to resource scoring."""
    if name:
        named = [m for m in methods if m.name == name]
        if len(named) == 1:
            return named[0]
        if named:
            return select_target_method(named, key_var, rtype) or named[0]
    return select_target_method(methods, key_var, rtype)


def _scoped_replay(
    clean_buggy: str,
    dam_buggy: str,
    dam_patched: str,
) -> Tuple[Optional[str], int, int]:
    """Project the window-level edit onto the clean method and replay it.

    Returns ``(patched_text, applied, dropped)``. Hunks that do not land inside
    the clean method are dropped: they edit a neighbouring method that the
    window happened to include, and are irrelevant to this bug.
    """
    clean_lines = clean_buggy.splitlines()
    dam_b_lines = dam_buggy.splitlines()
    dam_p_lines = dam_patched.splitlines()
    if not clean_lines or not dam_b_lines:
        return None, 0, 0

    # dam_buggy line -> clean_buggy line, so edits can be re-expressed on the
    # clean method even though the window starts at a different offset.
    to_clean = _index_map(clean_lines, dam_b_lines)

    sm = difflib.SequenceMatcher(
        None,
        [_norm(x) for x in dam_b_lines],
        [_norm(x) for x in dam_p_lines],
        autojunk=False,
    )
    hunks = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not hunks:
        return clean_buggy, 0, 0

    applied: List[Tuple[int, int, List[str]]] = []
    dropped = 0
    for tag, i1, i2, j1, j2 in hunks:
        new = dam_p_lines[j1:j2]
        if i1 == i2:  # pure insertion: anchor on the preceding kept line
            anchor = to_clean.get(i1 - 1)
            if anchor is None:
                dropped += 1
                continue
            applied.append((anchor + 1, anchor + 1, new))
            continue
        lo = to_clean.get(i1)
        hi = to_clean.get(i2 - 1)
        if lo is None or hi is None or hi < lo:
            dropped += 1
            continue
        applied.append((lo, hi + 1, new))

    if not applied:
        return None, 0, dropped

    out = list(clean_lines)
    for lo, hi, new in sorted(applied, key=lambda t: t[0], reverse=True):
        out[lo:hi] = new
    return "\n".join(out), len(applied), dropped


def _edit_signature(before: str, after: str) -> Tuple[set, set]:
    """Normalised (added, removed) line sets describing an edit."""
    a = [_norm(x) for x in (before or "").splitlines() if x.strip()]
    b = [_norm(x) for x in (after or "").splitlines() if x.strip()]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    added, removed = set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            added |= set(b[j1:j2])
        if tag in ("replace", "delete"):
            removed |= set(a[i1:i2])
    return added, removed


def _lines(code: str) -> set:
    return {_norm(x) for x in (code or "").splitlines() if x.strip()}


def _fidelity(clean_buggy: str, candidate: str, window_sig) -> Tuple[float, int]:
    """Score a reconstruction by whether it realises the window's edit.

    Scoring deliberately avoids diffing the reconstruction against the clean
    method. Upstream JLeaks and the ARPA window hold the same method at
    different formatting, and those pre-existing cosmetic differences would be
    counted as edits and swamp the signal.

    Instead the window's edit is checked by containment: lines the repair added
    should appear in the reconstruction, and lines it deleted should be gone.
    Window-edit lines belonging to neighbouring methods are filtered out first,
    since they are out of scope by construction.

    Returns ``(fidelity, n_checks)``; ``n_checks`` is how much evidence the
    score rests on, used to break ties.
    """
    w_add, w_rem = window_sig
    buggy_lines, cand_lines = _lines(clean_buggy), _lines(candidate)

    # Additions genuinely new to this method, and deletions that really hit it.
    rel_add = {x for x in w_add if x and x not in buggy_lines}
    rel_rem = {x for x in w_rem if x and x in buggy_lines}

    scores, checks = [], 0
    if rel_add:
        scores.append(len(rel_add & cand_lines) / len(rel_add))
        checks += len(rel_add)
    if rel_rem:
        scores.append(len(rel_rem - cand_lines) / len(rel_rem))
        checks += len(rel_rem)
    if not scores:
        return 1.0, 0
    return sum(scores) / len(scores), checks


def recover_pair(
    *,
    clean_buggy: str,
    dam_buggy: str,
    dam_patched: str,
    defect_method: str = "",
    key_var: str = "",
    resource_type: str = "",
    min_fidelity: float = 1.0,
) -> Optional[RecoveredPair]:
    """Recover a parseable (M, M') method pair, or None if coverage fails.

    Every strategy proposes a candidate; the candidate whose edit best matches
    the edit recorded in the raw windows wins. Scoring rather than short-
    circuiting matters because the highest-precision strategy is not reliably
    the most faithful: a patched window often also contains an *unmodified*
    copy of the target method, which a first-match cascade would happily accept
    and thereby silently discard the repair.
    """
    name = _target_name(defect_method)
    buggy_sel = _pick(extract_methods(clean_buggy), name, key_var, resource_type)
    if buggy_sel is None:
        return None
    buggy_text = buggy_sel.text
    window_sig = _edit_signature(dam_buggy, dam_patched)

    candidates: List[Tuple[str, str, int, int, List[str]]] = []

    direct = _pick(extract_methods(dam_patched), buggy_sel.name, key_var, resource_type)
    if direct is not None and direct.name == buggy_sel.name:
        candidates.append(("direct", direct.text, 0, 0, []))

    span = _span_extract(buggy_text, dam_patched)
    if span is not None and span[1] >= 0.30:
        text, coverage = span
        for variant in (text, _repair_braces(text)):
            if not variant:
                continue
            sel = _pick(extract_methods(variant), buggy_sel.name, key_var, resource_type)
            picked = sel.text if sel is not None else (
                variant if parse_method(variant)[0] is not None else None
            )
            if picked:
                candidates.append(
                    ("span_anchor", picked, 0, 0, [f"anchor-coverage={coverage:.2f}"])
                )
                break

    replayed, applied, droppedn = _scoped_replay(buggy_text, dam_buggy, dam_patched)
    if replayed is not None:
        for variant, tag in ((replayed, "scoped_replay"), (_repair_braces(replayed), "repaired_replay")):
            if not variant:
                continue
            sel = _pick(extract_methods(variant), buggy_sel.name, key_var, resource_type)
            if sel is not None:
                candidates.append((tag, sel.text, applied, droppedn, []))
                break

    if not candidates:
        return None

    scored = []
    for strat, text, ap, dr, notes in candidates:
        fid, checks = _fidelity(buggy_text, text, window_sig)
        if fid < min_fidelity:
            continue
        # A candidate the analysis cannot parse is worthless downstream, so a
        # parseable rival wins even at equal fidelity.
        usable = 1 if _is_whole_method(text) else 0
        scored.append((usable, fid, checks, strat, text, ap, dr, notes))

    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    _, *rest = scored[0]
    scored = [tuple(rest)]
    fid, checks, strat, text, ap, dr, notes = scored[0]
    return RecoveredPair(
        buggy_text, text, strat, buggy_sel.name, ap, dr,
        notes + [f"edit-fidelity={fid:.2f}", f"checks={checks}"],
    )


def _is_whole_method(text: str) -> bool:
    """Whether a candidate parses as one complete method declaration."""
    tree, node = parse_method(text)
    return tree is not None and node is not None


_PROSE = re.compile(r"```|^\s*The provided", re.MULTILINE)


def is_format_invalid(code: str) -> bool:
    """Non-code generator output: prose or a markdown fence rather than Java."""
    return bool(_PROSE.search(code or ""))
