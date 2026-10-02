"""Validate cross-version lineage on the HARD correspondences.

The easy oracle in fse_analysis.py scores lineage only on correspondences whose
identity is already obvious (a single obligation per side, or a preserved unique
variable name). This script does the opposite: it measures lineage exactly where
it is supposed to earn its keep -- renamed/rebound resources and wrapper
split/merge -- against a reference that does NOT use ROTA's weighted scorer.

Independent reference.
  We extract each obligation-carrying resource's allocation expression directly
  from the method source (the balanced `new T(...)` text at its allocation line),
  normalise it, and match buggy to patched resources by textual similarity of
  those expressions (difflib), ignoring variable names entirely. This is
  independent of ROTA, whose dominant cue is a shared variable symbol. The
  reference also reads a wrapper split/merge straight off the text: a buggy
  allocation that appears strictly nested inside a patched allocation is a split
  (the patch wrapped it); the mirror case is a merge.

Hard subset.
  A correspondence is HARD when the reference match pairs resources whose
  variable names differ (a rename/rebind), or when it is a split or a merge.
  These are precisely the cases the easy oracle skipped. We report how often
  ROTA's lineage agrees with the reference on this subset, plus split and merge
  detection.
"""
from __future__ import annotations

import collections
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rota.ir import build_ir  # noqa: E402
from rota.lineage import compute_lineage  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "data", "rota_corpus.json")

_NEW = re.compile(r"\bnew\s+[A-Za-z_][\w.]*\s*\(")


def _alloc_expr(text, line):
    """Balanced `new T(...)` expression whose `new` sits on 1-indexed `line`."""
    lines = text.split("\n")
    if not (1 <= line <= len(lines)):
        return ""
    # join from the alloc line onward so multi-line allocations are captured
    blob = "\n".join(lines[line - 1: line + 4])
    m = _NEW.search(blob)
    if not m:
        return ""
    i = blob.find("(", m.start())
    depth, j = 0, i
    while j < len(blob):
        if blob[j] == "(":
            depth += 1
        elif blob[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return re.sub(r"\s+", "", blob[m.start(): j + 1])


def _norm(expr):
    # drop variable-name noise, keep constructor types and structure
    return expr


def _obl(ir, text):
    out = {}
    for rid, r in ir.resources.items():
        if not r.carries_obligation:
            continue
        e = _alloc_expr(text, r.alloc_line)
        if e:
            out[rid] = (e, frozenset(r.symbols))
    return out


def _ref_match(ob, op):
    """Reference buggy->patched match, split, merge from allocation text only.

    ``match`` keeps only UNAMBIGUOUS correspondences: a buggy allocation whose
    single best patched allocation is a clear winner (similarity >=0.88 and at
    least 0.1 above the runner-up). This avoids penalising ROTA on genuine ties
    (several identical allocations), where no matcher has a defensible choice.
    """
    match, split, merge = {}, {}, {}
    for brid, (be, _) in ob.items():
        scored = sorted(
            ((difflib.SequenceMatcher(None, be, pe).ratio(), prid)
             for prid, (pe, _) in op.items()), reverse=True)
        if scored and scored[0][0] >= 0.88 and (
                len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.10):
            match[brid] = scored[0][1]
    # split: buggy alloc strictly nested inside exactly one patched alloc
    for brid, (be, _) in ob.items():
        hosts = [prid for prid, (pe, _) in op.items()
                 if be != pe and be in pe and len(be) < len(pe)]
        if len(hosts) == 1:
            split[brid] = hosts
    # merge: patched alloc strictly nested inside exactly one buggy alloc
    for prid, (pe, _) in op.items():
        hosts = [brid for brid, (be, _) in ob.items()
                 if pe != be and pe in be and len(pe) < len(be)]
        if len(hosts) == 1:
            merge[prid] = hosts
    return match, split, merge


def main():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    # counters over HARD reference correspondences
    rename_tot = rename_hit = 0
    split_tot = split_hit = 0
    merge_tot = merge_hit = 0
    match_tot = match_hit = 0            # all hard matches (rename-only)
    examples = []

    for p in corpus:
        kv, rt = p.get("key_var", ""), p.get("resource_type", "")
        tb, tp = p["buggy_method"], p["patched_method"]
        ir_b = build_ir(tb, key_var=kv, resource_type=rt)
        ir_p = build_ir(tp, key_var=kv, resource_type=rt)
        if not (ir_b.parse_ok and ir_p.parse_ok):
            continue
        ob, op = _obl(ir_b, tb), _obl(ir_p, tp)
        if not ob or not op:
            continue
        ref_m, ref_s, ref_merge = _ref_match(ob, op)
        lin = compute_lineage(ir_b, ir_p)

        # rename/rebind: reference matches a pair whose variable names are disjoint
        for brid, prid in ref_m.items():
            bs, ps = ob[brid][1], op[prid][1]
            if bs and ps and not (bs & ps):      # names differ -> HARD
                rename_tot += 1
                match_tot += 1
                ok = lin.forward.get(brid) == prid
                rename_hit += int(ok)
                match_hit += int(ok)
                if len(examples) < 12:
                    examples.append(("rename", p["patch_id"], ob[brid][0],
                                     op[prid][0], ok))
        # split
        for brid, prids in ref_s.items():
            split_tot += 1
            got = lin.split.get(brid, [])
            # ROTA split OR a forward match into one of the wrapper resources
            ok = bool(got) or lin.forward.get(brid) in prids
            split_hit += int(ok)
            if len(examples) < 12:
                examples.append(("split", p["patch_id"], ob[brid][0],
                                 "|".join(op[x][0] for x in prids), ok))
        # merge
        for prid, brids in ref_merge.items():
            merge_tot += 1
            got = lin.merged.get(prid, [])
            ok = bool(got) or any(lin.forward.get(b) == prid for b in brids)
            merge_hit += int(ok)

    def rate(h, t):
        return round(h / t, 4) if t else 0.0

    res = {
        "hard_matches_rename_rebind": {"n": rename_tot, "agree": rename_hit,
                                       "accuracy": rate(rename_hit, rename_tot)},
        "split": {"n": split_tot, "agree": split_hit,
                  "accuracy": rate(split_hit, split_tot)},
        "merge": {"n": merge_tot, "agree": merge_hit,
                  "accuracy": rate(merge_hit, merge_tot)},
        "all_hard": {"n": match_tot + split_tot + merge_tot,
                     "agree": match_hit + split_hit + merge_hit,
                     "accuracy": rate(match_hit + split_hit + merge_hit,
                                      match_tot + split_tot + merge_tot)},
    }
    out = os.path.join(ROOT, "results", "lineage_hardcase.json")
    json.dump({"summary": res, "examples": examples},
              open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    print("\nexamples (relation, patch, buggy-alloc, patched-alloc, ROTA-correct):")
    for e in examples:
        print("  ", e[0], e[1], "|", e[2][:34], "->", e[3][:34], "|", e[4])
    print("\nwrote", out)


if __name__ == "__main__":
    main()
