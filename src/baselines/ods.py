"""ODS — Obligation Descriptor Score (ASE baseline, rule-based)."""
from __future__ import annotations

import re
from typing import List

from legacy.cfg_lifecycle import compute_delta_l, extract_rlds, rld_to_dict


def compute_ods_features(
    buggy_code: str,
    patched_code: str,
) -> dict:
    orig_rlds = [rld_to_dict(r) for r in extract_rlds(buggy_code or "")]
    patch_rlds = [rld_to_dict(r) for r in extract_rlds(patched_code)]
    orig_objs = extract_rlds(buggy_code or "")
    patch_objs = extract_rlds(patched_code)
    delta_l = compute_delta_l(orig_objs, patch_objs) if buggy_code else {
        "fixed": [], "remaining": [], "new_violations": [], "dependency_issues": [],
    }

    orig_close = len(re.findall(r"\.close\s*\(", buggy_code or ""))
    patch_close = len(re.findall(r"\.close\s*\(", patched_code))
    return {
        "close_diff": patch_close - orig_close,
        "has_finally": int(bool(re.search(r"\bfinally\b", patched_code))),
        "has_twr": int(bool(re.search(r"try\s*\(", patched_code))),
        "n_leaked": sum(1 for r in patch_rlds if r["state"] != "Safe"),
        "n_remaining": len(delta_l.get("remaining", [])),
        "n_new": len(delta_l.get("new_violations", [])),
        "has_dep": int(bool(delta_l.get("dependency_issues"))),
    }


def ods_predict(features: dict) -> str:
    score = 0
    if features["n_leaked"] > 0:
        score += 3
    if features["n_remaining"] > 0:
        score += 3
    if features["n_new"] > 0:
        score += 3
    if features["has_dep"]:
        score += 2
    if features["close_diff"] <= 0:
        score += 1
    if not features["has_finally"] and not features["has_twr"]:
        score += 1
    return "INCORRECT" if score >= 3 else "CORRECT"


def evaluate_ods(buggy_code: str, patched_code: str) -> dict:
    ft = compute_ods_features(buggy_code, patched_code)
    return {
        "verdict": ods_predict(ft),
        "pipeline": "ods",
        "features": ft,
    }
