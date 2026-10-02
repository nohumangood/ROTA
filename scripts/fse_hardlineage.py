"""Major-1 experiment: patch-correctness accuracy on HARD-LINEAGE patches.

A hard-lineage patch is one where ROTA's cross-version matching links a resource
whose buggy-side and patched-side symbol sets are DISJOINT (a true rename/rebind):
exactly the case where name- or warning-based identity fails and an explicit
allocation-evidence lineage should matter. We compare ROTA against the transition-
aware and state-based baselines on this subset, held-out and corpus-wide.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rota.checker import run_rota
from rota.baselines import (independent_diff, patch_only_lifecycle,
                            unary_patched_only, warning_removal)
from rota.ir import build_ir
from rota.lineage import compute_lineage

ROOT = os.path.join(os.path.dirname(__file__), "..")
corpus = json.load(open(os.path.join(ROOT, "data", "rota_corpus.json"), encoding="utf-8"))

def is_hard_lineage(p):
    kv, rt = p.get("key_var",""), p.get("resource_type","")
    ib = build_ir(p["buggy_method"], key_var=kv, resource_type=rt)
    ip = build_ir(p["patched_method"], key_var=kv, resource_type=rt)
    if not (ib.parse_ok and ip.parse_ok):
        return False
    lin = compute_lineage(ib, ip, min_score=2.0)
    for brid, prid in lin.forward.items():
        bs = ib.resources[brid].symbols; ps = ip.resources[prid].symbols
        if bs and ps and not (bs & ps):   # matched across a full rename
            return True
    return False

def rota_pred(p):
    c = run_rota(p["buggy_method"], p["patched_method"], key_var=p.get("key_var",""),
                 resource_type=p.get("resource_type","")).certificate
    if c.reason == "UNPARSED": return "correct"   # conservative accept, as scored
    return {"INCORRECT":"incorrect","CORRECT":"correct"}.get(c.verdict,"abstain")

def base_pred(fn, p):
    r = fn(p["buggy_method"], p["patched_method"], key_var=p.get("key_var",""),
           resource_type=p.get("resource_type",""))
    return "incorrect" if r.verdict=="INCORRECT" else "correct"

def acc(rows):
    dec=[(pr,gt) for pr,gt in rows if pr in ("correct","incorrect")]
    if not dec: return (0,0,0.0)
    ok=sum(1 for pr,gt in dec if pr==gt)
    return (ok, len(dec), round(ok/len(dec),4))

def mcnemar(a_rows, b_rows):
    bo=co=0
    for (pa,ga),(pb,gb) in zip(a_rows,b_rows):
        if pa=="abstain" or pb=="abstain": continue
        ra,rb=pa==ga,pb==gb
        if ra and not rb: bo+=1
        elif rb and not ra: co+=1
    n=bo+co
    if n==0: return 1.0
    from math import comb
    k=min(bo,co); tail=sum(comb(n,i) for i in range(k+1))/(2**n)
    return round(min(1.0,2*tail),5)

systems = {"ROTA":rota_pred,
           "independent_diff":lambda p:base_pred(independent_diff,p),
           "patch_only":lambda p:base_pred(patch_only_lifecycle,p),
           "warning_removal":lambda p:base_pred(warning_removal,p),
           "unary_patched_only":lambda p:base_pred(unary_patched_only,p)}

out={}
for name, data in (("heldout",[p for p in corpus if p.get("split")=="heldout"]),
                   ("full", corpus)):
    hard=[p for p in data if is_hard_lineage(p)]
    inc=sum(1 for p in hard if p["gt"]=="incorrect")
    preds={s:[(fn(p),p["gt"]) for p in hard] for s,fn in systems.items()}
    res={"n":len(hard),"pct_incorrect":round(inc/len(hard),3) if hard else 0}
    for s in systems:
        ok,dec,a=acc(preds[s]); res[s]={"acc":a,"decided":dec}
    res["mcnemar_ROTA_vs_idiff"]=mcnemar(preds["ROTA"],preds["independent_diff"])
    res["mcnemar_ROTA_vs_polc"]=mcnemar(preds["ROTA"],preds["patch_only"])
    res["mcnemar_ROTA_vs_wrem"]=mcnemar(preds["ROTA"],preds["warning_removal"])
    out[name]=res

json.dump(out, open(os.path.join(ROOT,"results","fse_hardlineage.json"),"w"), indent=1)
print(json.dumps(out, indent=1))
