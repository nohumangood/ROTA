# Ported APCA signal baselines: feature mapping

These two baselines are **not** reproductions of the published tools. The full
INVALIDATOR (Le-Cong et al., IEEE TSE 2023) needs a passing/failing test suite,
a developer-correct reference program, Daikon invariants, and a trained CodeBERT
fallback; the full ODS (Ye et al., IEEE TSE 2021) trains an XGBoost model on 202
statically extracted AST features. None of these inputs exist in ROTA's
generator-independent, build-free, method-level post-hoc setting, whose only
inputs are `<M, M', kappa, tau>`.

We therefore port only the *decision signals* each method defines that are
computable from `<M, M', kappa, tau>` alone, without training, tests, or a
developer patch. The mapping below is exhaustive; everything in the source
method that depends on an unavailable input is dropped (not approximated).

## INVALIDATOR-style (`invalidator_jleaks.py`)

| INVALIDATOR source signal                         | Ported adaptation on `<M, M', kappa, tau>`                    | Status |
|---------------------------------------------------|---------------------------------------------------------------|--------|
| Semantic rule: patched program violates inferred behavioural invariants | Buggy leaked-resource lifecycle state persists in `M'` (leaked names still unreleased) | ported |
| Syntactic rule: edit matches an overfitting syntactic pattern | Scope error / reassignment / token-Jaccard thresholds on the `M -> M'` edit | ported |
| Daikon invariant inference from a test suite       | —                                                             | dropped (needs tests) |
| Developer-correct reference program                | —                                                             | dropped (needs oracle) |
| CodeBERT logistic-regression fallback classifier   | —                                                             | dropped (needs training) |

## ODS-style (`ods.py`)

| ODS source feature group                           | Ported adaptation on `<M, M', kappa, tau>`                    | Status |
|---------------------------------------------------|---------------------------------------------------------------|--------|
| Code-description features (AST edit counts)        | Release-count change across `M -> M'`                         | ported (subset) |
| Repair-pattern features (added invocation / control-flow edit) | Added `close` invocation; `try`/`finally`/try-with-resources introduction | ported (subset) |
| Contextual features (residual / newly introduced defect) | Residual obligation in `M'`; obligation introduced by the patch | ported (subset) |
| Full 202-feature static AST vector                 | —                                                             | dropped (only lifecycle-relevant subset used) |
| Supervised XGBoost classifier over labelled corpus | Hand-written threshold rule (no training)                     | replaced |

Because both baselines run only a subset of their source method's signals and
use no learned component, they are intentionally conservative. Their numbers in
the paper measure *what the signal families capture on these inputs*, not the
accuracy of the original, fully resourced tools.
