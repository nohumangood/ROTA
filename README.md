# ROTA — Replication Package

**Paper:** *ROTA: Automated Correctness Assessment of Resource-Leak Repair
Patches via Relational Obligation-Transformation Analysis* (FSE 2027, under
double-blind review).

This package is anonymized for double-blind review: it contains no author,
institution, or repository-identifying information.

## What ROTA is

ROTA decides whether an automatically generated **resource-leak repair patch**
is correct, by reasoning about the *change* between the buggy method `M` and the
patched method `M'`. A patch is correct when it validly **transforms** the
resource obligations that `M` carried: the reported leak is discharged or legally
handed off, nothing previously valid is invalidated, no unmeetable obligation is
introduced, and no obligation is silently dropped.

ROTA's inference reads only `<M, M', κ, τ>` — the two method versions, the
reported leaked resource `κ`, and its type `τ`. It is:

- **Generator-independent** — it never sees which tool produced the patch, the
  developer fix, labels, or tests.
- **LLM-free** — the analysis and the entire evaluation use no language model.
- **Build-free** — it works on a method in isolation (parsed with tree-sitter),
  with no compilation or whole-program context.
- **Certifying** — every `CORRECT`/`INCORRECT` verdict carries a machine-checkable
  certificate that an independent checker re-derives from the inputs.

The pipeline has four stages (paper Section 3): a per-version resource
**substrate** (`src/rota/ir.py`), a cross-version **lineage** `μ`
(`src/rota/lineage.py`), an obligation-**transformation** domain
(`src/rota/relational.py`), and four obligation-**refinement** conditions that
emit the certificate (`src/rota/checker.py`).

## Layout

```
FSE_2027_ROTA/
├── README.md
├── requirements.txt
├── src/
│   ├── rota/                 # ROTA analysis
│   │   ├── ir.py             #   per-version substrate (resources, ownership, wrappers, CFG)
│   │   ├── lineage.py        #   cross-version resource lineage μ
│   │   ├── relational.py     #   obligation-transformation domain + effect frontier
│   │   ├── checker.py        #   four refinement conditions + certificate (run_rota)
│   │   ├── baselines.py      #   the baseline ladder (state-/change-/transition-aware)
│   │   ├── corpus.py         #   corpus reconstruction from upstream method sources
│   │   └── javasrc.py        #   tree-sitter Java front end
│   ├── baselines/            # ported external-APCA decision signals (Invalidator, Ods)
│   └── legacy/               # single-version lifecycle warnings used by the ported signals
├── data/
│   ├── rota_corpus.json      # 862 reconstructed patches (M, M', κ, τ, label, split, project, tool)
│   └── rota_split.json       # project-level dev / held-out assignment
├── scripts/
│   ├── run_experiments.py    # main tables: ladder, component ablation, input ablation
│   ├── fse_external.py        # ROTA vs ported external-APCA signals
│   ├── fse_analysis.py        # attribution, challenge subsets, transfer, lineage acc., abstention
│   ├── fse_hardlineage.py     # disjoint-symbol (hard rename) lineage subset
│   ├── lineage_hardcase.py    # lineage-accuracy references
│   ├── check_certificate.py   # independent re-validation of every emitted certificate
│   ├── make_split.py          # (re)generate the project-level split
│   └── validate_corpus.py     # corpus sanity checks
└── results/                   # reference outputs this package reproduces
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No API keys, no network, and no external data are required: the reconstructed
corpus (`data/rota_corpus.json`) is self-contained.

## Reproducing the results

All commands are run from the package root and write to `results/`. Nothing calls
the network or a language model.

### Main tables — ladder, ablations, input ablations

```bash
python scripts/run_experiments.py --split heldout     # held-out (61 unseen projects, n=225)
python scripts/run_experiments.py --split dev         # development split
python scripts/run_experiments.py --split all         # whole corpus (n=862)
```

For each configuration this prints coverage, precision, recall, F1, accuracy, a
project-clustered bootstrap F1 interval, and an exact McNemar test against full
ROTA, and writes `results/rota_<split>_{predictions.csv,metrics.json,errors.csv}`.
The held-out run reproduces the headline numbers:

| system (held-out)            | Acc   | F1    |
|------------------------------|-------|-------|
| **ROTA (full)**              | 0.816 | 0.800 |
| patch-only lifecycle         | 0.813 | 0.777 |
| independent-diff             | 0.796 | 0.783 |
| warning-removal              | 0.773 | 0.727 |
| independent-analysis (state) | 0.707 | —     |
| patched-only (state)         | 0.698 | 0.702 |

It also reproduces the component ablation (removing lineage → 0.544, removing
refinement → F1 0.260, removing wrapper propagation → 0.728) and the input
ablation over `<M,M'>`, `+κ`, `+τ`, `+κτ`.

### External-APCA signal comparison

```bash
python scripts/fse_external.py        # -> results/fse_external.json
```

Ports the decision signals of two general-purpose APCA methods to ROTA's inputs
(these are **not** the original tools, which require tests / supervised
training). Reproduces Invalidator 0.622 / Ods 0.596 accuracy vs ROTA 0.816.

### Strengthening analyses

```bash
python scripts/fse_analysis.py        # -> results/fse_analysis.json
```

Reproduces: per-condition attribution (C1/C2/C3/C4 counts, C3 = 0), challenge
subsets (wrapper / multi-resource / rename-rebind / caller-owned / exceptional),
leave-one-generator-out transfer, the κ/τ input anomaly, automatic
lineage accuracy (99.3% on forced correspondences), and the abstention analysis.

### Independent certificate re-validation

```bash
python scripts/check_certificate.py   # re-derives every certificate from <M,M',κ,τ>
```

A separate implementation rebuilds the model, lineage, and transformations and
confirms it reproduces 100% of ROTA's emitted `CORRECT`/`INCORRECT` certificates
(213/213 held-out, 817/817 corpus-wide).

### Lineage on hard renames

```bash
python scripts/fse_hardlineage.py     # -> results/fse_hardlineage.json
python scripts/lineage_hardcase.py    # -> results/lineage_hardcase.json
```

## The corpus

`data/rota_corpus.json` contains 862 reconstructed real repair patches across 183
open-source projects and three repair tools. Each record carries the buggy and
patched method bodies, the reported resource `κ` (`key_var`) and type `τ`
(`resource_type`), the manual binary label (`gt`), a project-level `split`
(`dev` / `heldout`), and provenance (`project`, `tool`, `patch_id`). Labels are
manual; the positive class is **incorrect**. The split is by project, so no
project appears in both halves. `results/` ships the frozen reference outputs for
side-by-side comparison with a fresh run.
