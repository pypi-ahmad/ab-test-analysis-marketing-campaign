# A/B Test Analysis for a Marketing Campaign Landing Page

**Decision under uncertainty:** Should the company **ship** a new marketing-campaign landing page creative, **hold** the current creative, or **extend** the experiment?

This repository is a **production-minded, fully executed Jupyter analysis** of a real e-commerce A/B test. It is **inferential experiment analysis**, not predictive machine learning: no train/test split, no accuracy score, no AutoML, no LLM stack. The deliverable is a defensible **ship / hold / extend** recommendation behind a **trustworthiness scorecard** (SRM, cleaning, practical significance, power, multi-method confirmation).

| | |
|---|---|
| **Final recommendation (executed run)** | **HOLD** the old campaign creative |
| **Primary p-value (two-sided)** | **0.189883** (fail to reject \(H_0\) at α = 0.05) |
| **Observed lift (new − old)** | **−0.1578 percentage points** |
| **Bootstrap 95% CI (lift)** | **[−0.003973, 0.000769]** |
| **Country-adjusted odds ratio** | **0.9852** |
| **SRM** | p = **0.947**, imbalance **0.0062 pp** (not severe) |
| **Bayesian P(new > old)** | **9.39%** (uniform Beta–Binomial) |
| **Power for a +1 pp absolute MDE** | **~100%** at actual sample size |

> **Production honesty:** “Production-grade” means production **methodology** (Microsoft/DoorDash/Eppo-style gates, Wilson CIs, bootstrap, FDR segments, Bayesian cross-check). It does **not** mean forcing a creative “win.” The data support **HOLD**.

---

## Who this README is for

| Audience | What to read first |
|---|---|
| **Portfolio / technical evaluators** | [Architecture & design decisions](#1-for-portfolio-evaluators), [Evidence from the real run](#evidence-from-the-real-run), [Limitations](#honest-limitations) |
| **Hands-on operators** | [Quick start](#2-for-hands-on-users), [Re-run & troubleshooting](#re-run-the-full-pipeline), [Extending the analysis](#extending-the-local-workflow) |
| **Tutorial learners** | [Concepts before code](#3-for-tutorial-learners), [End-to-end flow](#implementation-flow-end-to-end), [How to read each metric](#how-to-read-the-metrics) |

---

## Project in one page

### Business framing

An organization tested **two creative versions of a marketing campaign landing page**:

| Random assignment (`group`) | Page recorded (`landing_page`) | Interpretation |
|---|---|---|
| `control` | `old_page` | Current campaign creative |
| `treatment` | `new_page` | Proposed new campaign creative |

Each row is a user exposure with a binary outcome `converted ∈ {0, 1}`. Country (`US` / `CA` / `UK`) is available as a covariate.

The only business question that matters at the end:

> Given cleaned data, is there statistically and practically meaningful evidence that the **new** creative improves conversion enough to **ship** it?

### What “success” means here

| Success criterion | How this project meets it |
|---|---|
| Real data, not toy numbers | Public Udacity DAND e-commerce A/B dataset (~294k raw rows) |
| Reproducible environment | `uv` + pinned Python 3.13.13 + lockfile + named Jupyter kernel |
| Honest inference | Pre-declared α = 0.05; two-sided test; no post-hoc α shopping |
| Trustworthiness gates | Schema check, exposure cleaning, **SRM**, practical band, power |
| Multi-method confirmation | z-test + χ² + bootstrap CI + country-adjusted logit + Bayesian P |
| Decision quality | Ship/hold/extend uses significance **and** magnitude **and** power **and** SRM |
| Teachable structure | Progressive markdown (why → how → interpret actual outputs) |

### Explicit non-goals

- Predictive modeling, AutoML, LazyPredict, PyCaret  
- Fine-tuning, Ollama, local LLMs, RAG, agent frameworks  
- Changing α, dropping extra rows, or re-specifying models after seeing an inconvenient p-value  

If you need a classifier for conversion propensity, that is a **different problem** than estimating the effect of a randomized creative.

---

# 1. For portfolio evaluators

This section is written for technical reviewers: architecture choices, reproducibility, evidence, and limitations.

## Architecture & design decisions

### Decision 1 — Inferential pipeline, not an ML training loop

**Choice:** Cleaning → **SRM gate** → EDA (Wilson CIs) → z-test + χ² + bootstrap → logit ORs → segment FDR → power curve → Bayesian cross-check → **decision scorecard**.

**Why:** Randomization + binary conversion is a classical two-sample proportion problem. Production platforms (Microsoft ExP, DoorDash, Eppo, Statsig) gate on **trustworthiness** before celebrating lifts. A predictive model can score individuals without answering “did the page change conversion?”

**Tradeoff:** Logistic regression is **not** used for prediction metrics (no ROC-AUC). Bayesian Beta–Binomial is a **secondary** communication aid, not a replacement for the primary frequentist test.

### Decision 2 — Cleaning is a first-class analysis stage

**Choice:** Explicit **Part 1** for group/page mismatches and duplicate `user_id`s, with before/after counts.

**Why:** This dataset’s known quality issue is pedagogically and methodologically load-bearing. Users with `treatment` + `old_page` (or the reverse) have **untrustworthy exposure**. Including them is not “more data”; it is mislabeled treatment.

**Rule applied:**

1. Keep only `(control, old_page)` and `(treatment, new_page)`.  
2. Then keep **one row per `user_id`** (`keep='first'`).

**Evidence from the real run:**

| Stage | Rows |
|---|---:|
| Raw | 294,478 |
| Mismatched (dropped) | 3,893 (1.32% of raw) |
| After mismatch drop | 290,585 |
| After de-duplication | **290,584** |

### Decision 3 — SRM as a hard trustworthiness gate

**Choice:** χ² goodness-of-fit of observed arm sizes vs intended 50/50; flag **severe** SRM if p is tiny **and** absolute traffic imbalance > 0.5 pp.

**Why:** Sample Ratio Mismatch is one of the most cited production failure modes (Microsoft Research, DoorDash). Analyzing effects under broken randomization yields wrong product decisions.

**Result from the real run:** χ² p ≈ **0.947**, imbalance **0.0062 pp** → not severe; analysis may proceed.

### Decision 4 — Two-sided test + multi-method confirmation at fixed α = 0.05

**Choice:** Primary `proportions_ztest` + `confint_proportions_2indep`; cross-check with `proportions_chisquare`; robustness via **2,000-resample bootstrap** CI; arm CIs use **Wilson** scores.

**Why:** Marketing *wants* improvement, but post-hoc one-sided tests are p-hacking. Bootstrap should agree with analytic CIs at this \(n\) if assumptions are fine.

**Tradeoff:** Slightly less power than a pre-registered one-sided test. Acceptable at ~145k/arm.

### Decision 5 — Logistic regression for OR + country adjustment + FDR segments

**Choice:**

1. `converted ~ ab_page`  
2. `converted ~ ab_page + C(country)`  
3. `converted ~ ab_page * C(country)` (heterogeneity check)  
4. Per-country z-tests with **Benjamini–Hochberg FDR**

**Why:** Odds ratios are interpretable for binary outcomes. Country adjustment checks confounding. Segment peeks without multiplicity control create false discoveries in production.

**Result from the real run:** Unadjusted OR **0.9851**; country-adjusted OR **0.9852** (p ≈ 0.19). No strong interaction; no FDR country discoveries that overturn the overall null.

### Decision 6 — Power + practical significance answer “extend?” without guessing

**Choice:** Practical indifference band **±0.5 pp**; MDE **+1 pp** absolute; 80% power target; **power curve** vs n/arm; Bayesian \(P(p_{\text{new}} > p_{\text{old}})\) as secondary.

**Why:** Non-significance under huge \(n\) is informative. If power for a business-relevant MDE is already ~100%, “run longer” is usually the wrong next step.

**Result from the real run:**

| Question | Number |
|---|---:|
| n/arm for 80% power @ observed effect | 663,574 |
| n/arm for 80% power @ +1 pp MDE | 17,209 |
| Actual min arm n | 145,274 |
| Achieved power @ observed difference | 0.2587 |
| Achieved power @ +1 pp MDE | **1.0000** |

Interpretation: the experiment is **underpowered for a microscopic observed gap** (as expected) but **overpowered for a marketing-relevant +1 pp lift**. Combined with a near-zero / slightly negative estimate, the rational decision is **HOLD**, not EXTEND.

### Decision 7 — Dual notebook representation (jupytext + executed ipynb)

**Choice:** Author as percent-format `.py`, convert to `.ipynb`, execute with `nbconvert` so committed outputs are real.

**Why:**

- `.py` is reviewable in diffs and runnable as a script.  
- Executed `.ipynb` is the portfolio artifact with plots and printed metrics.  
- CI-style re-execution is one command.

### Decision 7 — Flat project layout, uv-only environment

```text
.
├── notebooks/
│   ├── ab_test_marketing_analysis.ipynb   # executed deliverable
│   └── ab_test_marketing_analysis.py      # jupytext source
├── data/
│   ├── ab_data.csv                        # cached real data
│   └── countries.csv
├── pyproject.toml
├── uv.lock
├── .python-version                        # 3.13.13
├── LICENSE                                # MIT (code/docs)
└── README.md
```

**Why uv:** Deterministic installs, lockfile, no bare `pip install`. Kernel name: `ab-test-marketing-project`.

## System architecture (analysis pipeline)

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Data acquisition                            │
│  GitHub raw CSVs  ──(fallback: git clone)──►  data/*.csv        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Part 1 — Cleaning + SRM gate                                   │
│  mismatches → de-dupe → χ² SRM vs 50/50                         │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Part 2 — EDA (OEC = conversion)                                │
│  Wilson CIs · daily trend · traffic · countries                 │
└───────────────┬─────────────────────────────┬───────────────────┘
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│ Part 3 — Primary + robust │   │ Part 4 — Logit + segments       │
│ z-test · χ² · bootstrap   │   │ OR ± country · BH-FDR           │
└───────────────┬───────────┘   └─────────────────┬───────────────┘
                └────────────────┬────────────────┘
                                 ▼
                ┌────────────────────────────────┐
                │ Part 5–6 — Power curve + Bayes │
                │ MDE · practical band · P(win)  │
                └────────────────┬───────────────┘
                                 ▼
                ┌────────────────────────────────┐
                │ Production scorecard → decision│
                └────────────────────────────────┘
```

## Evidence from the real run

**Runtime fingerprint (executed notebook):**

```text
Kernel target name: ab-test-marketing-project
Python:      3.13.13
pandas:      3.0.3
numpy:       2.5.1
scipy:       1.18.0
statsmodels: 0.14.6
matplotlib:  3.11.0
seaborn:     0.13.2
Data load path: raw GitHub URLs (cached under data/)
Notebook QA: 14 code cells · 0 empty outputs · 0 errors · 6 plot outputs
```

### Conversion (cleaned A/B sample)

| Arm | Conversions | n | Rate |
|---|---:|---:|---:|
| Old page (control) | 17,489 | 145,274 | **12.0386%** |
| New page (treatment) | 17,264 | 145,310 | **11.8808%** |
| Difference (new − old) | — | — | **−0.1578 pp** (−1.311% relative) |

Wilson 95% CIs (per arm) are computed in the notebook (production-preferred vs plain normal).

### Trustworthiness — SRM

```text
SRM χ² p = 0.946754
Traffic imbalance = 0.0062 pp
Severe SRM = False  → analysis may proceed
```

### Hypothesis test + robustness

```text
z = -1.310924
p = 0.189883   →  FAIL TO REJECT H0 at α = 0.05
Analytic 95% CI for (p_new − p_old) = [-0.003938, 0.000781]   # includes 0
Bootstrap 95% CI                    = [-0.003973, 0.000769]   # agrees
```

### Logistic odds ratios (`ab_page` = new creative)

| Model | OR | Interpretation |
|---|---:|---|
| Unadjusted | 0.9851 | ~1.5% lower odds of conversion on new page |
| Country-adjusted | 0.9852 (p = 0.191245) | Effect unchanged after country |
| Interaction check | no strong country×page interaction | Use Model 2 as primary adjusted summary |

### Bayesian secondary check

```text
P(p_new > p_old | data) ≈ 9.39%   (uniform Beta–Binomial)
```

### Decision rule used (transparent, pre-structured)

```text
if severe SRM                         → HOLD (do not ship on broken randomization)
elif significant and lift ≥ +0.5 pp   → SHIP
elif significant and not improved     → HOLD
elif underpowered for +1 pp MDE       → EXTEND
else                                  → HOLD
```

**Applied outcome: HOLD.**

Rationale: no significant improvement; point estimate slightly negative; Bayesian P(new better) ≈ 9%; sample already large enough that a +1 pp win would almost certainly have been detected; SRM clean.

### Manager-ready brief (from executed summary cell)

> On **290,584** cleaned users, the new marketing-campaign landing page converted at **11.88%** vs **12.04%** for the old page (**−0.158 pp**). z = **−1.31**, p = **0.190** (not significant). Analytic and bootstrap CIs both cover 0. Country-adjusted OR = **0.985**. SRM not severe. Power for +1 pp ≈ **100%**. Bayesian P(new better) ≈ **9%**. **Decision: HOLD.**

## Honest limitations

1. **Exposure uncertainty for mismatched rows.** 3,893 rows were dropped because assignment and recorded page disagreed. We do not impute the “true” page; that would invent exposure.
2. **Duplicate handling policy.** One user appeared twice; we kept `first`. Alternative policies (`last`, average) could be sensitivity-checked; impact is one row.
3. **Country is not re-randomized.** It is a covariate for adjustment, not a factorial design factor.
4. **No pre-registered MDE in the public dataset narrative.** +1 pp and ±0.5 pp practical band are transparent *business* benchmarks used for this analysis.
5. **Two-sided α fixed at 0.05.** We did not switch to one-sided after seeing the sign of the estimate.
6. **No CUPED** variance reduction — this public log has no pre-period covariate.
7. **Bayesian check uses a weakly informative uniform prior**; different priors shift posterior probabilities slightly but not the ship decision here.
8. **Educational public data.** Results illustrate rigorous process; they are not a claim about a specific live product roadmap.
9. **MIT license covers this repo’s code and docs**, not necessarily the upstream dataset’s original distribution terms.

## Reproducibility checklist (evaluator view)

| Check | Status in this repo |
|---|---|
| Locked dependencies | `uv.lock` |
| Pinned Python | `.python-version` → 3.13.13 |
| Named kernel | `ab-test-marketing-project` |
| Real data download path | Notebook fetches GitHub raw CSVs; caches to `data/` |
| Executed notebook with outputs | `notebooks/ab_test_marketing_analysis.ipynb` |
| One-command re-execution | `jupyter nbconvert --execute ...` |
| Documented decision rule | Printed in notebook + this README |
| Null result reported honestly | Yes (p ≈ 0.19, Bayesian P≈9%, HOLD) |
| SRM / bootstrap / FDR / power curve | Implemented in executed notebook |

---

# 2. For hands-on users

Install, run, troubleshoot, and extend the local workflow.

## Prerequisites

- Linux (tested), macOS should work with the same commands  
- [uv](https://docs.astral.sh/uv/) installed  
- Network access for the first data download (GitHub raw)  
- Optional: JupyterLab / VS Code for interactive exploration  

## Quick start

```bash
cd "/path/to/AB Test Analysis for a Marketing Compaign"

# 1) Python
uv python install 3.13.13
# If 3.13.13 is unresolvable on your machine:
#   uv python install 3.12.10
#   then set requires-python / pin accordingly

# 2) Environment
uv venv --python 3.13.13
uv python pin 3.13.13
uv sync

# 3) Jupyter kernel
uv run python -m ipykernel install --user --name ab-test-marketing-project

# 4) Open the notebook with kernel: ab-test-marketing-project
#    notebooks/ab_test_marketing_analysis.ipynb
```

### Verify the environment

```bash
uv run python -c "
import sys, pandas, numpy, scipy, statsmodels, matplotlib, seaborn
print(sys.version)
print('pandas', pandas.__version__)
print('statsmodels', statsmodels.__version__)
print('scipy', scipy.__version__)
"
```

Expected (as of the verified run):

```text
3.13.13
pandas 3.0.3
statsmodels 0.14.6
scipy 1.18.0
```

## Re-run the full pipeline

### A. Execute the notebook (recommended)

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/ab_test_marketing_analysis.ipynb \
  --output ab_test_marketing_analysis.ipynb \
  --ExecutePreprocessor.kernel_name=ab-test-marketing-project \
  --ExecutePreprocessor.timeout=600
```

### B. Run the jupytext source as a script

```bash
# From project root (so data/ resolves correctly)
uv run python notebooks/ab_test_marketing_analysis.py
```

### C. Edit source → convert → execute

```bash
uv run jupytext --to ipynb notebooks/ab_test_marketing_analysis.py
uv run jupyter nbconvert --to notebook --execute \
  notebooks/ab_test_marketing_analysis.ipynb \
  --output ab_test_marketing_analysis.ipynb \
  --ExecutePreprocessor.kernel_name=ab-test-marketing-project \
  --ExecutePreprocessor.timeout=600
```

### D. Smoke-check outputs after execution

```bash
uv run python - <<'PY'
import nbformat
from pathlib import Path
nb = nbformat.read(Path("notebooks/ab_test_marketing_analysis.ipynb"), as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
empty = [i for i, c in enumerate(code) if not c.get("outputs")]
errs = [
    (i, o.get("ename"), o.get("evalue"))
    for i, c in enumerate(code)
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
print(f"code_cells={len(code)} empty={len(empty)} errors={len(errs)}")
assert not empty and not errs
print("OK")
PY
```

## Data paths

| Source | URL / path |
|---|---|
| Primary | `https://raw.githubusercontent.com/jemc36/Udacity-DAND-AB-test-ecommerce/master/ab_data.csv` |
| Primary | `https://raw.githubusercontent.com/jemc36/Udacity-DAND-AB-test-ecommerce/master/countries.csv` |
| Local cache | `data/ab_data.csv`, `data/countries.csv` |
| Fallback | `git clone https://github.com/jemc36/Udacity-DAND-AB-test-ecommerce` into `data/` |

The notebook prints which load path was used. On the verified run: **raw GitHub URLs**.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `requires-python` / venv mismatch | `uv init` defaulted to a newer Python | Set `requires-python = ">=3.13"`, `uv python pin 3.13.13`, `uv venv --clear --python 3.13.13`, `uv sync` |
| Kernel not found in Jupyter | Kernel not registered | `uv run python -m ipykernel install --user --name ab-test-marketing-project` |
| `nbconvert` fails on kernel name | Kernel missing or name typo | List kernels: `uv run jupyter kernelspec list`; re-register; or omit `kernel_name` and use `uv run python -m jupyter nbconvert ...` |
| URL download 404 / network error | GitHub raw blocked or offline | Notebook falls back to clone; or place CSVs manually in `data/` |
| `display is not defined` | Running outside IPython without fallback | Source already guards with a print fallback; use `uv run` + notebook kernel for full tables |
| Wrong project root / missing `data/` | CWD is not project root or `notebooks/` | Notebook adjusts if `Path.cwd().name == "notebooks"`; prefer execute from repo root via nbconvert |
| Plots missing in re-run | Backend / headless issues | Ensure notebook execution (not bare Agg script without `plt.show` capture); use nbconvert path above |
| Numbers differ slightly | Package version drift | Re-sync from `uv.lock`; compare printed package versions in setup cell |
| Want offline re-run | Network unavailable | Use already-cached `data/*.csv`; if code path always hits URL first, it will still need network unless you short-circuit to local files |

### Offline / local-only load (optional operator tweak)

If you need fully offline execution and `data/*.csv` already exist, you can temporarily force local reads at the top of the acquisition cell:

```python
ab_data = pd.read_csv(DATA_DIR / "ab_data.csv")
countries = pd.read_csv(DATA_DIR / "countries.csv")
```

(Prefer keeping the URL+cache path for reproducibility demos.)

## Extending the local workflow

Practical extensions that stay inside the **inferential** framing:

| Extension | Where to change | Notes |
|---|---|---|
| One-sided test for pre-registered “new > old” | Part 3 `alternative=` | Only if α and direction were fixed **before** looking at data |
| Different MDE (e.g. +0.5 pp) | Part 5 `mde_pp` | Recompute power; may change EXTEND vs HOLD |
| Sensitivity: keep mismatches as intent-to-treat | Part 1 | Compare ITT vs as-treated; document policy |
| Bayesian proportion test | New section | Optional; do not replace frequentist primary without saying so |
| Segment deep-dive (country-only z-tests) | After EDA | Multiple comparisons → correct or pre-register |
| Export manager PDF / HTML | `nbconvert --to html` | Good for stakeholder packs |
| CI job | GitHub Actions + `uv sync` + nbconvert | Fail if any cell errors |

### CLI-oriented operator map

There is no custom CLI package yet; the operator surface is **uv + Jupyter tooling**:

```text
uv sync                              # install
uv run python -m ipykernel ...       # kernel
uv run jupytext --to ipynb ...       # source → notebook
uv run jupyter nbconvert --execute   # full real run
uv run python notebooks/...py        # script path
```

If you add a thin CLI later, keep it thin:

```text
ab-test run      → execute notebook / script
ab-test summary  → print manager metrics from last run
ab-test power --mde 0.01
```

Do **not** bolt on predictive AutoML “for completeness.” It solves a different problem.

## Dependencies (locked via uv)

Core analysis stack:

- `pandas`, `numpy` — data  
- `scipy`, `statsmodels` — tests, CI, logit, power  
- `matplotlib`, `seaborn` — plots  
- `jupyter`, `nbconvert`, `ipykernel`, `jupytext` — execution & authoring  

Install only through:

```bash
uv add <package>   # if you must extend
uv sync            # routine installs
```

Never bare `pip install` into this project.

---

# 3. For tutorial learners

Conceptual backbone and implementation flow. (This project is **A/B testing / inference**, not RAG. The learning object is experiment design + statistics, not retrieval.)

## Concepts before code

### What is an A/B test?

You show two versions of something (here: campaign landing page creatives) to randomly assigned users and compare a metric (here: conversion rate). Randomization is what makes “page caused the change” more credible than “different kinds of users saw different pages.”

### Why not just pick the version with the higher sample rate?

Because of **sampling noise**. With finite users, rates bounce. A 12.04% vs 11.88% gap can appear even when true rates are equal. Hypothesis testing quantifies how surprising the gap is under “no real difference.”

### Null and alternative (this project)

- **\(H_0\)**: new and old creatives convert equally (\(p_{\text{new}} = p_{\text{old}}\)).  
- **\(H_1\)**: they differ (two-sided).  
- **α = 0.05**: if p < 0.05, we call the result statistically significant.

**Important:** “Not significant” ≠ “proven equal.” It means the data are compatible with no difference (or a difference too small to resolve at this α). That is why we also report **confidence intervals**, **effect size**, and **power**.

### Two-proportion z-test (intuition)

For large samples, the difference of two sample proportions is approximately normal. The z-statistic scales that difference by its standard error. Large |z| → small p → harder to attribute the gap to chance alone.

In code (order matters for sign):

```python
count = [conversions_new, conversions_old]
nobs  = [n_new, n_old]
z, p = proportions_ztest(count, nobs, alternative="two-sided")
# CI for p_new - p_old
ci_low, ci_high = confint_proportions_2indep(
    conversions_new, n_new, conversions_old, n_old, compare="diff"
)
```

### Why clean mismatches before testing?

If `group` says treatment but `landing_page` says old page, you do not know what the user saw. Putting those rows into either arm invents a causal story. Dropping them is the standard conservative rule for this dataset.

### Logistic regression in an A/B notebook (without turning it into ML)

Logistic regression models:

\[
\log\frac{P(y=1)}{P(y=0)} = \beta_0 + \beta_1 \cdot \text{ab\_page} + \cdots
\]

- \(\beta_1\) is the **log-odds** shift for the new page.  
- \(\mathrm{OR} = e^{\beta_1}\) is the **odds ratio**.  
- OR = 1 → no change in odds; OR = 0.985 → slightly lower odds on the new page.

We add country dummies to ask: “Does the page effect survive after accounting for market?” That is **adjustment**, not prediction.

### Statistical power (why “extend?” is a math question)

**Power** is the probability of detecting a real effect of a stated size.  

- Tiny true lifts need enormous \(n\).  
- Large \(n\) can still miss a microscopic effect (low power for that tiny effect) while still having high power for a **business-relevant** effect.

This notebook’s observed gap would need ~**663k per arm** for 80% power — a sign the observed effect is tiny. A **+1 pp** absolute lift needed only ~**17k per arm**; we had ~**145k**, so power ≈ **100%** for that MDE.

### Novelty effects and time trends

Sometimes a “new” page wins only on day one (curiosity) and then fades. Plotting **daily conversion by arm** checks for drift or novelty. Stable overlapping curves support a stable null; a growing gap would change the story (and might motivate time-aware analysis).

## Implementation flow (end-to-end)

Walk this in order when studying the notebook:

| Step | Notebook section | You should understand |
|---|---|---|
| 0 | Title / framing | Business decision + why this is not predictive ML |
| 1 | Setup | Versions, kernel, paths |
| 2 | Data acquisition | What each CSV means; load path printout |
| 3 | Part 1 cleaning | Mismatch rule, duplicate rule, before/after n |
| 4 | Part 2 EDA | Rates, CI bars, daily trend, country merge |
| 5 | Part 3 z-test | H0/H1, z, p, CI, plain-language decision |
| 6 | Part 4 logit | OR tables; adjusted vs unadjusted; interaction check |
| 7 | Part 5 power | Observed effect vs MDE; extend or not |
| 8 | Conclusion | Manager summary with **actual** numbers |

### Component map (learner view)

```text
[Randomized experiment log]
        │
        ▼
[Exposure validity check] ── mismatches / dups ──► clean analysis set
        │
        ▼
[Descriptive evidence] ── rates, plots, segments
        │
        ├──────────────► [Primary test] two-proportion z + CI
        │
        ├──────────────► [Effect model] logit ORs ± country
        │
        └──────────────► [Design adequacy] power / required n
                              │
                              ▼
                     [Decision policy] ship / hold / extend
```

## How to read the metrics

| Metric | What it answers | This run |
|---|---|---|
| Conversion rate by arm | How often users convert | 12.04% vs 11.88% |
| Absolute difference (pp) | Business magnitude | −0.16 pp |
| Relative difference (%) | Magnitude vs baseline | −1.31% |
| z / p-value | Compatibility with “no difference” | z=−1.31, p=0.190 |
| 95% CI for difference | Plausible range for true lift | [−0.39 pp, +0.08 pp] |
| Odds ratio | Multiplicative change in odds | ~0.985 |
| Power @ MDE | Could we have seen a real +1 pp win? | ~100% |

### A short interpretation drill

1. Look at the **point estimate** (−0.16 pp): new page is not winning on the mean.  
2. Look at the **CI**: includes 0 → data compatible with no lift (and with a small loss).  
3. Look at **p**: 0.19 → not significant at 0.05.  
4. Look at **power for +1 pp**: already maxed → you are not “missing” a large win due to small n.  
5. Conclude **HOLD**, not “collect more data hoping for significance.”

## Guided study prompts

1. Why is dropping mismatched rows more defensible than guessing the true page?  
2. Why report both p-value **and** CI?  
3. Why is OR ≈ 0.985 consistent with the z-test result?  
4. Why can achieved power for the observed effect be low (~0.26) while power for +1 pp is ~1.0?  
5. When would EXTEND be the right call under the same decision rule?

---

## Artifact map

| Artifact | Role |
|---|---|
| `notebooks/ab_test_marketing_analysis.ipynb` | Primary deliverable (executed, plots, numbers) |
| `notebooks/ab_test_marketing_analysis.py` | Reviewable source (jupytext percent) |
| `data/ab_data.csv` | Cached experiment log (~15.9 MB) |
| `data/countries.csv` | Cached country map (~2.9 MB) |
| `pyproject.toml` / `uv.lock` | Environment contract |
| `.python-version` | Python 3.13.13 pin |
| `LICENSE` | MIT for code and documentation |
| `README.md` | This document |

---

## Dataset citation / source

Public educational A/B dataset commonly used in Udacity’s Data Analyst Nanodegree track, mirrored at:

- Repository: [jemc36/Udacity-DAND-AB-test-ecommerce](https://github.com/jemc36/Udacity-DAND-AB-test-ecommerce)

Use of the data remains subject to the source’s terms. This repository’s **MIT license applies to the analysis code and documentation**, not automatically to third-party data.

---

## Contributing & community

- [CONTRIBUTING.md](CONTRIBUTING.md) — setup, run checks, PR expectations  
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Contributor Covenant  
- [Issue templates](.github/ISSUE_TEMPLATE/) — bug report, feature request, question  

Please open issues with the appropriate template. For security-sensitive reports, contact the owner privately rather than filing a public issue.

## License

[MIT](LICENSE) © 2026 Ahmad — for this project’s code and docs.

---

## Method stack (production checklist)

| Stage | Technique | Library / approach |
|---|---|---|
| Schema gate | Column/value assertions | pandas |
| Exposure integrity | Drop group≠page; de-dupe `user_id` | pandas |
| SRM | χ² GOF vs 50/50 | `scipy.stats.chisquare` |
| Arm uncertainty | Wilson score CI | `proportion_confint(..., method="wilson")` |
| Primary effect | Two-proportion z-test + CI | `proportions_ztest`, `confint_proportions_2indep` |
| Cross-check | Proportions χ² | `proportions_chisquare` |
| Robustness | Bootstrap percentile CI | NumPy RNG (seeded) |
| Adjusted effect | Logit odds ratios ± country | `statsmodels.formula.api.logit` |
| Segments | Per-country z + BH-FDR | `multipletests` |
| Design adequacy | Power @ MDE + power curve | `NormalIndPower`, `power_proportions_2indep` |
| Secondary | Beta–Binomial P(new > old) | `scipy.stats.beta` |
| Decision | Scorecard → SHIP / HOLD / EXTEND | explicit rules |

### Industry references informing this design

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments*
- Microsoft Research — diagnosing Sample Ratio Mismatch  
- DoorDash / Eppo / Statsig engineering posts on SRM as a release gate  
- statsmodels proportion APIs  

## Bottom line

This project demonstrates a **production-minded experiment analysis** workflow: clean messy assignments, pass an **SRM** gate, estimate effects with multi-method confirmation, adjust with covariates without sliding into predictive ML, and turn numbers into a **HOLD** decision with power- and probability-backed reasoning. The null result is not a failure of the analysis — it is the analysis doing its job.
