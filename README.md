# A/B Test Analysis for a Marketing Campaign Landing Page

**Decision under uncertainty:** Should the company **ship** a new marketing-campaign landing page creative, **hold** the current creative, or **extend** the experiment?

This repository is a **production-minded, fully executed Jupyter analysis** of a real e-commerce A/B test. It is **inferential experiment analysis**, not predictive machine learning: no train/test split, no accuracy score, no AutoML, no LLM stack. The deliverable is a defensible **ship / hold / extend** recommendation behind a **trustworthiness scorecard** (SRM, cleaning, practical significance, power, multi-method confirmation).

| | |
|---|---|
| **Current recommendation (v3 advanced + v2 notebook)** | **HOLD** the old campaign creative |
| **Primary p-value (two-sided)** | **0.189883** (fail to reject \(H_0\) at α = 0.05) |
| **Observed lift (new − old)** | **−0.1578 percentage points** |
| **Bootstrap 95% CI (lift)** | **[−0.003973, 0.000769]** *(since v2)* |
| **Country-adjusted odds ratio** | **0.9852** |
| **SRM** | p = **0.946754**, imbalance **0.0062 pp** (not severe) *(since v2)* |
| **TOST equivalent (±0.5 pp)** | **True** *(new in v3)* |
| **ITT vs as-treated agreement** | **YES** (both HOLD) *(new in v3)* |
| **Bayesian P(new > old)** | **~9.4–9.6%** (uniform Beta–Binomial) |
| **Power for a +1 pp absolute MDE** | **~100%** at actual sample size |
| **Simulation: true +1 pp / +2 pp** | Pipeline recovers **SHIP** *(new in v3)* |

> **What “better results” means here:** We improved **decision quality and trustworthiness**, not conversion. The creative still does not win.  
> - **v1 → v2:** SRM, Wilson CIs, bootstrap, χ², FDR, power curve, Bayesian, scorecard narrative ([comparison](#baseline-v1-vs-production-v2--outputs--techniques)).  
> - **v2 → v3:** TOST equivalence, ITT sensitivity, simulation harness, CUPED teaching demo, CLI + `metrics.json` + HTML ([v3 section](#advanced-v3--package-cli-tutorials)).  
> Baseline notebook files are **kept unchanged for learning**.

---

## Who this README is for

| Audience | What to read first |
|---|---|
| **Portfolio / technical evaluators** | [v1 vs v2](#baseline-v1-vs-production-v2--outputs--techniques), [v3 advanced](#advanced-v3--package-cli-tutorials), [Architecture](#1-for-portfolio-evaluators) |
| **Hands-on operators** | [Quick start](#2-for-hands-on-users), [CLI](#cli-advanced-v3), [Re-run](#re-run-the-full-pipeline) |
| **Tutorial learners** | [tutorials/](tutorials/README.md), baseline notebook, [advanced notebook](notebooks/02_advanced_production_upgrades.ipynb) |

---

## Baseline (v1) vs production (v2) — outputs & techniques

This section **keeps the original baseline numbers** and compares them to the current production-grade notebook run. Same dataset, same cleaning rules, same α = 0.05 — upgraded **methods and decision gates**.

### How “better” is defined (and what it is not)

| Dimension | Baseline v1 | Production v2 | Better? |
|---|---|---|---|
| Creative conversion “win” | New page slightly worse / null | Same data → still null / slightly worse | **Not a conversion win** (honest) |
| Trust that randomization worked | Assumed after cleaning | **Measured via SRM** | Yes — experiment integrity |
| Uncertainty on the lift | Analytic CI only | Analytic CI **+ bootstrap** | Yes — robustness |
| Arm rate intervals | Normal approx | **Wilson score** | Yes — better proportion CI practice |
| Effect confirmation | z-test + logit | z-test + **χ²** + logit + **Bayes** | Yes — multi-method |
| Segment safety | Country rates only | Country z-tests + **BH-FDR** | Yes — controls false discoveries |
| “Should we extend?” | n for MDE + power number | Same **+ power curve plot** | Yes — design transparency |
| Decision packaging | Manager paragraph | **Scorecard gates** + manager summary | Yes — production decision UX |
| Notebook artifacts | 3 plots | **6 plots** | Yes — richer evidence surface |

**Bottom line on improvement:** v2 does **not** claim a higher conversion rate or a lower p-value “hack.” It claims a **higher-confidence HOLD**: randomization looks healthy (SRM), the null is stable under bootstrap and Bayesian views, segments do not rescue the new page after FDR, and the test is already powered for a business-relevant +1 pp MDE.

### Techniques added in v2 (what we used and why)

| New technique | Role | Source of practice |
|---|---|---|
| **Sample Ratio Mismatch (SRM)** χ² GOF vs 50/50 | Trustworthiness gate before trusting effects | Microsoft ExP, DoorDash, Eppo, Statsig |
| **Wilson score CIs** for arm rates | Production-preferred proportion intervals | statsmodels `proportion_confint(method="wilson")` |
| **`proportions_chisquare` cross-check** | Confirms two-proportion z-test | statsmodels |
| **Bootstrap percentile CI** (2,000 resamples, seed 42) | Nonparametric robustness on \(p_{\text{new}}-p_{\text{old}}\) | Standard resampling |
| **Practical significance band (±0.5 pp)** | Separates “statistically nonzero” from “business-relevant” | Kohavi et al. trustworthy experimentation |
| **BH-FDR country segment tests** | Safe exploratory segments | statsmodels `multipletests` |
| **Power curve** vs n/arm for +1 pp MDE | Visual answer to EXTEND | `power_proportions_2indep` |
| **Bayesian Beta–Binomial** \(P(p_{\text{new}}>p_{\text{old}})\) | Secondary communication of uncertainty | Uniform prior cross-check (not primary) |
| **Production decision scorecard** | Explicit SHIP / HOLD / EXTEND gates incl. severe SRM | Experiment platform playbooks |

### Side-by-side outputs (executed numbers)

Primary effect estimates are intentionally **stable** across versions (same clean sample). New rows appear only where v2 adds methods.

| Metric | Baseline v1 (first full run) | Production v2 (current notebook) | Change |
|---|---|---|---|
| Raw rows | 294,478 | 294,478 | Same |
| Mismatched rows dropped | 3,893 (1.32%) | 3,893 (1.32%) | Same cleaning rule |
| Cleaned unique users | **290,584** | **290,584** | Same |
| Control n / rate | 145,274 / **12.0386%** | 145,274 / **12.0386%** | Same |
| Treatment n / rate | 145,310 / **11.8808%** | 145,310 / **11.8808%** | Same |
| Lift (new − old) | **−0.1578 pp** (−1.31% rel.) | **−0.1578 pp** (−1.31% rel.) | Same |
| Control conversions | 17,489 | 17,489 | Same |
| Treatment conversions | 17,264 | 17,264 | Same |
| z-statistic | **−1.3109** | **−1.3109** | Same |
| Two-sided p-value | **0.189883** | **0.189883** | Same → fail to reject \(H_0\) |
| Analytic 95% CI (diff) | **[−0.003938, 0.000781]** | **[−0.003938, 0.000781]** | Same (includes 0) |
| Arm 95% CI method | Normal approx | **Wilson** | Method upgrade |
| Bootstrap 95% CI (diff) | *(not computed)* | **[−0.003973, 0.000769]** | **New** — agrees with analytic |
| χ² proportions cross-check | *(not reported)* | Aligns with z-test (large-n) | **New** |
| OR unadjusted (`ab_page`) | **0.9851** | **0.9851** | Same |
| OR country-adjusted | **0.9852** (p=0.191245) | **0.9852** (p=0.191245) | Same |
| Country × page interaction | Explored; weak | Explored; min interaction p not significant at 0.05 | Same conclusion |
| SRM χ² p / imbalance | *(not computed)* | **0.946754** / **0.0062 pp** (not severe) | **New** — PASS |
| n/arm for 80% power @ observed effect | 663,574 | 663,574 | Same |
| n/arm for 80% power @ +1 pp MDE | 17,209 | 17,209 | Same |
| Power @ +1 pp MDE (actual n) | **~100%** | **100.0%** | Same (now + power curve) |
| Bayesian P(new > old) | *(not computed)* | **9.39%** | **New** — favors old page |
| Segment FDR discoveries | *(not computed)* | **0** countries after BH-FDR | **New** |
| Recommendation | **HOLD** | **HOLD** | Same call, stronger justification |
| Decision packaging | Manager paragraph | Scorecard + manager summary | **New** |
| Plot outputs in notebook | 3 | **6** | Richer diagnostics |
| Code cells / errors (QA) | 14 / 0 | 14 / 0 | Clean re-execution |

### Baseline v1 manager brief (preserved)

From the original executed analysis (pre-production suite):

```text
Raw rows:                      294,478
Mismatched group/page rows:    3,893  (dropped)
After de-duplicate user_id:    290,584

Old page (control):   12.0386%   n=145,274
New page (treatment): 11.8808%   n=145,310
Difference (new−old): -0.1578% absolute  (-1.31% relative)

Two-proportion z-test (α=0.05, two-sided)
  z = -1.3109
  p = 0.189883   →  FAIL TO REJECT H0
  95% CI for (p_new − p_old): [-0.003938, 0.000781]

Logistic regression odds ratio for new page (ab_page)
  Unadjusted OR:          0.9851
  Country-adjusted OR:    0.9852   (p=0.191245)

Power / sample size
  n/arm for 80% power @ observed effect: 663,574
  n/arm for 80% power @ +1 pp MDE:       17,209
  Actual min arm n:                      145,274
  Power for +1 pp MDE at actual n:       100.0%

RECOMMENDATION: HOLD
```

### Production v2 manager brief (current run)

```text
Cleaned users: 290,584  |  merge n: 290,584
Old page: 12.0386% (n=145,274)  |  New page: 11.8808% (n=145,310)
Lift: -0.1578 pp  |  relative -1.31%
z=-1.3109, p=0.189883, 95% CI diff=[-0.003938, 0.000781]
Bootstrap 95% CI: [-0.003973, 0.000769]
OR unadj=0.9851, OR country-adj=0.9852 (p=0.191245)
SRM χ² p=0.946754, imbalance=0.0062pp, severe=False
Power @ +1pp MDE: 100.0%  |  Bayesian P(new>old)=9.39%
RECOMMENDATION: HOLD
```

### What improved between v1 and v2 (summary)

1. **Integrity:** SRM shows the 50/50 assignment is healthy (p≈0.95, tiny imbalance).  
2. **Robustness:** Bootstrap CI matches the analytic CI → lift estimate is not an artifact of one formula.  
3. **Practical framing:** ±0.5 pp indifference band + +1 pp MDE power curve make EXTEND vs HOLD a quantitative call.  
4. **Multi-method agreement:** Frequentist null + OR≈1 + Bayesian P(new better)≈9% all point the same way.  
5. **Safer exploration:** Country peeks are FDR-controlled.  
6. **Production UX:** Scorecard encodes gates (quality → SRM → significance → practical → power → recommend).

Primary conversion metrics **did not get “better” for the new page** — and that is correct. The **analysis quality** got better.

---

## Advanced v3 — package, CLI, tutorials

**Design rule:** do **not** rewrite or delete the baseline notebook. v3 is additive.

| Path | Role |
|---|---|
| `notebooks/ab_test_marketing_analysis.ipynb` | **Unchanged** baseline/v2 learning notebook |
| `notebooks/02_advanced_production_upgrades.ipynb` | **New** executed advanced tutorial |
| `ab_test/` | Importable package (cleaning, SRM, TOST, sim, CUPED, scorecard, CLI) |
| `tutorials/` | Conceptual lessons (why each upgrade) |
| `metrics/latest.json` | Machine-readable decision metrics |
| `artifacts/decision_report.html` | Manager-facing HTML memo |

### New techniques in v3 (and why)

| Technique | Why we do it | How it improves the “result” | Live outcome on this data |
|---|---|---|---|
| **TOST equivalence (±0.5 pp)** | “Not significant” ≠ “practically the same” | Formal language for indifference | **Equivalent = True** (p_lo≈0.0022, p_hi≈2.3e-8) |
| **ITT vs as-treated** | Exposure logging can be wrong | Recommendation robust to cleaning policy | **Both HOLD**, agreement **YES** |
| **Simulation harness** | Prove pipeline is not hard-coded HOLD | Recovers SHIP when true lift is large | True +1 pp / +2 pp → **SHIP**; null → **HOLD** |
| **CUPED demo (synthetic pre-period)** | Teach production variance reduction | Literacy + hook for real covariates | ~**0.6%** var reduction on synthetic X (demo only) |
| **CLI + metrics.json + HTML** | Operator / CI surface | Repeatable artifacts | `uv run ab-test run\|scorecard\|simulate\|report` |

### v3 outputs (from `uv run ab-test run`)

```text
Clean n (as-treated): 290,584
Rates: control=12.0386%  treatment=11.8808%  diff=-0.1578 pp
z=-1.3109  p=0.189883
CI=[-0.003938, 0.000781]  boot=[-0.003973, 0.000769]
SRM p=0.946754  imbalance=0.0062pp  severe=False
TOST equivalent (±0.50pp): True
Bayesian P(new>old)≈9.6%
ITT: rec=HOLD  agree=YES
RECOMMENDATION: HOLD
```

### Simulation suite (from `uv run ab-test simulate`)

| Scenario | True lift | Observed (example run) | Significant | Rec |
|---|---:|---:|---|---|
| true_null_lift_0 | 0.00 pp | ~0.02 pp | False | **HOLD** |
| true_tiny_lift_plus_0.1pp | +0.10 pp | ~0.24 pp | False | **HOLD** |
| true_mde_lift_plus_1pp | +1.00 pp | ~1.06 pp | True | **SHIP** |
| true_large_lift_plus_2pp | +2.00 pp | ~1.93 pp | True | **SHIP** |
| true_negative_lift_minus_1pp | −1.00 pp | ~−1.08 pp | True | **HOLD** |

This is the key “awesome” proof: **the system can ship when the truth is a real lift** — it just refuses to ship this Udacity creative.

### CLI (advanced v3)

```bash
uv sync

uv run ab-test run          # analysis → metrics/latest.json
uv run ab-test scorecard    # print gates
uv run ab-test simulate     # known-truth calibration
uv run ab-test report       # metrics.json + artifacts/decision_report.html
```

### Tutorials (explanations)

Start at [`tutorials/README.md`](tutorials/README.md):

0. Learning path (keep old notebook)  
1. Trustworthiness gates  
2. TOST equivalence  
3. ITT vs as-treated  
4. Simulation harness  
5. CUPED demo  
6. CLI / metrics / HTML

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

### Decision 8 — Flat project layout, uv-only environment

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

Numbers below are from the **current production v2** executed notebook. For the full **v1 baseline vs v2** table (old numbers preserved), see [Baseline vs production upgrade](#baseline-v1-vs-production-v2--outputs--techniques).

**Runtime fingerprint (v2 executed notebook):**

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

### Conversion (cleaned A/B sample) — unchanged from v1

| Arm | Conversions | n | Rate |
|---|---:|---:|---:|
| Old page (control) | 17,489 | 145,274 | **12.0386%** |
| New page (treatment) | 17,264 | 145,310 | **11.8808%** |
| Difference (new − old) | — | — | **−0.1578 pp** (−1.311% relative) |

**v1:** normal-approx arm CIs. **v2:** Wilson score arm CIs (production-preferred for proportions).

### Trustworthiness — SRM *(v2 only)*

```text
SRM χ² = 0.004460
SRM p  = 0.946754
Traffic imbalance = 0.0062 pp
Severe SRM = False  → analysis may proceed
SRM gate (p >= 0.001): PASS
```

### Hypothesis test + robustness

```text
# Same primary test as v1
z = -1.310924
p = 0.189883   →  FAIL TO REJECT H0 at α = 0.05
Analytic 95% CI for (p_new − p_old) = [-0.003938, 0.000781]   # includes 0

# v2 additions
Bootstrap 95% CI                    = [-0.003973, 0.000769]   # agrees with analytic
proportions χ² cross-check          = consistent with z-test at this n
```

### Logistic odds ratios (`ab_page` = new creative) — same point estimates as v1

| Model | OR | Interpretation |
|---|---:|---|
| Unadjusted | 0.9851 | ~1.5% lower odds of conversion on new page |
| Country-adjusted | 0.9852 (p = 0.191245) | Effect unchanged after country |
| Interaction check | no strong country×page interaction | Use Model 2 as primary adjusted summary |

### Bayesian secondary check *(v2 only)*

```text
P(p_new > p_old | data) = 9.3890%   (uniform Beta–Binomial, 100k MC draws)
Central 95% posterior interval for lift includes values around the frequentist CI
```

### Power *(same required-n math as v1; v2 adds curve)*

| Check | Value |
|---|---:|
| n/arm for 80% power @ observed effect | 663,574 |
| n/arm for 80% power @ +1 pp MDE | 17,209 |
| Actual min arm n | 145,274 |
| Power for +1 pp MDE at actual n | **100.0%** |

### Decision rule used (transparent, pre-structured) — v2 scorecard

```text
if severe SRM                         → HOLD (do not ship on broken randomization)
elif significant and lift ≥ +0.5 pp   → SHIP
elif significant and not improved     → HOLD
elif underpowered for +1 pp MDE       → EXTEND
else                                  → HOLD
```

**Applied outcome (v1 and v2): HOLD.**

**v2 rationale (stronger than v1):** no significant improvement; point estimate slightly negative; analytic **and** bootstrap CIs cover 0; OR≈0.985; Bayesian P(new better)≈9%; SRM clean; already powered for +1 pp MDE; no FDR country rescue of the new page.

### Manager-ready brief (v2 executed summary cell)

> On **290,584** cleaned users, the new marketing-campaign landing page converted at **11.88%** vs **12.04%** for the old page (**−0.158 pp**). z = **−1.31**, p = **0.190** (not significant). Analytic CI **[−0.003938, 0.000781]** and bootstrap CI **[−0.003973, 0.000769]** both cover 0. Country-adjusted OR = **0.985**. SRM p = **0.947** (not severe). Power for +1 pp = **100%**. Bayesian P(new better) = **9.39%**. **Decision: HOLD.**

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

| Version | What you get | Learning artifact |
|---|---|---|
| **v1 baseline** | Clean data, z-test, logit ORs, power, **HOLD** | Preserved in README tables + original notebook history |
| **v2 production** | + SRM, Wilson, bootstrap, χ², FDR, power curve, Bayesian, scorecard | `notebooks/ab_test_marketing_analysis.ipynb` (**kept**) |
| **v3 advanced** | + TOST, ITT sensitivity, simulation, CUPED demo, CLI/JSON/HTML | `02_advanced_*.ipynb`, `ab_test/`, `tutorials/` |

This project demonstrates a **production-minded experiment analysis** workflow: clean messy assignments, pass an **SRM** gate, estimate effects with multi-method confirmation, prove calibration via simulation, and export operator artifacts — then still make an honest **HOLD** call when the creative does not win. **“Better results” means a better decision system — not a fabricated lift.**
