# Tutorials — advanced production A/B upgrades

These tutorials explain **why** we added each upgrade and **how** it improves decision quality.

The original learning notebook is unchanged:

- `notebooks/ab_test_marketing_analysis.ipynb` — baseline → production v2 story (keep for learning)

New executable companion:

- `notebooks/02_advanced_production_upgrades.ipynb` — TOST, ITT, simulation, CUPED demo, CLI artifacts

| # | Tutorial | Topic |
|---|---|---|
| 00 | [00_learning_path.md](00_learning_path.md) | What to read in what order |
| 01 | [01_why_trustworthiness_gates.md](01_why_trustworthiness_gates.md) | SRM, cleaning, scorecards |
| 02 | [02_tost_equivalence.md](02_tost_equivalence.md) | Practical equivalence |
| 03 | [03_itt_vs_as_treated.md](03_itt_vs_as_treated.md) | Sensitivity to exposure policy |
| 04 | [04_simulation_harness.md](04_simulation_harness.md) | Calibrating SHIP/HOLD |
| 05 | [05_cuped_demo.md](05_cuped_demo.md) | Variance reduction literacy |
| 06 | [06_cli_metrics_html.md](06_cli_metrics_html.md) | Operator surface |

**Core message:** better results here mean a better **decision system**, not a fabricated conversion win.
