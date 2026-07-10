# 06 — CLI, metrics.json, and HTML decision memo

## Why leave the notebook alone *and* add a CLI?

| Artifact | Audience |
|---|---|
| Baseline notebook | Learners, narrative, plots |
| Advanced notebook | Deep upgrades with explanation |
| CLI | Operators, CI, demos |
| `metrics/latest.json` | Machines, diffs, dashboards |
| HTML report | Managers, PR attachments |

## Commands

From the project root (after `uv sync`):

```bash
# Full advanced analysis → metrics/latest.json
uv run ab-test run

# Print scorecard gates
uv run ab-test scorecard

# Simulation harness
uv run ab-test simulate

# metrics.json + artifacts/decision_report.html
uv run ab-test report
```

Useful flags:

```bash
uv run ab-test run --mde-pp 0.01 --practical-pp 0.005 --seed 42
uv run ab-test report --html-out artifacts/decision_report.html
```

## How this improves results

- **Reproducibility:** same entrypoint every time  
- **Reviewability:** JSON + HTML outlive notebook scrolling  
- **Separation of concerns:** teaching notebooks stay pedagogical; package code is tested/used by CLI  

## CI sketch (optional)

```bash
uv sync
uv run ab-test run
uv run ab-test simulate
test -f metrics/latest.json
```
