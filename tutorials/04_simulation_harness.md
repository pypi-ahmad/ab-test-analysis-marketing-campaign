# 04 — Simulation harness: proving the pipeline is calibrated

## Why

If a system always says HOLD on a famous null dataset, skeptics ask:

> “Would it ever say SHIP?”

Simulation answers with **known ground truth**.

## Scenarios we run

| Scenario | True lift | What “good” looks like |
|---|---|---|
| Null | 0 | Usually not significant → HOLD |
| Tiny +0.1 pp | +0.001 | Often undetected even at large n |
| MDE +1 pp | +0.01 | High power at ~145k/arm → often significant |
| Large +2 pp | +0.02 | Should clearly detect |
| Negative −1 pp | −0.01 | Significant wrong-way → HOLD (don’t ship) |

## How this improves results

It improves **method credibility**:

- Shows the scorecard is not hard-coded to HOLD  
- Separates “this creative failed” from “the analyzer is blind”

## Where to run

```bash
uv run ab-test simulate
```

Also advanced notebook §4.

## Reading the table

- `correct_hold_when_null`: for true lift 0, recommendation is HOLD  
- `correct_ship_when_large`: for large positive lifts, observed effect is significant and ≥ practical band  
