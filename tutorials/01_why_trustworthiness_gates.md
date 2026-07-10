# 01 — Why trustworthiness gates (and how they improve results)

## The problem

If you only look at conversion rates, you can ship a “winner” that is:

- based on broken randomization,
- based on mislabeled exposure,
- statistically noisy,
- or practically meaningless.

Production platforms (Microsoft ExP, DoorDash, Eppo, Statsig) therefore run **gates before celebrating lifts**.

## Gates we use

| Gate | Question | Failure mode if skipped |
|---|---|---|
| Schema | Is the log well-formed? | Silent column bugs |
| Exposure cleaning | Did the user actually see the assigned page? | Biased “treatment” labels |
| SRM | Is traffic split as designed? | Confounded comparisons |
| Primary test + CI | Is there a detectable difference? | Overreacting to noise |
| Practical band / TOST | Is the difference *business-relevant*? | Shipping tiny junk lifts |
| Power vs MDE | Could we have seen a real win? | Endless “extend” theater |
| Scorecard | Can a human audit the decision? | Irreproducible tribal knowledge |

## How this improves “results”

It improves **decision quality**:

- Fewer false ships
- Clearer HOLDs
- Documented reasons

It does **not** magically raise the new page’s conversion rate.

## Where to see it

- Baseline notebook Parts 1b + scorecard-style conclusion  
- `ab_test/scorecard.py`  
- CLI: `uv run ab-test scorecard`
