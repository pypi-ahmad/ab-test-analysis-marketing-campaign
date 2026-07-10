# 02 — TOST equivalence: when “not significant” is not enough

## Why we add this

People misread p = 0.19 as:

- “the pages are the same,” or  
- “we need more data.”

Neither follows automatically.

**TOST** (two one-sided tests) answers a different question:

> Can we claim the true lift is inside a **practical indifference band** ±δ?

## How it works (intuition)

Pick δ (we use **0.5 percentage points** absolute).

1. Test whether lift is **greater than −δ**  
2. Test whether lift is **less than +δ**  

If both succeed at level α, declare **equivalence**.

Also check whether the usual 95% CI for the difference sits entirely inside (−δ, δ).

## How this improves the analysis

| Before | After TOST |
|---|---|
| “Fail to reject H0” (ambiguous) | “Equivalent within ±0.5 pp” **or** “not proven equivalent” |
| Extend by default | Extend only if underpowered for a pre-set MDE |

## Important

Equivalence is **not** the same as “new page wins.”  
On this dataset, the point estimate is slightly **negative**; TOST may or may not declare equivalence depending on δ and CI width — the notebook prints the live outcome.

## Where to run

- Advanced notebook §2  
- `ab_test.inference.tost_equivalence`  
- Metrics key: `tost_equivalent`
