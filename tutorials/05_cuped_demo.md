# 05 — CUPED demo (synthetic pre-period)

## Why production teams care

**CUPED** (Controlled-experiment Using Pre-Experiment Data) reduces variance of the metric using covariates observed *before* the experiment. Lower variance ⇒ more power for the same sample size.

## Why our demo is synthetic

`ab_data.csv` has **no pre-period metric**. Claiming real CUPED on this file would be dishonest.

We therefore:

1. Simulate a pre-period proxy `X` correlated with conversion `Y`  
2. Fit θ = Cov(Y, X) / Var(X)  
3. Form Y_cuped = Y − θ (X − mean X)  
4. Compare Var(Y) vs Var(Y_cuped) and SE of the arm difference  

## How this improves the project

| Improvement | Explanation |
|---|---|
| Literacy | You can explain CUPED in interviews |
| Engineering readiness | Hook exists when real pre-period data appears |
| Honesty | Labeled as demo, not fake lift |

## What it does *not* do

It does not change the Udacity page’s true conversion rates or justify SHIP.

## Where to run

- Advanced notebook §5  
- `ab_test.cuped.simulate_preperiod_and_cuped`  
- Metrics: `cuped_demo` in `metrics/latest.json`
