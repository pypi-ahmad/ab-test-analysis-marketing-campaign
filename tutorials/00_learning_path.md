# 00 — Learning path (keep the old notebook)

## Goal

Learn A/B analysis in layers without losing the original tutorial.

## Path

1. **Baseline teaching notebook (required first)**  
   `notebooks/ab_test_marketing_analysis.ipynb`  
   - Cleaning mismatches/duplicates  
   - SRM, z-test, logit ORs, power, Bayesian cross-check (v2)  
   - Real HOLD recommendation  

2. **README comparison**  
   Section *Baseline (v1) vs production (v2)*  
   - Old numbers preserved  
   - New techniques called out  

3. **These conceptual tutorials (01–06)**  
   - Why each upgrade exists  
   - What “improvement” means  

4. **Advanced notebook (v3)**  
   `notebooks/02_advanced_production_upgrades.ipynb`  
   - TOST, ITT sensitivity, simulation, CUPED demo, exports  

5. **CLI**  
   ```bash
   uv run ab-test run
   uv run ab-test scorecard
   uv run ab-test simulate
   uv run ab-test report
   ```

## What we never do

- Edit away the baseline notebook “to look better”
- Change α after seeing p-values
- Pretend the new page won when it did not

## Success check

You can explain in one minute:

> “The creative still doesn’t win. The **system** that decides ship/hold is now production-grade.”
