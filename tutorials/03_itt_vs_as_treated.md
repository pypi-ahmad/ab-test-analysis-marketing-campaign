# 03 — ITT vs as-treated sensitivity

## Why two cleaning policies?

### As-treated (primary in baseline)

Drop rows where `group` and `landing_page` disagree.  
**Rationale:** if we do not know what page the user saw, we should not score conversion for that exposure.

### Intent-to-treat (ITT) by assignment

Keep users; analyze only by randomized `group`.  
**Rationale:** randomization is the design; instrumentation may mis-log the page.

## How this improves the decision

A recommendation is **more awesome** when it survives reasonable policy changes:

| Pattern | Interpretation |
|---|---|
| Both HOLD | Robust — ship is unjustified either way |
| Both SHIP | Robust win (not what this dataset shows) |
| Disagree | Escalate: fix logging / re-run — don’t cherry-pick |

## What we report

- As-treated: n, diff, p, recommendation  
- ITT: n, diff, p, recommendation  
- `itt_agreement`: YES/NO  

## Where to run

- Advanced notebook §3  
- `ab_test.cleaning.clean_as_treated` / `clean_itt_by_assignment`  
- CLI metrics: `itt_*` fields in `metrics/latest.json`
