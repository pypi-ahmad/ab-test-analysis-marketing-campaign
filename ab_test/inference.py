"""Frequentist effect tests: z-test, χ², bootstrap, TOST equivalence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_chisquare,
    proportions_ztest,
)


@dataclass
class ProportionTestResult:
    p_control: float
    p_treatment: float
    diff: float
    zstat: float
    pvalue: float
    ci_low: float
    ci_high: float
    chi2stat: float
    chi2_pvalue: float
    significant: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BootstrapCI:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_boot: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TOSTResult:
    """Two one-sided tests for equivalence within ±margin."""

    margin: float
    diff: float
    se: float
    # H0: diff <= -margin  vs H1: diff > -margin
    t_lower: float
    p_lower: float
    # H0: diff >= +margin  vs H1: diff < +margin
    t_upper: float
    p_upper: float
    equivalent: bool
    # CI-based equivalence: entire CI inside (-margin, margin)
    ci_low: float
    ci_high: float
    ci_inside_margin: bool

    def to_dict(self) -> dict:
        return asdict(self)


def two_proportion_test(
    conv_treatment: int,
    n_treatment: int,
    conv_control: int,
    n_control: int,
    *,
    alpha: float = 0.05,
) -> ProportionTestResult:
    count = np.array([conv_treatment, conv_control], dtype=float)
    nobs = np.array([n_treatment, n_control], dtype=float)
    p_t = conv_treatment / n_treatment
    p_c = conv_control / n_control
    z, p = proportions_ztest(count, nobs, alternative="two-sided")
    ci_low, ci_high = confint_proportions_2indep(
        count1=conv_treatment,
        nobs1=n_treatment,
        count2=conv_control,
        nobs2=n_control,
        compare="diff",
        alpha=alpha,
    )
    chi2, chi2_p, _ = proportions_chisquare(count, nobs)
    return ProportionTestResult(
        p_control=float(p_c),
        p_treatment=float(p_t),
        diff=float(p_t - p_c),
        zstat=float(z),
        pvalue=float(p),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        chi2stat=float(chi2),
        chi2_pvalue=float(chi2_p),
        significant=bool(p < alpha),
    )


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def bootstrap_diff_ci(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bc = rng.choice(y_control, size=y_control.size, replace=True)
        bt = rng.choice(y_treatment, size=y_treatment.size, replace=True)
        diffs[i] = bt.mean() - bc.mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(
        mean_diff=float(diffs.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
    )


def _pooled_se(p1: float, n1: int, p2: float, n2: int) -> float:
    return float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


def tost_equivalence(
    conv_treatment: int,
    n_treatment: int,
    conv_control: int,
    n_control: int,
    *,
    margin: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """TOST for equivalence of two proportions within ±margin.

    Reject both one-sided nulls at level ``alpha`` ⇒ declare equivalence
    (practically the same conversion within the indifference band).

    Also reports whether the two-sided (1-2α) style CI logic via the usual
    95% diff CI lies entirely inside (-margin, margin) when alpha=0.05.
    """
    p_t = conv_treatment / n_treatment
    p_c = conv_control / n_control
    diff = p_t - p_c
    se = _pooled_se(p_t, n_treatment, p_c, n_control)
    if se <= 0:
        raise ValueError("non-positive SE")

    # Lower TOST: H0: diff <= -margin
    t_lo = (diff - (-margin)) / se
    p_lo = 1.0 - stats.norm.cdf(t_lo)
    # Upper TOST: H0: diff >= +margin
    t_hi = (margin - diff) / se
    p_hi = 1.0 - stats.norm.cdf(t_hi)

    # Analytic 95% CI for reporting (same as primary test at alpha=0.05)
    primary = two_proportion_test(
        conv_treatment, n_treatment, conv_control, n_control, alpha=alpha
    )
    ci_inside = (primary.ci_low > -margin) and (primary.ci_high < margin)
    equivalent = (p_lo < alpha) and (p_hi < alpha)

    return TOSTResult(
        margin=float(margin),
        diff=float(diff),
        se=float(se),
        t_lower=float(t_lo),
        p_lower=float(p_lo),
        t_upper=float(t_hi),
        p_upper=float(p_hi),
        equivalent=bool(equivalent),
        ci_low=float(primary.ci_low),
        ci_high=float(primary.ci_high),
        ci_inside_margin=bool(ci_inside),
    )
