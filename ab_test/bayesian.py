"""Beta–Binomial secondary view for two proportions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class BayesResult:
    p_new_better: float
    p_old_better: float
    posterior_mean_lift: float
    ci_low: float
    ci_high: float
    mc: int

    def to_dict(self) -> dict:
        return asdict(self)


def beta_binomial_lift(
    conv_treatment: int,
    n_treatment: int,
    conv_control: int,
    n_control: int,
    *,
    mc: int = 50_000,
    seed: int = 42,
) -> BayesResult:
    """Uniform Beta(1,1) prior on each arm."""
    rng = np.random.default_rng(seed)
    post_c = stats.beta(1 + conv_control, 1 + (n_control - conv_control))
    post_t = stats.beta(1 + conv_treatment, 1 + (n_treatment - conv_treatment))
    draws_c = post_c.rvs(mc, random_state=rng)
    draws_t = post_t.rvs(mc, random_state=rng)
    lift = draws_t - draws_c
    lo, hi = np.quantile(lift, [0.025, 0.975])
    return BayesResult(
        p_new_better=float(np.mean(draws_t > draws_c)),
        p_old_better=float(np.mean(draws_c > draws_t)),
        posterior_mean_lift=float(lift.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        mc=mc,
    )
