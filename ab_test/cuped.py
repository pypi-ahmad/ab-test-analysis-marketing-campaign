"""CUPED-style variance reduction demo on synthetic pre-period covariates.

The public Udacity log has no pre-experiment covariate. For teaching, we
simulate a pre-period proxy correlated with conversion, then apply the
classic CUPED residualization:

    Y_cuped = Y - θ (X - mean(X))
    θ = Cov(Y, X) / Var(X)

This demonstrates *how* production platforms reduce variance — not a claim
that this dataset ships with real pre-period metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CUPEDDemoResult:
    var_y: float
    var_cuped: float
    var_reduction_pct: float
    theta: float
    corr_yx: float
    se_diff_raw: float
    se_diff_cuped: float
    se_reduction_pct: float
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _arm_diff_se(y: np.ndarray, group: np.ndarray) -> float:
    """SE of mean(treatment) - mean(control) for binary/continuous y."""
    yt = y[group == 1]
    yc = y[group == 0]
    return float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))


def simulate_preperiod_and_cuped(
    converted: np.ndarray,
    is_treatment: np.ndarray,
    *,
    corr_strength: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, CUPEDDemoResult]:
    """Build synthetic pre-period X correlated with Y and apply CUPED.

    Parameters
    ----------
    converted:
        Outcome 0/1.
    is_treatment:
        1 for treatment, 0 for control.
    corr_strength:
        How strongly latent signal drives both X and Y noise (0–1-ish).
    """
    rng = np.random.default_rng(seed)
    y = converted.astype(float)
    n = y.size
    # Shared latent + noise → continuous pre-period proxy in [0, 1]-ish
    latent = 0.5 * y + rng.normal(0, 1.0, size=n)
    x = corr_strength * latent + (1 - corr_strength) * rng.normal(0, 1.0, size=n)
    x = (x - x.mean()) / (x.std() + 1e-12)

    cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
    var_x = float(np.var(x, ddof=1))
    theta = cov_yx / var_x if var_x > 0 else 0.0
    y_cuped = y - theta * (x - x.mean())

    var_y = float(np.var(y, ddof=1))
    var_c = float(np.var(y_cuped, ddof=1))
    red = 100.0 * (1.0 - var_c / var_y) if var_y > 0 else 0.0

    se_raw = _arm_diff_se(y, is_treatment)
    se_c = _arm_diff_se(y_cuped, is_treatment)
    se_red = 100.0 * (1.0 - se_c / se_raw) if se_raw > 0 else 0.0

    df = pd.DataFrame(
        {
            "converted": y,
            "ab_page": is_treatment.astype(int),
            "preperiod_x": x,
            "converted_cuped": y_cuped,
        }
    )
    result = CUPEDDemoResult(
        var_y=var_y,
        var_cuped=var_c,
        var_reduction_pct=float(red),
        theta=float(theta),
        corr_yx=float(np.corrcoef(y, x)[0, 1]),
        se_diff_raw=se_raw,
        se_diff_cuped=se_c,
        se_reduction_pct=float(se_red),
        note=(
            "Synthetic pre-period covariate for teaching CUPED only. "
            "Not present in the public ab_data.csv."
        ),
    )
    return df, result
