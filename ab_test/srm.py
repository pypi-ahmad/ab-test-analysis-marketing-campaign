"""Sample Ratio Mismatch (SRM) trustworthiness gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    n_control: int
    n_treatment: int
    expected_split: float
    chi2: float
    pvalue: float
    treatment_share: float
    imbalance_pp: float
    pass_gate: bool
    severe: bool

    def to_dict(self) -> dict:
        return asdict(self)


def sample_ratio_mismatch(
    n_control: int,
    n_treatment: int,
    *,
    expected_split: float = 0.5,
    srm_alpha: float = 0.001,
    severe_imbalance_pp: float = 0.5,
) -> SRMResult:
    n_total = n_control + n_treatment
    if n_total <= 0:
        raise ValueError("empty arms")
    obs = np.array([n_control, n_treatment], dtype=float)
    exp = np.array(
        [expected_split * n_total, (1.0 - expected_split) * n_total],
        dtype=float,
    )
    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
    share = n_treatment / n_total
    imb = abs(share - expected_split) * 100.0
    pass_gate = p >= srm_alpha
    severe = (p < srm_alpha) and (imb > severe_imbalance_pp)
    return SRMResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_split=expected_split,
        chi2=float(chi2),
        pvalue=float(p),
        treatment_share=float(share),
        imbalance_pp=float(imb),
        pass_gate=bool(pass_gate),
        severe=bool(severe),
    )
