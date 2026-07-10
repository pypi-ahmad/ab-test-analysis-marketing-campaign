"""Power and required sample size helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import power_proportions_2indep, proportion_effectsize


@dataclass
class PowerReport:
    p_control: float
    p_treatment: float
    diff: float
    es_obs: float
    n_for_obs_effect: float
    power_obs: float
    mde_abs_pp: float
    es_mde: float
    n_for_mde: float
    power_mde: float
    n_per_arm_actual: int
    underpowered_for_mde: bool

    def to_dict(self) -> dict:
        return asdict(self)


def power_report(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    *,
    mde_abs_pp: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> PowerReport:
    es_obs = float(proportion_effectsize(p_treatment, p_control))
    if abs(es_obs) < 1e-12:
        n_for_obs = float("inf")
    else:
        n_for_obs = float(
            NormalIndPower().solve_power(
                effect_size=abs(es_obs),
                alpha=alpha,
                power=power_target,
                ratio=1.0,
                alternative="two-sided",
            )
        )
    pwr_obs = power_proportions_2indep(
        diff=p_treatment - p_control,
        prop2=p_control,
        nobs1=n_treatment,
        ratio=n_control / n_treatment,
        alpha=alpha,
        alternative="two-sided",
        return_results=True,
    )
    p_mde = p_control + mde_abs_pp
    es_mde = float(proportion_effectsize(p_mde, p_control))
    n_for_mde = float(
        NormalIndPower().solve_power(
            effect_size=es_mde,
            alpha=alpha,
            power=power_target,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    pwr_mde = power_proportions_2indep(
        diff=mde_abs_pp,
        prop2=p_control,
        nobs1=n_treatment,
        ratio=n_control / n_treatment,
        alpha=alpha,
        alternative="two-sided",
        return_results=True,
    )
    actual = min(n_control, n_treatment)
    return PowerReport(
        p_control=p_control,
        p_treatment=p_treatment,
        diff=p_treatment - p_control,
        es_obs=es_obs,
        n_for_obs_effect=n_for_obs,
        power_obs=float(pwr_obs.power),
        mde_abs_pp=mde_abs_pp,
        es_mde=es_mde,
        n_for_mde=n_for_mde,
        power_mde=float(pwr_mde.power),
        n_per_arm_actual=actual,
        underpowered_for_mde=bool(pwr_mde.power < power_target),
    )
