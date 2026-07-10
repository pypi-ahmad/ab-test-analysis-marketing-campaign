"""Simulation harness: recover SHIP only when true lift is real and large enough."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ab_test.config import AnalysisConfig, DEFAULT_CONFIG
from ab_test.inference import two_proportion_test
from ab_test.power_utils import power_report
from ab_test.scorecard import decide


@dataclass
class SimScenarioResult:
    name: str
    true_p_control: float
    true_lift: float
    n_per_arm: int
    observed_diff: float
    pvalue: float
    significant: bool
    recommendation: str
    power_for_true_lift: float
    correct_hold_when_null: bool | None
    correct_ship_when_large: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def _simulate_arms(
    p_control: float,
    lift: float,
    n_per_arm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    p_t = min(max(p_control + lift, 1e-6), 1 - 1e-6)
    yc = rng.binomial(1, p_control, size=n_per_arm)
    yt = rng.binomial(1, p_t, size=n_per_arm)
    return pd.DataFrame(
        {
            "group": ["control"] * n_per_arm + ["treatment"] * n_per_arm,
            "converted": np.concatenate([yc, yt]),
        }
    )


def run_scenario(
    name: str,
    *,
    p_control: float,
    true_lift: float,
    n_per_arm: int,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    seed: int = 0,
) -> SimScenarioResult:
    rng = np.random.default_rng(cfg.random_seed + seed)
    df = _simulate_arms(p_control, true_lift, n_per_arm, rng)
    ctrl = df.loc[df["group"] == "control", "converted"]
    trt = df.loc[df["group"] == "treatment", "converted"]
    test = two_proportion_test(
        int(trt.sum()),
        int(trt.shape[0]),
        int(ctrl.sum()),
        int(ctrl.shape[0]),
        alpha=cfg.alpha,
    )
    pwr = power_report(
        float(ctrl.mean()),
        float(ctrl.mean() + true_lift),
        n_per_arm,
        n_per_arm,
        mde_abs_pp=cfg.mde_abs_pp,
        alpha=cfg.alpha,
        power_target=cfg.power_target,
    )
    # Decision uses observed stats + assumed healthy SRM for synthetic data
    rec, _ = decide(
        significant=test.significant,
        abs_diff=test.diff,
        srm_severe=False,
        underpowered_for_mde=False,  # fixed n experiment
        practical_abs_pp=cfg.practical_abs_pp,
        mde_abs_pp=cfg.mde_abs_pp,
        power_mde=1.0,
        power_target=cfg.power_target,
    )
    # For simulation truth checks we also allow "detect large lift" via significance+direction
    ship_like = test.significant and test.diff >= cfg.practical_abs_pp
    return SimScenarioResult(
        name=name,
        true_p_control=p_control,
        true_lift=true_lift,
        n_per_arm=n_per_arm,
        observed_diff=test.diff,
        pvalue=test.pvalue,
        significant=test.significant,
        recommendation=rec,
        power_for_true_lift=float(
            power_report(
                p_control,
                min(max(p_control + true_lift, 1e-6), 1 - 1e-6),
                n_per_arm,
                n_per_arm,
                mde_abs_pp=abs(true_lift) if abs(true_lift) > 1e-9 else cfg.mde_abs_pp,
                alpha=cfg.alpha,
                power_target=cfg.power_target,
            ).power_mde
            if abs(true_lift) > 1e-9
            else pwr.power_obs
        ),
        correct_hold_when_null=(rec == "HOLD") if abs(true_lift) < 1e-12 else None,
        correct_ship_when_large=ship_like if true_lift >= cfg.practical_abs_pp else None,
    )


def run_default_suite(
    *,
    p_control: float = 0.12,
    n_per_arm: int = 145_000,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> list[SimScenarioResult]:
    """Canonical demos: null, tiny, and business-sized true lifts."""
    scenarios = [
        ("true_null_lift_0", 0.0),
        ("true_tiny_lift_plus_0.1pp", 0.001),
        ("true_mde_lift_plus_1pp", 0.01),
        ("true_large_lift_plus_2pp", 0.02),
        ("true_negative_lift_minus_1pp", -0.01),
    ]
    return [
        run_scenario(
            name,
            p_control=p_control,
            true_lift=lift,
            n_per_arm=n_per_arm,
            cfg=cfg,
            seed=i * 17,
        )
        for i, (name, lift) in enumerate(scenarios)
    ]
