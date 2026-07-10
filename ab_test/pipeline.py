"""End-to-end advanced analysis → metrics dict + scorecard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ab_test.bayesian import beta_binomial_lift
from ab_test.cleaning import (
    clean_as_treated,
    clean_itt_by_assignment,
    conversion_summary,
)
from ab_test.config import AnalysisConfig, DEFAULT_CONFIG
from ab_test.cuped import simulate_preperiod_and_cuped
from ab_test.data import load_ab_and_countries
from ab_test.inference import bootstrap_diff_ci, tost_equivalence, two_proportion_test
from ab_test.power_utils import power_report
from ab_test.scorecard import Scorecard, build_scorecard
from ab_test.srm import sample_ratio_mismatch


def _policy_block(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    s = conversion_summary(df)
    test = two_proportion_test(
        int(s["conv_treatment"]),
        int(s["n_treatment"]),
        int(s["conv_control"]),
        int(s["n_control"]),
        alpha=cfg.alpha,
    )
    srm = sample_ratio_mismatch(
        int(s["n_control"]),
        int(s["n_treatment"]),
        expected_split=cfg.expected_split,
        srm_alpha=cfg.srm_alpha,
        severe_imbalance_pp=cfg.srm_severe_imbalance_pp,
    )
    y_c = df.loc[df["group"] == "control", "converted"].to_numpy()
    y_t = df.loc[df["group"] == "treatment", "converted"].to_numpy()
    boot = bootstrap_diff_ci(
        y_c, y_t, n_boot=cfg.bootstrap_reps, alpha=cfg.alpha, seed=cfg.random_seed
    )
    tost = tost_equivalence(
        int(s["conv_treatment"]),
        int(s["n_treatment"]),
        int(s["conv_control"]),
        int(s["n_control"]),
        margin=cfg.practical_abs_pp,
        alpha=cfg.alpha,
    )
    pwr = power_report(
        float(s["p_control"]),
        float(s["p_treatment"]),
        int(s["n_control"]),
        int(s["n_treatment"]),
        mde_abs_pp=cfg.mde_abs_pp,
        alpha=cfg.alpha,
        power_target=cfg.power_target,
    )
    bayes = beta_binomial_lift(
        int(s["conv_treatment"]),
        int(s["n_treatment"]),
        int(s["conv_control"]),
        int(s["n_control"]),
        mc=cfg.bayes_mc,
        seed=cfg.random_seed,
    )
    return {
        "summary": s,
        "test": test,
        "srm": srm,
        "bootstrap": boot,
        "tost": tost,
        "power": pwr,
        "bayes": bayes,
    }


def analyze(
    *,
    prefer_local: bool = True,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    include_cuped_demo: bool = True,
) -> tuple[dict[str, Any], Scorecard]:
    """Run advanced pipeline on the public dataset."""
    ab, _countries, load_path = load_ab_and_countries(prefer_local=prefer_local)

    as_treated = clean_as_treated(ab)
    itt = clean_itt_by_assignment(ab)

    primary = _policy_block(as_treated.frame, cfg)
    itt_block = _policy_block(itt.frame, cfg)

    # ITT agreement: same recommendation family for ship vs hold
    from ab_test.scorecard import decide

    rec_p, _ = decide(
        significant=primary["test"].significant,
        abs_diff=primary["test"].diff,
        srm_severe=primary["srm"].severe,
        underpowered_for_mde=primary["power"].underpowered_for_mde,
        practical_abs_pp=cfg.practical_abs_pp,
        mde_abs_pp=cfg.mde_abs_pp,
        power_mde=primary["power"].power_mde,
        power_target=cfg.power_target,
    )
    rec_i, _ = decide(
        significant=itt_block["test"].significant,
        abs_diff=itt_block["test"].diff,
        srm_severe=itt_block["srm"].severe,
        underpowered_for_mde=itt_block["power"].underpowered_for_mde,
        practical_abs_pp=cfg.practical_abs_pp,
        mde_abs_pp=cfg.mde_abs_pp,
        power_mde=itt_block["power"].power_mde,
        power_target=cfg.power_target,
    )
    itt_agree = "YES" if rec_p == rec_i else "NO"

    cuped_dict: dict[str, Any] | None = None
    if include_cuped_demo:
        y = as_treated.frame["converted"].to_numpy()
        t = (as_treated.frame["group"] == "treatment").to_numpy().astype(int)
        _df_c, cuped = simulate_preperiod_and_cuped(
            y, t, corr_strength=0.35, seed=cfg.random_seed
        )
        cuped_dict = cuped.to_dict()

    metrics: dict[str, Any] = {
        "version": "advanced_v3",
        "load_path": load_path,
        "cleaning_detail": (
            f"as-treated: raw={as_treated.n_raw:,}, mismatch_drop={as_treated.n_mismatch:,}, "
            f"dedup={as_treated.n_after_dedup:,}"
        ),
        "n_raw": as_treated.n_raw,
        "n_mismatch": as_treated.n_mismatch,
        "n_clean": as_treated.n_after_dedup,
        "n_itt": itt.n_after_dedup,
        "p_control": primary["summary"]["p_control"],
        "p_treatment": primary["summary"]["p_treatment"],
        "diff": primary["test"].diff,
        "zstat": primary["test"].zstat,
        "pvalue": primary["test"].pvalue,
        "ci_low": primary["test"].ci_low,
        "ci_high": primary["test"].ci_high,
        "chi2_pvalue": primary["test"].chi2_pvalue,
        "significant": primary["test"].significant,
        "bootstrap_ci_low": primary["bootstrap"].ci_low,
        "bootstrap_ci_high": primary["bootstrap"].ci_high,
        "srm_pvalue": primary["srm"].pvalue,
        "srm_imbalance_pp": primary["srm"].imbalance_pp,
        "srm_severe": primary["srm"].severe,
        "tost_equivalent": primary["tost"].equivalent,
        "tost_p_lower": primary["tost"].p_lower,
        "tost_p_upper": primary["tost"].p_upper,
        "tost_ci_inside_margin": primary["tost"].ci_inside_margin,
        "power_mde": primary["power"].power_mde,
        "power_obs": primary["power"].power_obs,
        "n_for_mde": primary["power"].n_for_mde,
        "underpowered_for_mde": primary["power"].underpowered_for_mde,
        "bayes_p_new_better": primary["bayes"].p_new_better,
        "bayes_ci_low": primary["bayes"].ci_low,
        "bayes_ci_high": primary["bayes"].ci_high,
        "itt_diff": itt_block["test"].diff,
        "itt_pvalue": itt_block["test"].pvalue,
        "itt_recommendation": rec_i,
        "primary_recommendation_pre_scorecard": rec_p,
        "itt_agreement": itt_agree,
        "itt_detail": (
            f"as-treated rec={rec_p}, ITT rec={rec_i}; "
            f"ITT diff={itt_block['test'].diff:.6f}, p={itt_block['test'].pvalue:.6f}"
        ),
        "cuped_demo": cuped_dict,
        "as_treated": {
            "policy": as_treated.policy,
            **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in primary["summary"].items()},
            "test": primary["test"].to_dict(),
            "srm": primary["srm"].to_dict(),
            "bootstrap": primary["bootstrap"].to_dict(),
            "tost": primary["tost"].to_dict(),
            "power": primary["power"].to_dict(),
            "bayes": primary["bayes"].to_dict(),
        },
        "itt": {
            "policy": itt.policy,
            **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in itt_block["summary"].items()},
            "test": itt_block["test"].to_dict(),
        },
    }
    scorecard = build_scorecard(metrics=metrics, cfg=cfg)
    metrics["recommendation"] = scorecard.recommendation
    metrics["reason"] = scorecard.reason
    return metrics, scorecard
