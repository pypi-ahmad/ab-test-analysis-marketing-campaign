"""Exposure cleaning and policy variants (as-treated vs ITT)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CleanResult:
    frame: pd.DataFrame
    n_raw: int
    n_after_policy: int
    n_after_dedup: int
    n_mismatch: int
    policy: str


def count_mismatches(ab: pd.DataFrame) -> int:
    consistent = (
        ((ab["group"] == "control") & (ab["landing_page"] == "old_page"))
        | ((ab["group"] == "treatment") & (ab["landing_page"] == "new_page"))
    )
    return int((~consistent).sum())


def clean_as_treated(ab: pd.DataFrame) -> CleanResult:
    """Drop group/page mismatches, then one row per user (primary analysis)."""
    n_raw = len(ab)
    consistent = (
        ((ab["group"] == "control") & (ab["landing_page"] == "old_page"))
        | ((ab["group"] == "treatment") & (ab["landing_page"] == "new_page"))
    )
    n_mismatch = int((~consistent).sum())
    out = ab.loc[consistent].copy()
    n_after = len(out)
    out = out.drop_duplicates(subset="user_id", keep="first")
    return CleanResult(
        frame=out,
        n_raw=n_raw,
        n_after_policy=n_after,
        n_after_dedup=len(out),
        n_mismatch=n_mismatch,
        policy="as_treated_drop_mismatches",
    )


def clean_itt_by_assignment(ab: pd.DataFrame) -> CleanResult:
    """Intent-to-treat: keep all rows; analyze by randomized ``group`` only.

    Landing-page mismatches stay in the data. Exposure may be noisy, but the
    arm is defined by assignment — a standard sensitivity when instrumentation
    is imperfect.
    """
    n_raw = len(ab)
    n_mismatch = count_mismatches(ab)
    out = ab.copy()
    out = out.drop_duplicates(subset="user_id", keep="first")
    return CleanResult(
        frame=out,
        n_raw=n_raw,
        n_after_policy=n_raw,
        n_after_dedup=len(out),
        n_mismatch=n_mismatch,
        policy="itt_by_assignment",
    )


def arm_counts(df: pd.DataFrame) -> dict[str, int]:
    vc = df["group"].value_counts()
    return {
        "n_control": int(vc.get("control", 0)),
        "n_treatment": int(vc.get("treatment", 0)),
    }


def conversion_summary(df: pd.DataFrame) -> dict[str, float | int]:
    ctrl = df.loc[df["group"] == "control", "converted"]
    trt = df.loc[df["group"] == "treatment", "converted"]
    p_c, p_t = float(ctrl.mean()), float(trt.mean())
    return {
        "n_control": int(ctrl.shape[0]),
        "n_treatment": int(trt.shape[0]),
        "conv_control": int(ctrl.sum()),
        "conv_treatment": int(trt.sum()),
        "p_control": p_c,
        "p_treatment": p_t,
        "diff": p_t - p_c,
        "rel_diff": (p_t - p_c) / p_c if p_c else float("nan"),
    }
