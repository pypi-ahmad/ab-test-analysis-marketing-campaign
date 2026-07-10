"""Decision policy and full scorecard assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ab_test.config import AnalysisConfig, DEFAULT_CONFIG


def decide(
    *,
    significant: bool,
    abs_diff: float,
    srm_severe: bool,
    underpowered_for_mde: bool,
    practical_abs_pp: float,
    mde_abs_pp: float,
    power_mde: float,
    power_target: float,
) -> tuple[str, str]:
    """Return (recommendation, reason)."""
    improved = abs_diff > 0
    meaningful = significant and abs_diff >= practical_abs_pp

    if srm_severe:
        return (
            "HOLD",
            "Severe SRM / trustworthiness failure — do not ship until diagnosed.",
        )
    if meaningful:
        return (
            "SHIP",
            f"Statistically significant improvement ≥ {practical_abs_pp * 100:.2f} pp "
            "and SRM not severe.",
        )
    if significant and not improved:
        return (
            "HOLD",
            "New creative is significantly worse or not better; keep the old page.",
        )
    if underpowered_for_mde:
        return (
            "EXTEND",
            f"Non-decisive for +{mde_abs_pp * 100:.2f} pp MDE "
            f"(power {power_mde:.1%} < {power_target:.0%}).",
        )
    return (
        "HOLD",
        "No significant improvement; experiment already powered for a business-relevant MDE. "
        "Keep the old campaign creative.",
    )


@dataclass
class Scorecard:
    recommendation: str
    reason: str
    gates: list[dict[str, str]]
    metrics: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def build_scorecard(
    *,
    metrics: dict[str, Any],
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> Scorecard:
    """Build gates from a flat metrics dict (see pipeline.analyze)."""
    rec, reason = decide(
        significant=bool(metrics["significant"]),
        abs_diff=float(metrics["diff"]),
        srm_severe=bool(metrics["srm_severe"]),
        underpowered_for_mde=bool(metrics["underpowered_for_mde"]),
        practical_abs_pp=cfg.practical_abs_pp,
        mde_abs_pp=cfg.mde_abs_pp,
        power_mde=float(metrics["power_mde"]),
        power_target=cfg.power_target,
    )
    gates = [
        {
            "gate": "Exposure cleaning",
            "status": "PASS",
            "detail": metrics.get("cleaning_detail", "ok"),
        },
        {
            "gate": "SRM (not severe)",
            "status": "PASS" if not metrics["srm_severe"] else "FAIL",
            "detail": (
                f"p={metrics['srm_pvalue']:.6g}, imbalance={metrics['srm_imbalance_pp']:.4f}pp"
            ),
        },
        {
            "gate": f"Primary OEC z-test (α={cfg.alpha})",
            "status": "SIGNIFICANT" if metrics["significant"] else "NOT SIGNIFICANT",
            "detail": f"z={metrics['zstat']:.4f}, p={metrics['pvalue']:.6f}",
        },
        {
            "gate": "Practical significance",
            "status": (
                "YES"
                if metrics["significant"] and metrics["diff"] >= cfg.practical_abs_pp
                else "NO"
            ),
            "detail": (
                f"diff={metrics['diff'] * 100:.4f}pp; "
                f"band=±{cfg.practical_abs_pp * 100:.2f}pp"
            ),
        },
        {
            "gate": "TOST equivalence",
            "status": "EQUIVALENT" if metrics.get("tost_equivalent") else "NOT EQUIVALENT",
            "detail": (
                f"margin=±{cfg.practical_abs_pp * 100:.2f}pp; "
                f"p_lo={metrics.get('tost_p_lower', float('nan')):.4g}, "
                f"p_hi={metrics.get('tost_p_upper', float('nan')):.4g}"
            ),
        },
        {
            "gate": f"Power for +{cfg.mde_abs_pp * 100:.0f}pp MDE",
            "status": "PASS" if not metrics["underpowered_for_mde"] else "FAIL",
            "detail": f"power={metrics['power_mde']:.1%}",
        },
        {
            "gate": "Bayesian P(new > old)",
            "status": f"{metrics.get('bayes_p_new_better', float('nan')):.1%}",
            "detail": "Uniform Beta–Binomial secondary check",
        },
        {
            "gate": "ITT sensitivity agrees on HOLD/SHIP",
            "status": metrics.get("itt_agreement", "n/a"),
            "detail": metrics.get("itt_detail", ""),
        },
        {"gate": "RECOMMENDATION", "status": rec, "detail": reason},
    ]
    return Scorecard(
        recommendation=rec,
        reason=reason,
        gates=gates,
        metrics=metrics,
        config=cfg.to_dict(),
    )
