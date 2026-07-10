"""Pre-declared experiment analysis constants.

Change these in config (or CLI flags) *before* interpreting results — not after.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Decision and design parameters for the scorecard."""

    alpha: float = 0.05
    expected_split: float = 0.5
    srm_alpha: float = 0.001
    srm_severe_imbalance_pp: float = 0.5
    practical_abs_pp: float = 0.005  # ±0.5 pp indifference band
    mde_abs_pp: float = 0.01  # +1 pp absolute MDE for power
    power_target: float = 0.80
    bootstrap_reps: int = 2000
    bayes_mc: int = 50_000
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = AnalysisConfig()
