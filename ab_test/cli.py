"""Command-line interface for the advanced A/B analysis package.

Examples
--------
uv run ab-test run
uv run ab-test scorecard
uv run ab-test simulate
uv run ab-test report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ab_test.config import AnalysisConfig
from ab_test.data import project_root_from
from ab_test.pipeline import analyze
from ab_test.report import write_html_report, write_metrics_json
from ab_test.simulation import run_default_suite


def _cfg_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        alpha=args.alpha,
        practical_abs_pp=args.practical_pp,
        mde_abs_pp=args.mde_pp,
        power_target=args.power_target,
        bootstrap_reps=args.bootstrap_reps,
        bayes_mc=args.bayes_mc,
        random_seed=args.seed,
    )


def cmd_run(args: argparse.Namespace) -> int:
    root = project_root_from()
    cfg = _cfg_from_args(args)
    metrics, scorecard = analyze(
        prefer_local=not args.refresh_data,
        cfg=cfg,
        include_cuped_demo=not args.no_cuped,
    )
    metrics_path = Path(args.metrics_out) if args.metrics_out else root / "metrics" / "latest.json"
    write_metrics_json(metrics, metrics_path)

    print("=== Advanced A/B pipeline (v3) ===")
    print(f"Load path: {metrics['load_path']}")
    print(f"Clean n (as-treated): {metrics['n_clean']:,}")
    print(
        f"Rates: control={metrics['p_control']:.4%}  treatment={metrics['p_treatment']:.4%}  "
        f"diff={metrics['diff'] * 100:+.4f} pp"
    )
    print(f"z={metrics['zstat']:.4f}  p={metrics['pvalue']:.6f}")
    print(
        f"CI=[{metrics['ci_low']:.6f}, {metrics['ci_high']:.6f}]  "
        f"boot=[{metrics['bootstrap_ci_low']:.6f}, {metrics['bootstrap_ci_high']:.6f}]"
    )
    print(
        f"SRM p={metrics['srm_pvalue']:.6g}  imbalance={metrics['srm_imbalance_pp']:.4f}pp  "
        f"severe={metrics['srm_severe']}"
    )
    print(
        f"TOST equivalent (±{cfg.practical_abs_pp * 100:.2f}pp): {metrics['tost_equivalent']}  "
        f"(p_lo={metrics['tost_p_lower']:.4g}, p_hi={metrics['tost_p_upper']:.4g})"
    )
    print(f"Bayesian P(new>old)={metrics['bayes_p_new_better']:.2%}")
    print(f"ITT: rec={metrics['itt_recommendation']}  agree={metrics['itt_agreement']}")
    if metrics.get("cuped_demo"):
        c = metrics["cuped_demo"]
        print(
            f"CUPED demo: var reduction={c['var_reduction_pct']:.2f}%  "
            f"SE reduction={c['se_reduction_pct']:.2f}%  (synthetic pre-period)"
        )
    print(f"RECOMMENDATION: {scorecard.recommendation}")
    print(f"Reason: {scorecard.reason}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


def cmd_scorecard(args: argparse.Namespace) -> int:
    root = project_root_from()
    cfg = _cfg_from_args(args)
    metrics, scorecard = analyze(prefer_local=not args.refresh_data, cfg=cfg)
    print(f"{'Gate':40} {'Status':18} Detail")
    print("-" * 100)
    for g in scorecard.gates:
        print(f"{g['gate'][:40]:40} {g['status'][:18]:18} {g['detail']}")
    out = Path(args.metrics_out) if args.metrics_out else root / "metrics" / "latest.json"
    write_metrics_json(metrics, out)
    print(f"\nWrote {out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    results = run_default_suite(
        p_control=args.p_control,
        n_per_arm=args.n_per_arm,
        cfg=cfg,
    )
    print(
        f"{'scenario':32} {'true_lift':>10} {'obs_diff':>10} {'p':>10} {'sig':>5} {'rec':>6} "
        f"{'null_ok':>8} {'ship_ok':>8}"
    )
    for r in results:
        print(
            f"{r.name:32} {r.true_lift * 100:9.2f}pp {r.observed_diff * 100:9.3f}pp "
            f"{r.pvalue:10.4f} {str(r.significant):>5} {r.recommendation:>6} "
            f"{str(r.correct_hold_when_null):>8} {str(r.correct_ship_when_large):>8}"
        )
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([r.to_dict() for r in results], indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    root = project_root_from()
    cfg = _cfg_from_args(args)
    metrics, scorecard = analyze(prefer_local=not args.refresh_data, cfg=cfg)
    metrics_path = Path(args.metrics_out) if args.metrics_out else root / "metrics" / "latest.json"
    html_path = Path(args.html_out) if args.html_out else root / "artifacts" / "decision_report.html"
    write_metrics_json(metrics, metrics_path)
    write_html_report(scorecard, html_path)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {html_path}")
    print(f"Recommendation: {scorecard.recommendation}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ab-test",
        description="Advanced production A/B analysis CLI (baseline notebook left intact).",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--practical-pp", type=float, default=0.005, help="Equivalence margin (abs pp)")
    p.add_argument("--mde-pp", type=float, default=0.01, help="Business MDE absolute pp")
    p.add_argument("--power-target", type=float, default=0.80)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bayes-mc", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-data", action="store_true", help="Re-download CSVs instead of cache")
    p.add_argument("--no-cuped", action="store_true")
    p.add_argument("--metrics-out", type=str, default="")

    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Full advanced analysis + metrics.json")
    run_p.set_defaults(func=cmd_run)

    sc_p = sub.add_parser("scorecard", help="Print production scorecard gates")
    sc_p.set_defaults(func=cmd_scorecard)

    sim_p = sub.add_parser("simulate", help="Simulation harness (known true lifts)")
    sim_p.add_argument("--p-control", type=float, default=0.12)
    sim_p.add_argument("--n-per-arm", type=int, default=145_000)
    sim_p.set_defaults(func=cmd_simulate)

    rep_p = sub.add_parser("report", help="Write metrics.json + HTML decision memo")
    rep_p.add_argument("--html-out", type=str, default="")
    rep_p.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
