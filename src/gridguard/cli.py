"""Console commands for the complete reproducible demo."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from gridguard.data import load_consumption
from gridguard.features import FEATURE_CONFIGS
from gridguard.synthetic import save_synthetic
from gridguard.training import experiment, finalize, score, train


def export_sqlite(source: str, destination: str, limit_customers: int) -> int:
    """Small-scale long-format export for learning SQL; intentionally bounded."""
    if limit_customers < 1 or limit_customers > 1000:
        raise ValueError("Choose 1 to 1,000 customers to keep the SQLite demo bounded.")
    data = load_consumption(source, require_labels=False)
    count = min(limit_customers, len(data))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE IF EXISTS consumption")
        conn.execute(
            "CREATE TABLE consumption (customer_id TEXT NOT NULL, "
            "reading_date TEXT NOT NULL, kwh REAL NOT NULL, "
            "PRIMARY KEY (customer_id, reading_date))"
        )
        dates = list(data.readings.columns)
        batch = []
        for customer_id, values in zip(
            data.customer_ids.iloc[:count], data.readings.iloc[:count].to_numpy()
        ):
            for date, value in zip(dates, values):
                if pd.notna(value):
                    batch.append((str(customer_id), date, float(value)))
            if len(batch) >= 10_000:
                conn.executemany("INSERT INTO consumption VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO consumption VALUES (?, ?, ?)", batch)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consumption_date ON consumption(reading_date)")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gridguard", description="Electricity loss risk-ranking portfolio demo"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="Generate synthetic CSV, train and evaluate")
    demo.add_argument("--output", default="data/demo.csv")
    demo.add_argument("--output-dir", default="artifacts")
    demo.add_argument("--customers", type=int, default=600)
    demo.add_argument("--days", type=int, default=120)
    demo.add_argument("--seed", type=int, default=42)

    training = subcommands.add_parser("train", help="Train and evaluate labeled data")
    training.add_argument("--input", required=True)
    training.add_argument("--output-dir", default="artifacts")
    training.add_argument("--sample-size", type=int)
    training.add_argument("--source", choices=["custom", "synthetic"], default="custom")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--feature-config", choices=FEATURE_CONFIGS, default="baseline")
    training.add_argument("--experiment-id")
    training.add_argument(
        "--validation-only",
        action="store_true",
        help="Select on validation without test predictions or metrics",
    )

    experiments = subcommands.add_parser("experiment", help="Run four validation-only ablations")
    experiments.add_argument("--input", required=True)
    experiments.add_argument("--output-dir", required=True)
    experiments.add_argument("--sample-size", type=int)
    experiments.add_argument("--source", choices=["custom", "synthetic"], default="custom")
    experiments.add_argument("--seed", type=int, default=42)
    experiments.add_argument("--experiment-id")

    finalizing = subcommands.add_parser("finalize", help="Evaluate the frozen validation winner")
    finalizing.add_argument("--input", required=True)
    finalizing.add_argument("--experiment-dir", required=True)
    finalizing.add_argument("--output-dir", required=True)
    finalizing.add_argument("--experiment-id")

    calibration = subcommands.add_parser(
        "calibrate", help="Development-only calibration; exclude the recorded final holdout"
    )
    calibration.add_argument("--input", required=True)
    calibration.add_argument("--reference-dir", required=True)
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--seed", type=int, default=42)
    calibration.add_argument("--feature-config", choices=FEATURE_CONFIGS, default="baseline")
    calibration.add_argument("--n-bins", type=int, default=10)
    calibration.add_argument("--experiment-id")

    anomaly = subcommands.add_parser(
        "benchmark-anomaly", help="Fixed development-only anomaly comparison"
    )
    anomaly.add_argument("--input", required=True)
    anomaly.add_argument("--reference-dir", required=True)
    anomaly.add_argument("--output-dir", required=True)
    anomaly.add_argument("--seed", type=int, default=42)
    anomaly.add_argument("--feature-config", choices=FEATURE_CONFIGS, default="baseline")
    anomaly.add_argument("--capacities", type=int, nargs="+", default=[50, 100, 200])
    anomaly.add_argument("--min-group-size", type=int, default=20)
    anomaly.add_argument("--experiment-id")

    explaining = subcommands.add_parser(
        "explain", help="Development-only permutation importance and error analysis"
    )
    explaining.add_argument("--input", required=True)
    explaining.add_argument("--reference-dir", required=True)
    explaining.add_argument("--output-dir", required=True)
    explaining.add_argument("--seed", type=int, default=42)
    explaining.add_argument("--feature-config", choices=FEATURE_CONFIGS, default="baseline")
    explaining.add_argument("--n-repeats", type=int, default=10)
    explaining.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    explaining.add_argument("--capacities", type=int, nargs="+", default=[50, 100, 200])
    explaining.add_argument("--min-group-size", type=int, default=20)
    explaining.add_argument("--experiment-id")

    scoring = subcommands.add_parser("score", help="Rank all customers with saved model")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--model", default="artifacts/model.joblib")
    scoring.add_argument("--output", default="reports/risk_ranking.csv")

    exporting = subcommands.add_parser("export-sqlite", help="Explore daily readings using SQL")
    exporting.add_argument("--input", required=True)
    exporting.add_argument("--output", default="data/demo.sqlite")
    exporting.add_argument("--limit-customers", type=int, default=200)

    context = subcommands.add_parser(
        "generate-context", help="Generate independent SYNTHETIC context for the exact SGCC CSV"
    )
    context.add_argument("--input", required=True)
    context.add_argument("--output-dir", required=True)
    context.add_argument("--reference-metrics", help="Verify the existing experiment's source SHA256")
    context.add_argument("--seed", type=int)
    context.add_argument("--regions", type=int)
    context.add_argument("--config", help="JSON GenerationConfig settings; explicit flags override")

    gate = subcommands.add_parser("context-gate", help="Annotate saved scores with SYNTHETIC context")
    gate.add_argument("--predictions", required=True)
    gate.add_argument("--context-dir", required=True)
    gate.add_argument("--output-dir", required=True)
    gate.add_argument("--evaluation-date", required=True)
    gate.add_argument("--window-start", required=True)
    gate.add_argument("--window-end", required=True)
    gate.add_argument("--config", help="JSON GateConfig settings")

    args = parser.parse_args(argv)
    if args.command == "generate-context":
        from gridguard.operational import GenerationConfig, save_context

        settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        for key in ("seed", "regions"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        result = save_context(
            args.input, output_dir=args.output_dir, config=GenerationConfig(**settings),
            reference_metrics=args.reference_metrics,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "context-gate":
        from gridguard.decision_gate import GateConfig, save_decisions

        settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        result = save_decisions(
            args.predictions, context_dir=args.context_dir, output_dir=args.output_dir,
            evaluation_date=args.evaluation_date, window_start=args.window_start,
            window_end=args.window_end, config=GateConfig(**settings),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "demo":
        path = save_synthetic(args.output, customers=args.customers, days=args.days, seed=args.seed)
        result = train(path, output_dir=args.output_dir, source="synthetic", seed=args.seed)
        print("SYNTHETIC DATA ONLY - not indicative of operational detection quality")
        print(
            json.dumps(
                {"selected_model": result["selected_model"], "test": result["test"]}, indent=2
            )
        )
    elif args.command == "train":
        result = train(
            args.input,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            source=args.source,
            seed=args.seed,
            feature_config=args.feature_config,
            experiment_id=args.experiment_id,
            validation_only=args.validation_only,
        )
        print(
            json.dumps(
                {key: result[key] for key in ("selected_model", "validation", "test")}, indent=2
            )
        )
    elif args.command == "experiment":
        result = experiment(
            args.input,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            source=args.source,
            seed=args.seed,
            experiment_id=args.experiment_id,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    elif args.command == "finalize":
        result = finalize(
            args.input,
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            json.dumps(
                {"selected_model": result["selected_model"], "test": result["test"]}, indent=2
            )
        )
    elif args.command == "calibrate":
        from gridguard.calibration import calibrate

        result = calibrate(
            args.input,
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            feature_config=args.feature_config,
            n_bins=args.n_bins,
            experiment_id=args.experiment_id,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "experiment_id",
                        "evaluation_scope",
                        "selected_model",
                        "selected_method",
                        "skipped_methods",
                    )
                },
                indent=2,
            )
        )
    elif args.command == "benchmark-anomaly":
        from gridguard.anomaly import benchmark

        result = benchmark(
            args.input,
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            feature_config=args.feature_config,
            capacities=args.capacities,
            min_group_size=args.min_group_size,
            experiment_id=args.experiment_id,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "evaluation_scope": result["evaluation_scope"],
                    "split_sizes": result["split_sizes"],
                },
                indent=2,
            )
        )
    elif args.command == "explain":
        from gridguard.explainability import explain

        result = explain(
            args.input,
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            feature_config=args.feature_config,
            n_repeats=args.n_repeats,
            thresholds=args.thresholds,
            capacities=args.capacities,
            min_group_size=args.min_group_size,
            experiment_id=args.experiment_id,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "evaluation_scope": result["evaluation_scope"],
                    "importance_customers": result["importance"]["customers"],
                    "error_customers": result["errors"]["customers"],
                },
                indent=2,
            )
        )
    elif args.command == "score":
        ranked = score(args.input, args.model, args.output)
        print(f"Scored {len(ranked)} customers; saved ranking to {args.output}")
    elif args.command == "export-sqlite":
        count = export_sqlite(args.input, args.output, args.limit_customers)
        print(f"Exported readings for {count} customers to {args.output}")


if __name__ == "__main__":
    main()
