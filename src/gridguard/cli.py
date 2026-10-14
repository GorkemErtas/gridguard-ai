"""Console commands for the complete reproducible demo."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from gridguard.data import load_consumption
from gridguard.synthetic import save_synthetic
from gridguard.training import score, train


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
    training.add_argument("--source", choices=["sgcc", "custom", "synthetic"], default="custom")
    training.add_argument("--seed", type=int, default=42)

    scoring = subcommands.add_parser("score", help="Rank all customers with saved model")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--model", default="artifacts/model.joblib")
    scoring.add_argument("--output", default="reports/risk_ranking.csv")

    exporting = subcommands.add_parser("export-sqlite", help="Explore daily readings using SQL")
    exporting.add_argument("--input", required=True)
    exporting.add_argument("--output", default="data/demo.sqlite")
    exporting.add_argument("--limit-customers", type=int, default=200)

    args = parser.parse_args(argv)
    if args.command == "demo":
        path = save_synthetic(
            args.output, customers=args.customers, days=args.days, seed=args.seed
        )
        result = train(path, output_dir=args.output_dir, source="synthetic", seed=args.seed)
        print("SYNTHETIC DATA ONLY - not indicative of operational detection quality")
        print(json.dumps({"selected_model": result["selected_model"], "test": result["test"]}, indent=2))
    elif args.command == "train":
        result = train(
            args.input,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            source=args.source,
            seed=args.seed,
        )
        print(json.dumps({"selected_model": result["selected_model"], "test": result["test"]}, indent=2))
    elif args.command == "score":
        ranked = score(args.input, args.model, args.output)
        print(f"Scored {len(ranked)} customers; saved ranking to {args.output}")
    elif args.command == "export-sqlite":
        count = export_sqlite(args.input, args.output, args.limit_customers)
        print(f"Exported readings for {count} customers to {args.output}")


if __name__ == "__main__":
    main()
