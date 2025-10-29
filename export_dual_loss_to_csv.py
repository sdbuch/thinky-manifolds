import argparse
from datetime import datetime

import pandas as pd
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Export fc1.weight/e0_d00/dual_loss plot data to CSV."
)
parser.add_argument(
    "--after",
    type=str,
    default=None,
    help='Filter runs after this datetime (format: YYYYMMDD-HHMMSS, e.g., "20251006-143000")',
)
parser.add_argument(
    "--before",
    type=str,
    default=None,
    help='Filter runs before this datetime (format: YYYYMMDD-HHMMSS, e.g., "20251006-143000")',
)
parser.add_argument(
    "--run-ids",
    type=str,
    nargs="+",
    default=None,
    help="Specific run IDs to export (space-separated list, if not specified will use all filtered runs)",
)
parser.add_argument(
    "--trace",
    type=str,
    default="fc1.weight/e0_d00/dual_loss",
    help="WandB metric key to export (default: 'fc1.weight/e0_d00/dual_loss')",
)
args = parser.parse_args()

# Initialize wandb API with increased timeout for slow connections
api = wandb.Api(timeout=120)

# Build filter
filters = {}
if args.after:
    # Parse the datetime string (format: YYYYMMDD-HHMMSS)
    dt = datetime.strptime(args.after, "%Y%m%d-%H%M%S")
    # Convert to ISO format for wandb API
    filters["created_at"] = {"$gt": dt.isoformat()}
    print(f"Filtering runs created after: {dt}")
if args.before:
    # Parse the datetime string (format: YYYYMMDD-HHMMSS)
    dt = datetime.strptime(args.before, "%Y%m%d-%H%M%S")
    # Convert to ISO format for wandb API
    if "created_at" in filters:
        filters["created_at"]["$lt"] = dt.isoformat()
    else:
        filters["created_at"] = {"$lt": dt.isoformat()}
    print(f"Filtering runs created before: {dt}")

# Get runs
if args.run_ids:
    # Get specific runs
    print(f"Loading {len(args.run_ids)} specific run(s)")
    runs = []
    for run_id in args.run_ids:
        try:
            runs.append(api.run(f"thinky-manifolds/{run_id}"))
            print(f"  -> Loaded run: {run_id}")
        except Exception as e:
            print(f"  -> Error loading run {run_id}: {e}")
else:
    # Get filtered runs from the project
    print(f"Loading runs from thinky-manifolds project...")
    runs = api.runs("thinky-manifolds", filters=filters if filters else None)

# Process each run and create individual CSV files
metric_key = args.trace
successful_exports = 0

# Extract a clean name for the metric column from the trace
# e.g., "fc1.weight/e0_d00/dual_loss" -> "dual_loss"
# or "fc1.weight/e0_d00/primal_residual" -> "primal_residual"
metric_column_name = metric_key.split("/")[-1]

for i, run in enumerate(runs):
    print(f"\nProcessing run {i + 1}: {run.name} (ID: {run.id})")

    # Get the history (time-series data) for this run
    history = run.history(keys=[metric_key, "_step"], pandas=True)

    # Check if the metric exists in this run
    if metric_key not in history.columns or history[metric_key].isna().all():
        print(f"  -> Skipping: {metric_key} not found in this run")
        continue

    # Filter out rows where the metric is NaN
    history = history[[metric_key, "_step"]].dropna()

    if len(history) == 0:
        print(f"  -> Skipping: No data points found")
        continue

    print(f"  -> Found {len(history)} data points")

    # Add run metadata
    history["run_id"] = run.id
    history["run_name"] = run.name
    history["trace"] = metric_key  # Store the WandB metric key

    # Add config parameters that might be useful for grouping
    history["manifold_steps"] = run.config.get("manifold_steps")
    history["admm_rho"] = run.config.get("admm_rho")
    history["lr"] = run.config.get("initial_lr")
    history["update_rule"] = run.config.get("update_rule")

    # Rename columns for clarity
    history = history.rename(columns={metric_key: metric_column_name, "_step": "step"})

    # Sort by step
    history = history.sort_values("step")

    # Save to CSV with run_id as filename
    output_file = f"{run.id}.csv"
    history.to_csv(output_file, index=False)
    print(f"  -> Exported {len(history)} data points to {output_file}")
    successful_exports += 1

# Summary
if successful_exports == 0:
    print("\nNo data found for the specified metric!")
    exit(1)

print(f"\n{'=' * 60}")
print(f"Successfully exported {successful_exports} run(s)")
