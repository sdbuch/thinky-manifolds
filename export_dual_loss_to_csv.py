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
    "--output",
    type=str,
    default="dual_loss_data.csv",
    help="Output CSV filename (default: dual_loss_data.csv)",
)
parser.add_argument(
    "--run-id",
    type=str,
    default=None,
    help="Specific run ID to export (if not specified, will use all filtered runs)",
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
if args.run_id:
    # Get specific run
    print(f"Loading specific run: {args.run_id}")
    runs = [api.run(f"thinky-manifolds/{args.run_id}")]
else:
    # Get filtered runs from the project
    print(f"Loading runs from thinky-manifolds project...")
    runs = api.runs("thinky-manifolds", filters=filters if filters else None)

# Collect data from all runs
all_data = []
metric_key = "fc1.weight/e0_d00/dual_loss"

for i, run in enumerate(runs):
    print(f"Processing run {i + 1}: {run.name} (ID: {run.id})")

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

    # Add config parameters that might be useful for grouping
    history["manifold_steps"] = run.config.get("manifold_steps")
    history["admm_rho"] = run.config.get("admm_rho")
    history["lr"] = run.config.get("initial_lr")
    history["update_rule"] = run.config.get("update_rule")

    # Rename columns for clarity
    history = history.rename(columns={metric_key: "dual_loss", "_step": "step"})

    all_data.append(history)

# Combine all data
if not all_data:
    print("\nNo data found for the specified metric!")
    exit(1)

df = pd.DataFrame(pd.concat(all_data, ignore_index=True))

# Sort by run and step
df = df.sort_values(["run_id", "step"])

# Save to CSV
df.to_csv(args.output, index=False)
print(f"\nExported {len(df)} data points to {args.output}")
print(f"Number of runs: {df['run_id'].nunique()}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(10))
