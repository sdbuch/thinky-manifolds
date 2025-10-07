import argparse
from datetime import datetime

import pandas as pd
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description="Load wandb results filtered by datetime.")
parser.add_argument(
    "--after",
    type=str,
    default=None,
    help='Filter runs after this datetime (format: YYYYMMDD-HHMMSS, e.g., "20251006-143000")',
)
args = parser.parse_args()

# Initialize wandb API
api = wandb.Api()

# Build filter
filters = {}
if args.after:
    # Parse the datetime string (format: YYYYMMDD-HHMMSS)
    dt = datetime.strptime(args.after, "%Y%m%d-%H%M%S")
    # Convert to ISO format for wandb API
    filters["created_at"] = {"$gt": dt.isoformat()}
    print(f"Filtering runs created after: {dt}")

# Get filtered runs from the project
runs = api.runs("thinky-manifolds", filters=filters if filters else None)

# Collect data from all runs
data = []
for run in runs:
    # Debug: print first run's structure
    if len(data) == 0:
        print(f"\nDebug info for first run:")
        print(f"Run state: {run.state}")
        print(f"Run ID: {run.id}")
        print(f"Config type: {type(run.config)}")
        print(f"Summary type: {type(run.summary)}")
        # Try to access the run's full data
        try:
            full_run = api.run(f"thinky-manifolds/{run.id}")
            print(f"Full run config type: {type(full_run.config)}")
            print(f"Full run summary type: {type(full_run.summary)}")
            if isinstance(full_run.config, dict):
                print(f"Config keys: {list(full_run.config.keys())[:5]}")
            if hasattr(full_run.summary, "_json_dict") and isinstance(
                full_run.summary._json_dict, dict
            ):
                print(f"Summary keys: {list(full_run.summary._json_dict.keys())[:5]}")
        except Exception as e:
            print(f"Error accessing full run: {e}")
        print()

    # Try to get the full run data
    try:
        full_run = api.run(f"thinky-manifolds/{run.id}")
        run_config = full_run.config if isinstance(full_run.config, dict) else {}
        if hasattr(full_run.summary, "_json_dict") and isinstance(
            full_run.summary._json_dict, dict
        ):
            summary = full_run.summary._json_dict
        else:
            summary = {}
    except Exception:
        run_config = {}
        summary = {}

    # Get config parameters
    config = {
        "run_name": run.name,
        "state": run.state,
        "manifold_steps": run_config.get("manifold_steps"),
        "admm_rho": run_config.get("admm_rho"),
        "lr": run_config.get("initial_lr"),
        "epochs": run_config.get("epochs"),
        "update_rule": run_config.get("update_rule"),
    }

    # Get final metrics
    config.update(
        {
            "test_accuracy": summary.get("final/test_accuracy"),
            "train_accuracy": summary.get("final/train_accuracy"),
            "test_loss": summary.get("final/test_loss"),
            "train_loss": summary.get("final/train_loss"),
        }
    )

    data.append(config)

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort by manifold_steps and admm_rho for easier viewing
df = df.sort_values(["manifold_steps", "admm_rho"])

# Display the table
print(df.to_string(index=False))

# Save to CSV
df.to_csv("wandb_results.csv", index=False)
print("\nResults saved to wandb_results.csv")
